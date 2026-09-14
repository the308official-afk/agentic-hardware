from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import json
import unittest
from unittest.mock import Mock

from agentic_prompt_codec import CodecConfig, EncodedSegment, PromptEncoder
from agentic_prompt_codec.adapters import PayloadAdapter
from agentic_prompt_codec.codecs.relations import AgentTraceRelationsCodec
from agentic_prompt_codec.validation import decode


class CharacterCounter:
    """Deliberate test double, never used as a production token estimate."""
    identity = "test-character-counter"

    def count_text(self, text):
        return len(text)

    def count_request(self, payload, api_kind):
        return len(json.dumps(payload, sort_keys=True))


PROSE = "The test failed because the expected value did not match the actual value. "
RELATION_PROSE = (
    "Read `src/database/mongo/main.js` to understand where scheduler records are stored.\n"
    "Run `execute(\"python -m pytest\")` to ensure tests pass.\n"
) * 80


class PromptCodecTests(unittest.TestCase):
    def encoder(self, **kwargs):
        return PromptEncoder(CodecConfig(codec="agent_trace_relations_v1", max_encode_ms=5000,
                                         min_saved_ratio=0.01, **kwargs), CharacterCounter())

    def test_identity_does_not_call_tokenizer_or_copy_payload(self):
        class ExplodingCounter(CharacterCounter):
            def count_request(self, *args):
                raise AssertionError("identity must not tokenize")
        payload = {"messages": [{"role": "user", "content": PROSE * 30}]}
        result = PromptEncoder(counter=ExplodingCounter()).encode(payload)
        self.assertIs(result.payload, payload)
        self.assertEqual(result.status, "unchanged")

    def test_agent_trace_relations_round_trip_preserves_protected_content(self):
        text = RELATION_PROSE + '\n```python\nprint("expected value")\n```\n' + "literal P(review) and `file.py`"
        codec = AgentTraceRelationsCodec()
        encoded = codec.encode(text, self.encoder().config, CharacterCounter(), lambda: None)
        self.assertEqual(decode(encoded), text)
        self.assertTrue(encoded.expansions)
        self.assertGreater(dict(encoded.rule_counts).get("read_for_purpose", 0), 0)
        self.assertGreater(dict(encoded.rule_counts).get("run_for_purpose", 0), 0)
        self.assertNotIn("P(review)", dict(encoded.expansions))
        self.assertIn('```python\nprint("expected value")\n```', encoded.body)
        self.assertLess(len(encoded.text), len(text))

    def test_enabled_rules_can_select_one_rule_family(self):
        text = "Phase: review\n" + RELATION_PROSE
        config = self.encoder(enabled_rules=("phase",)).config
        encoded = AgentTraceRelationsCodec().encode(text, config, CharacterCounter(), lambda: None)
        self.assertEqual(decode(encoded), text)
        self.assertIn("P(review)", encoded.body)
        self.assertIn("Read `src/database/mongo/main.js`", encoded.body)
        self.assertEqual(dict(encoded.rule_counts), {"phase": 1})

    def test_repeated_file_aliases_are_request_local_and_reversible(self):
        text = (
            "Read `src/database/mongo/main.js` to understand storage behavior.\n"
            "Inspect `src/database/mongo/main.js` to find scheduler writes.\n"
            "Modify `src/database/mongo/main.js` to record controller metadata.\n"
        )
        config = CodecConfig(codec="agent_trace_relations_v1", max_encode_ms=5000, min_saved_ratio=0.0,
                             enabled_rules=("file_path_alias", "read_for_purpose", "modify_to_goal"))
        encoded = AgentTraceRelationsCodec().encode(text, config, CharacterCounter(), lambda: None)
        self.assertEqual(decode(encoded), text)
        self.assertIn("F1=`src/database/mongo/main.js`", encoded.legend)
        self.assertIn("R(F1; understand storage behavior)", encoded.body)
        self.assertIn("M(F1; record controller metadata)", encoded.body)
        self.assertEqual(dict(encoded.expansions)["F1"], "`src/database/mongo/main.js`")
        self.assertEqual(dict(encoded.rule_counts)["file_path_alias"], 3)

    def test_repeated_function_aliases_are_request_local_and_reversible(self):
        text = (
            "Call normalize_url(input) before parsing.\n"
            "The normalize_url(input) helper should preserve empty hosts.\n"
            "Review normalize_url(input) test coverage.\n"
        )
        config = CodecConfig(codec="agent_trace_relations_v1", max_encode_ms=5000, min_saved_ratio=0.0,
                             enabled_rules=("function_alias",))
        encoded = AgentTraceRelationsCodec().encode(text, config, CharacterCounter(), lambda: None)
        self.assertEqual(decode(encoded), text)
        self.assertIn("FN1=normalize_url", encoded.legend)
        self.assertIn("Call FN1(input)", encoded.body)
        self.assertEqual(dict(encoded.expansions)["FN1"], "normalize_url")
        self.assertEqual(dict(encoded.rule_counts)["function_alias"], 3)

    def test_workspace_root_aliases_are_request_local_and_reversible(self):
        root = "/home/ec2-user/kv_cache_offloading/agentbench/repos/qutebrowser__qutebrowser"
        text = (
            f"Read `{root}/qutebrowser/utils/qtlog.py` before editing.\n"
            f"Run python -m pytest {root}/tests/unit/utils/test_qtlog.py after editing.\n"
            f"Inspect `{root}/tests/unit/utils/test_qtlog.py` for expected behavior.\n"
        )
        config = CodecConfig(codec="agent_trace_relations_v1", max_encode_ms=5000, min_saved_ratio=0.0,
                             enabled_rules=("workspace_root_alias", "file_path_alias"))
        encoded = AgentTraceRelationsCodec().encode(text, config, CharacterCounter(), lambda: None)
        self.assertEqual(decode(encoded), text)
        self.assertIn(f"WR1={root}", encoded.legend)
        self.assertIn("`WR1/qutebrowser/utils/qtlog.py`", encoded.body)
        self.assertEqual(dict(encoded.rule_counts)["workspace_root_alias"], 3)

    def test_validation_command_alias_and_tool_command_are_reversible(self):
        text = (
            "Validation command: python -m pytest tests/unit/utils/test_qtlog.py\n"
            "Selected command: python -m pytest tests/unit/utils/test_qtlog.py\n"
            "Use execute(\"read_file tests/unit/utils/test_qtlog.py\") next.\n"
        )
        config = CodecConfig(codec="agent_trace_relations_v1", max_encode_ms=5000, min_saved_ratio=0.0,
                             enabled_rules=("validation_command_alias", "execute_tool_command"))
        encoded = AgentTraceRelationsCodec().encode(text, config, CharacterCounter(), lambda: None)
        self.assertEqual(decode(encoded), text)
        self.assertIn("VC1=python -m pytest tests/unit/utils/test_qtlog.py", encoded.legend)
        self.assertIn("Validation command: VC1", encoded.body)
        self.assertIn("TC(read_file tests/unit/utils/test_qtlog.py)", encoded.body)
        counts = dict(encoded.rule_counts)
        self.assertEqual(counts["validation_command_alias"], 2)
        self.assertEqual(counts["execute_tool_command"], 1)

    def test_workflow_heading_shorthand_is_reversible(self):
        text = (
            "## planning\n"
            "### Step 1: Inspect F1\n"
            "1. **Identify Relevant Files**\n"
        )
        config = CodecConfig(codec="agent_trace_relations_v1", max_encode_ms=5000, min_saved_ratio=0.0,
                             enabled_rules=("markdown_phase_heading", "markdown_step_heading", "numbered_bold_step"))
        encoded = AgentTraceRelationsCodec().encode(text, config, CharacterCounter(), lambda: None)
        self.assertEqual(decode(encoded), text)
        self.assertIn("PH(planning)", encoded.body)
        self.assertIn("SH(1; Inspect F1)", encoded.body)
        self.assertIn("SB(1; Identify Relevant Files)", encoded.body)

    def test_gateway_metadata_and_roles_survive_without_input_mutation(self):
        payload = {"messages": [{"role": "system", "content": RELATION_PROSE},
                                {"role": "user", "content": RELATION_PROSE},
                                {"role": "assistant", "tool_calls": [{"id": "a", "function": {"arguments": "{}"}}]},
                                {"role": "tool", "tool_call_id": "a", "content": RELATION_PROSE}],
                   "priority": 100, "cache_salt": "tenant", "custom_params": {"request_id": "r"},
                   "tools": [{"description": PROSE}], "extra": {"future": True}}
        before = copy.deepcopy(payload)
        result = self.encoder().encode(payload)
        self.assertEqual(payload, before)
        self.assertEqual(result.status, "applied")
        self.assertGreater(result.rule_counts.get("read_for_purpose", 0), 0)
        self.assertEqual(result.payload["messages"][0], payload["messages"][0])
        self.assertEqual(result.payload["messages"][2], payload["messages"][2])
        self.assertEqual(result.payload["messages"][3]["tool_call_id"], "a")
        for key in set(payload) - {"messages"}:
            self.assertEqual(result.payload[key], payload[key])

    def test_protocol_adapters_preserve_unknown_and_nontext_blocks(self):
        cases = [("anthropic", {"system": "trusted", "messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t", "content": [{"type": "text", "text": RELATION_PROSE, "cache_control": {"type": "ephemeral"}}]},
            {"type": "image", "source": {"data": "opaque"}}]}]}),
            ("responses", {"instructions": "trusted", "previous_response_id": "p", "input": [
                {"type": "function_call_output", "call_id": "c", "output": RELATION_PROSE},
                {"type": "reasoning", "encrypted_content": "opaque"}]} )]
        for kind, payload in cases:
            result = self.encoder().encode(payload, kind)
            self.assertEqual(result.status, "applied")
            adapter = PayloadAdapter(kind)
            restore = {s.path: s.text for s in adapter.extract(payload).segments if s.eligible}
            self.assertEqual(adapter.replace(adapter.extract(result.payload), restore), payload)

    def test_legend_cost_and_full_request_count_can_reject_candidate(self):
        payload = {"messages": [{"role": "user", "content": RELATION_PROSE}]}
        class RenderingCounter(CharacterCounter):
            def count_request(self, payload, api_kind):
                return 10_000 if "Relational shorthand" in json.dumps(payload) else 100
        result = PromptEncoder(self.encoder().config, RenderingCounter()).encode(payload)
        self.assertEqual(result.reason, "insufficient_net_savings")
        self.assertIs(result.payload, payload)
        self.assertEqual(result.encoded_tokens, result.original_tokens)
        self.assertEqual(result.candidate_tokens, 10_000)

    def test_missing_tokenizer_budget_size_and_unsupported_api_bypass(self):
        payload = {"messages": [{"role": "user", "content": RELATION_PROSE}]}
        self.assertEqual(PromptEncoder(CodecConfig(codec="agent_trace_relations_v1")).encode(payload).reason, "tokenizer_unavailable")
        self.assertEqual(self.encoder().encode(payload, budget_ms=0).reason, "budget_exhausted")
        self.assertEqual(self.encoder(max_input_chars=5).encode(payload).reason, "input_limit")
        self.assertEqual(self.encoder().encode(payload, "unknown").status, "failed")

    def test_tampered_codec_falls_back(self):
        class BadCodec:
            name = "agent_trace_relations_v1"
            def encode(self, *args):
                return EncodedSegment("changed", "changed")
        payload = {"messages": [{"role": "user", "content": PROSE * 30}]}
        result = PromptEncoder(self.encoder().config, CharacterCounter(), BadCodec()).encode(payload)
        self.assertEqual(result.status, "failed")
        self.assertIs(result.payload, payload)

    def test_request_local_determinism_under_concurrency(self):
        encoder = self.encoder()
        texts = [RELATION_PROSE, "Phase: review\nTool calls: none\n" * 80] * 5
        def run(text):
            return encoder.encode(text, "text").payload
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(run, texts))
        self.assertTrue(all(value == results[i % 2] for i, value in enumerate(results)))

    def test_relations_do_not_rewrite_unsupported_negation_uncertainty_or_code(self):
        text = RELATION_PROSE + "The cat is not on the table.\nThe cat might be on the table.\n`exact_output.py`\n"
        result = AgentTraceRelationsCodec().encode(text, self.encoder().config, CharacterCounter(), lambda: None)
        self.assertEqual(decode(result), text)
        self.assertIn("R(F1; understand where scheduler records are stored)", result.body)
        self.assertIn("F1=`src/database/mongo/main.js`", result.legend)
        self.assertIn("The cat is not on the table.", result.body)
        self.assertIn("The cat might be on the table.", result.body)
        self.assertIn("`exact_output.py`", result.body)

    def test_very_short_request_stays_original(self):
        result = self.encoder().encode("the cat is on the table", "text")
        self.assertEqual(result.status, "unchanged")

    def test_chat_counter_requests_token_ids_not_mapping_length(self):
        from agentic_prompt_codec.tokenizers import HuggingFaceTokenCounter
        counter = object.__new__(HuggingFaceTokenCounter)
        counter.renderer = None
        counter.tokenizer = Mock()
        counter.tokenizer.apply_chat_template.return_value = [1, 2, 3, 4, 5]
        request = {"messages": [{"role": "user", "content": "hello"}]}
        self.assertEqual(counter.count_request(request, "openai_chat"), 5)
        self.assertFalse(counter.tokenizer.apply_chat_template.call_args.kwargs["return_dict"])

    def test_tokenizer_is_part_of_experiment_identity(self):
        first = self.encoder()
        second_counter = CharacterCounter()
        second_counter.identity = "another-tokenizer"
        second = PromptEncoder(first.config, second_counter)
        self.assertNotEqual(first.fingerprint, second.fingerprint)

    def test_summary_plugin_is_not_mislabeled_as_lossless(self):
        from agentic_prompt_codec.codecs.summary import SummaryCodec
        config = CodecConfig(codec="summary_v1", max_encode_ms=1000)
        encoder = PromptEncoder(config, CharacterCounter(), SummaryCodec(lambda text, timeout: "The test failed. "))
        result = encoder.encode(PROSE * 30, "text")
        self.assertEqual(result.status, "applied")
        self.assertEqual(result.validation, "protected_content_only")


if __name__ == "__main__":
    unittest.main()
