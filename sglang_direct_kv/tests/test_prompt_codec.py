from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import json
import unittest
from unittest.mock import Mock

from agentic_prompt_codec import CodecConfig, EncodedSegment, PromptEncoder
from agentic_prompt_codec.adapters import PayloadAdapter
from agentic_prompt_codec.codecs.dictionary import DictionaryCodec
from agentic_prompt_codec.codecs.relations import RelationsCodec
from agentic_prompt_codec.validation import decode


class CharacterCounter:
    """Deliberate test double, never used as a production token estimate."""
    identity = "test-character-counter"

    def count_text(self, text):
        return len(text)

    def count_request(self, payload, api_kind):
        return len(json.dumps(payload, sort_keys=True))


PROSE = "The test failed because the expected value did not match the actual value. "


class PromptCodecTests(unittest.TestCase):
    def encoder(self, **kwargs):
        return PromptEncoder(CodecConfig(codec="dictionary_v1", max_encode_ms=5000, **kwargs), CharacterCounter())

    def test_identity_does_not_call_tokenizer_or_copy_payload(self):
        class ExplodingCounter(CharacterCounter):
            def count_request(self, *args):
                raise AssertionError("identity must not tokenize")
        payload = {"messages": [{"role": "user", "content": PROSE * 30}]}
        result = PromptEncoder(counter=ExplodingCounter()).encode(payload)
        self.assertIs(result.payload, payload)
        self.assertEqual(result.status, "unchanged")

    def test_dictionary_round_trip_preserves_literals_and_scopes_legend(self):
        text = PROSE * 30 + '\n```python\nprint("expected value")\n```\n' + "literal ~0~ and `file.py`"
        codec = DictionaryCodec()
        encoded = codec.encode(text, self.encoder().config, CharacterCounter(), lambda: None)
        self.assertEqual(decode(encoded), text)
        self.assertTrue(encoded.dictionary)
        self.assertNotIn("~0~", dict(encoded.dictionary))
        self.assertIn('```python\nprint("expected value")\n```', encoded.body)
        self.assertLess(len(encoded.text), len(text))

    def test_gateway_metadata_and_roles_survive_without_input_mutation(self):
        payload = {"messages": [{"role": "system", "content": PROSE * 20},
                                {"role": "user", "content": PROSE * 30},
                                {"role": "assistant", "tool_calls": [{"id": "a", "function": {"arguments": "{}"}}]},
                                {"role": "tool", "tool_call_id": "a", "content": PROSE * 30}],
                   "priority": 100, "cache_salt": "tenant", "custom_params": {"request_id": "r"},
                   "tools": [{"description": PROSE}], "extra": {"future": True}}
        before = copy.deepcopy(payload)
        result = self.encoder().encode(payload)
        self.assertEqual(payload, before)
        self.assertEqual(result.status, "applied")
        self.assertEqual(result.payload["messages"][0], payload["messages"][0])
        self.assertEqual(result.payload["messages"][2], payload["messages"][2])
        self.assertEqual(result.payload["messages"][3]["tool_call_id"], "a")
        for key in set(payload) - {"messages"}:
            self.assertEqual(result.payload[key], payload[key])

    def test_protocol_adapters_preserve_unknown_and_nontext_blocks(self):
        cases = [("anthropic", {"system": "trusted", "messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t", "content": [{"type": "text", "text": PROSE * 30, "cache_control": {"type": "ephemeral"}}]},
            {"type": "image", "source": {"data": "opaque"}}]}]}),
            ("responses", {"instructions": "trusted", "previous_response_id": "p", "input": [
                {"type": "function_call_output", "call_id": "c", "output": PROSE * 30},
                {"type": "reasoning", "encrypted_content": "opaque"}]} )]
        for kind, payload in cases:
            result = self.encoder().encode(payload, kind)
            self.assertEqual(result.status, "applied")
            adapter = PayloadAdapter(kind)
            restore = {s.path: s.text for s in adapter.extract(payload).segments if s.eligible}
            self.assertEqual(adapter.replace(adapter.extract(result.payload), restore), payload)

    def test_legend_cost_and_full_request_count_can_reject_candidate(self):
        payload = {"messages": [{"role": "user", "content": PROSE * 30}]}
        class RenderingCounter(CharacterCounter):
            def count_request(self, payload, api_kind):
                return 10_000 if "Local shorthand" in json.dumps(payload) else 100
        result = PromptEncoder(self.encoder().config, RenderingCounter()).encode(payload)
        self.assertEqual(result.reason, "insufficient_net_savings")
        self.assertIs(result.payload, payload)
        self.assertEqual(result.encoded_tokens, result.original_tokens)
        self.assertEqual(result.candidate_tokens, 10_000)

    def test_missing_tokenizer_budget_size_and_unsupported_api_bypass(self):
        payload = {"messages": [{"role": "user", "content": PROSE * 30}]}
        self.assertEqual(PromptEncoder(CodecConfig(codec="dictionary_v1")).encode(payload).reason, "tokenizer_unavailable")
        self.assertEqual(self.encoder().encode(payload, budget_ms=0).reason, "budget_exhausted")
        self.assertEqual(self.encoder(max_input_chars=5).encode(payload).reason, "input_limit")
        self.assertEqual(self.encoder().encode(payload, "unknown").status, "failed")

    def test_tampered_codec_falls_back(self):
        class BadCodec:
            name = "dictionary_v1"
            def encode(self, *args):
                return EncodedSegment("changed", "changed")
        payload = {"messages": [{"role": "user", "content": PROSE * 30}]}
        result = PromptEncoder(self.encoder().config, CharacterCounter(), BadCodec()).encode(payload)
        self.assertEqual(result.status, "failed")
        self.assertIs(result.payload, payload)

    def test_request_local_determinism_under_concurrency(self):
        encoder = self.encoder()
        texts = [PROSE * 30, "This other request contains an entirely different repeated explanation. " * 30] * 5
        def run(text):
            return encoder.encode(text, "text").payload
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(run, texts))
        self.assertTrue(all(value == results[i % 2] for i, value in enumerate(results)))

    def test_relations_do_not_rewrite_negation_uncertainty_or_code(self):
        text = "the cat is on the table.\n" * 20 + "the cat is not on the table.\nthe cat might be on the table.\n"
        result = RelationsCodec().encode(text, self.encoder().config, CharacterCounter(), lambda: None)
        self.assertEqual(decode(result), text)
        self.assertIn("the cat @ the table.", result.body)
        self.assertIn("the cat is not on the table.", result.body)
        self.assertIn("the cat might be on the table.", result.body)

    def test_very_short_request_stays_original(self):
        result = self.encoder().encode("the cat is on the table", "text")
        self.assertEqual(result.status, "unchanged")

    def test_chat_counter_requests_token_ids_not_batch_dictionary_length(self):
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
