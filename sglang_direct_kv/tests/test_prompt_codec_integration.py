from __future__ import annotations

import base64
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import harness_sglang_gateway as gateway
import build_multi_harness_deadline_summary as report
import evaluate_prompt_codec as evaluation
from agentic_prompt_codec import CodecConfig, PromptEncoder
from agentic_prompt_codec.proxy import make_proxy_handler
from test_prompt_codec import CharacterCounter, PROSE


@contextmanager
def serve(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class IntegrationTests(unittest.TestCase):
    def backend(self, requests, *, release=None, empty=False):
        class Backend(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def do_POST(self):
                requests.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                if not empty:
                    self.wfile.write(b'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n')
                    self.wfile.flush()
                    if release:
                        release.wait(3)
                self.wfile.write(b'data: [DONE]\n\n')
                self.wfile.flush()
        return Backend

    def marked(self):
        meta = {"session_id": "s", "label": "s_replay", "phase": "replay", "mode": "controller_full",
                "controller_sglang_priority": 100, "harness": "hatcher", "prompt_hash": "driver-hash"}
        marker = base64.urlsafe_b64encode(json.dumps(meta).encode()).decode()
        return {"messages": [{"role": "user", "content": PROSE * 30 + "\nHARNESS_REPLAY_EXPERIMENT_JSON:" + marker}]}, meta

    def test_gateway_payload_and_explicit_first_content_time_include_encoding(self):
        requests = []
        encoder = PromptEncoder(CodecConfig(codec="dictionary_v1", max_encode_ms=5000), CharacterCounter())
        original_encode = encoder.encode
        def delayed(*args):
            time.sleep(0.025)
            return original_encode(*args)
        encoder.encode = delayed
        with tempfile.TemporaryDirectory() as folder, serve(self.backend(requests)) as backend:
            case = Path(folder) / "hatcher_p0_control_controller_full_tw500_f0"
            case.mkdir()
            trace, log = case / "m27_trace.jsonl", case / "harness_gateway_events.jsonl"
            gateway.write_jsonl(trace, {"event": "m27.replay.due", "session_id": "s"})
            for event in ("m27.controller_demote_restore.demote_start", "m27.controller_admission.decision"):
                gateway.write_jsonl(trace, {"event": event, "session_id": "s", "harness": "hatcher",
                                           "mode": "controller_full", "backend_acted": True,
                                           "decision": "skip", "reason": "full controller excludes warmup"})
            handler = gateway.make_handler(backend, trace, log, "test", encoder)
            handler.log_message = lambda *args: None
            payload, meta = self.marked()
            with serve(handler) as proxy:
                response = httpx.post(proxy + "/v1/chat/completions", json=payload)
            gateway.write_jsonl(trace, {"event": "m27.controller_demote_restore.restored", "session_id": "s", "backend_acted": True})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(requests[0]["priority"], 100)
            self.assertIn("Local shorthand", requests[0]["messages"][0]["content"])
            self.assertNotIn("HARNESS_REPLAY_EXPERIMENT_JSON", requests[0]["messages"][0]["content"])
            end = next(r for r in map(json.loads, trace.read_text().splitlines()) if r["event"] == "m27.request.end")
            self.assertGreater(end["gateway_to_first_content_ms"], end["ttft_ms"] + 20)
            self.assertEqual(end["encoding_status"], "applied")
            rows = report.collect_rows(Path(folder))
            self.assertEqual(rows[0]["first_token_ts_ns"], end["first_content_ts_ns"])
            self.assertEqual(rows[0]["first_token_source"], "gateway_first_content_timestamp")
            out = Path(folder) / "report"
            with patch.object(sys, "argv", ["report", "--root", folder, "--out-dir", str(out)]):
                report.main()
            self.assertIn("Prompt encoding", (out / "master_report.html").read_text())
            self.assertTrue((out / "prompt_encoding_proof.csv").exists())
            self.assertIn("CF = Full controller", (out / "controller_admission_proof.csv").read_text())
            self.assertIn("CF = Full controller", (out / "controller_demote_restore_proof.csv").read_text())
            with patch.object(sys, "argv", ["report", "--root", str(Path(folder)/"no-raw-traces"),
                                            "--out-dir", str(out), "--rows-csv", str(out/"global_kv_readiness_by_mode.csv")]):
                report.main()
            self.assertIn("applied", (out / "prompt_encoding_proof.csv").read_text())

    def test_disabled_gateway_preserves_legacy_outbound_payload(self):
        requests = []
        payload, meta = self.marked()
        expected = gateway.build_sglang_payload(payload, meta, "openai_chat", "test")
        with tempfile.TemporaryDirectory() as folder, serve(self.backend(requests)) as backend:
            handler = gateway.make_handler(backend, Path(folder)/"trace", Path(folder)/"log", "test")
            handler.log_message = lambda *args: None
            with serve(handler) as proxy:
                self.assertEqual(httpx.post(proxy + "/v1/chat/completions", json=payload).status_code, 200)
        self.assertEqual(requests, [expected])

    def test_empty_backend_is_not_counted_as_first_token(self):
        requests = []
        with serve(self.backend(requests, empty=True)) as backend:
            result = gateway.call_sglang(backend, {"messages": []})
        self.assertEqual(result[0], "")
        self.assertIsNone(result[1])
        self.assertIsNone(result[5])

    def test_cost_accounting_splits_target_and_filler_timing(self):
        with tempfile.TemporaryDirectory() as folder:
            case = Path(folder) / "hatcher_p3_high_controller_full_tw50_f1"
            case.mkdir()
            trace = case / "m27_trace.jsonl"
            rows = [
                {"event": "m27.replay.due", "session_id": "target", "ts_ns": 1_000_000_000},
                {
                    "event": "m27.request.start",
                    "session_id": "target",
                    "label": "target_replay",
                    "phase": "replay",
                    "mode": "controller_full",
                    "harness": "hatcher",
                    "ts_ns": 1_010_000_000,
                    "sglang_priority": 100,
                    "gateway_priority_translation_source": "controller_ready_decision",
                },
                {
                    "event": "m27.request.end",
                    "session_id": "target",
                    "label": "target_replay",
                    "phase": "replay",
                    "mode": "controller_full",
                    "harness": "hatcher",
                    "ts_ns": 1_080_000_000,
                    "first_content_ts_ns": 1_060_000_000,
                    "ttft_ms": 50,
                    "sglang_priority": 100,
                    "gateway_priority_translation_source": "controller_ready_decision",
                },
                {
                    "event": "m27.request.start",
                    "session_id": "target_pressure_000",
                    "label": "target_pressure_000_request",
                    "phase": "pressure_filler",
                    "mode": "controller_full",
                    "harness": "hatcher",
                    "ts_ns": 1_005_000_000,
                },
                {
                    "event": "m27.request.end",
                    "session_id": "target_pressure_000",
                    "label": "target_pressure_000_request",
                    "phase": "pressure_filler",
                    "mode": "controller_full",
                    "harness": "hatcher",
                    "ts_ns": 1_200_000_000,
                    "first_content_ts_ns": 1_155_000_000,
                    "ttft_ms": 150,
                },
            ]
            for row in rows:
                gateway.write_jsonl(trace, row)
            timing_rows = report.collect_rows(Path(folder))
            self.assertEqual([row["request_group"] for row in timing_rows], ["target", "filler"])
            chart_rows = report.target_replay_rows(timing_rows)
            self.assertEqual(len(chart_rows), 1)
            cost_rows = report.collect_cost_accounting_summary(timing_rows)
            self.assertEqual(len(cost_rows), 1)
            cost = cost_rows[0]
            self.assertEqual(cost["target_request_count"], 1)
            self.assertEqual(cost["filler_request_count"], 1)
            self.assertEqual(cost["sum_target_ttft_ms"], 50)
            self.assertEqual(cost["sum_filler_ttft_ms"], 150)
            self.assertEqual(cost["sum_total_ttft_ms"], 200)
            self.assertEqual(cost["sum_target_replay_debt_ms"], 60)
            self.assertEqual(cost["sum_filler_replay_debt_ms"], 0)
            self.assertEqual(cost["filler_replay_debt_unmeasured_requests"], 1)

    def test_streaming_proxy_delivers_first_chunk_before_backend_finishes(self):
        requests, events = [], []
        release = threading.Event()
        payload = {"messages": [{"role": "system", "content": "keep this"}, {"role": "user", "content": PROSE * 30}],
                   "stream": True, "priority": 100, "cache_salt": "same-tenant"}
        encoder = PromptEncoder(CodecConfig(codec="dictionary_v1", max_encode_ms=5000), CharacterCounter())
        with serve(self.backend(requests, release=release)) as backend:
            with serve(make_proxy_handler(backend, encoder, record=events.append)) as proxy:
                with httpx.stream("POST", proxy + "/v1/chat/completions", json=payload) as response:
                    for line in response.iter_lines():
                        if "answer" in line:
                            self.assertFalse(release.is_set())
                            release.set()
                            break
                release.set()
        self.assertEqual(requests[0]["messages"][0], payload["messages"][0])
        self.assertEqual(requests[0]["priority"], 100)
        self.assertEqual(events[0]["encoding_status"], "applied")
        self.assertIsNotNone(events[0]["first_client_content_ns"])

    def test_report_keeps_codec_arms_separate_and_retains_failures(self):
        base = {"harness": "hatcher", "pressure_level": "p0_control", "mode": "no_prefetch",
                "first_token_lateness_ms": 100, "ttft_ms": 100}
        rows = [dict(base, encoding_codec="identity", encoding_config_hash="a"),
                dict(base, encoding_codec="dictionary_v1", encoding_config_hash="b", first_token_lateness_ms=20),
                dict(base, encoding_codec="dictionary_v1", encoding_config_hash="b", first_token_lateness_ms="", error="empty")]
        summary = report.summarize(rows)
        self.assertEqual(len(summary), 2)
        candidate = next(row for row in summary if row["encoding_codec"] == "dictionary_v1")
        self.assertEqual(candidate["requests"], 2)
        self.assertEqual(candidate["failures"], 1)
        self.assertEqual(candidate["median_first_token_lateness_ms"], 20)

    def test_quality_evaluator_scores_both_arms_and_isolates_warmup(self):
        requests = []
        encoder = PromptEncoder(CodecConfig(codec="dictionary_v1", max_encode_ms=5000), CharacterCounter())
        with tempfile.TemporaryDirectory() as folder, serve(self.backend(requests)) as backend:
            workload = Path(folder) / "workload.jsonl"
            workload.write_text(json.dumps({"id": "quality", "replay_prompt": PROSE * 30, "expected_answer": "answer"}) + "\n")
            out = Path(folder) / "evaluation"
            args = ["evaluation", "--workload", str(workload), "--config", str(workload),
                    "--out-dir", str(out), "--base-url", backend, "--cache-condition", "warm"]
            with patch.object(sys, "argv", args), patch.object(evaluation, "load_encoder", return_value=encoder):
                evaluation.main()
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["identity"]["accuracy"], 1)
            self.assertEqual(summary["candidate"]["accuracy"], 1)
            self.assertEqual(len(requests), 4)
            self.assertEqual(requests[0]["cache_salt"], requests[1]["cache_salt"])
            self.assertEqual(requests[2]["cache_salt"], requests[3]["cache_salt"])
            self.assertNotEqual(requests[0]["cache_salt"], requests[2]["cache_salt"])


if __name__ == "__main__":
    unittest.main()
