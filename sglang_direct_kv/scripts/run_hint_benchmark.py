#!/usr/bin/env python3
"""Run harness hint benchmark scenarios."""
from __future__ import annotations

import argparse
import asyncio
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import shlex
import sys
import subprocess
import threading
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_kv.hint_benchmark import (
    HintBenchmarkConfigError,
    build_direct_api_payloads,
    build_dry_run,
    build_fixture_observations,
    build_nat_payload_observations,
    build_payload_observations,
    load_knob_profiles,
    load_observations_jsonl,
    load_benchmark_inputs,
    select_knob_profile,
    select_scenarios,
    validate_hint_evidence,
    write_dry_run_outputs,
)


DEFAULT_CONFIGS = {
    "nemo_agent_toolkit": {
        "manifest": REPO_ROOT / "configs" / "hint_benchmark" / "nat_hints.json",
        "scenarios": REPO_ROOT / "configs" / "hint_benchmark" / "nat_scenarios.json",
        "knobs": REPO_ROOT / "configs" / "hint_benchmark" / "nat_knobs.json",
    },
    "claude_code": {
        "manifest": REPO_ROOT / "configs" / "hint_benchmark" / "claude_hints.json",
        "scenarios": REPO_ROOT / "configs" / "hint_benchmark" / "claude_scenarios.json",
        "knobs": REPO_ROOT / "configs" / "hint_benchmark" / "claude_knobs.json",
    },
}
DEFAULT_OUT_ROOT = REPO_ROOT / "artifacts" / "results" / "hint_benchmark"


def scenario_workflow_metadata(scenario: dict[str, Any]) -> dict[str, Any]:
    setup = scenario.get("client_setup") or scenario.get("synthetic_setup", {})
    metadata = setup.get("workflow_metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def scenario_priority_sensitivity(scenario: dict[str, Any]) -> int:
    metadata = scenario_workflow_metadata(scenario)
    raw = metadata.get("latency_sensitivity")
    if raw is not None:
        if raw == "high":
            return 100
        if raw == "low":
            return 2
        return int(raw)
    raw_priority = metadata.get("priority") or scenario.get("workload_shape", {}).get("priority")
    if raw_priority == "high":
        return 100
    if raw_priority == "low":
        return 2
    return 2


def scenario_static_hint_config(scenario: dict[str, Any]) -> dict[str, Any]:
    metadata = scenario_workflow_metadata(scenario)
    cache_control = metadata.get("cache_control", {})
    if not isinstance(cache_control, dict):
        cache_control = {}
    return {
        "prefix_id": str(metadata.get("prefix_id") or metadata.get("workflow_id") or f"{scenario['id']}_prefix"),
        "total_requests": int(metadata.get("total_requests") or 1),
        "osl": int(metadata.get("osl") or 512),
        "iat": int(metadata.get("iat") or 300),
        "latency_sensitivity": scenario_priority_sensitivity(scenario),
        "cache_control_enabled": scenario.get("workload_shape", {}).get("cache_control") != "disabled",
        "cache_control_mode": str(cache_control.get("mode") or "always"),
    }


def deep_merge_dict(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def scenario_base_payload(scenario: dict[str, Any], index: int) -> dict[str, Any]:
    setup = scenario.get("client_setup") or scenario.get("synthetic_setup", {})
    payload = {
        "model": "nat-hint-benchmark-model",
        "messages": [
            {
                "role": "user",
                "content": f"NAT hint benchmark {scenario['id']} request {index + 1}.",
            }
        ],
        "max_tokens": 8,
        "temperature": 0,
        "stream": False,
    }
    client_metadata = setup.get("client_metadata", {})
    if isinstance(client_metadata, dict):
        payload = deep_merge_dict(payload, client_metadata)
    provider_metadata = setup.get("provider_metadata", {})
    if isinstance(provider_metadata, dict) and provider_metadata:
        payload["provider"] = provider_metadata
    return payload


async def capture_nat_dynamo_payloads(scenarios: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    try:
        import httpx
        from nat.builder.context import Context
        from nat.llm.dynamo_llm import CacheControlMode
        from nat.llm.dynamo_llm import CachePinType
        from nat.llm.dynamo_llm import DynamoPrefixContext
        from nat.llm.dynamo_llm import _DynamoTransport
    except ImportError as exc:
        raise HintBenchmarkConfigError(
            "NAT capture requires the NAT Python environment. Run with the NAT venv python, "
            "for example: .venvs/nat_py311/bin/python scripts/run_hint_benchmark.py "
            "--nat-dynamo-transport-capture ..."
        ) from exc

    class CaptureTransport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.payloads: list[dict[str, Any]] = []

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            body = await request.aread()
            payload = json.loads(body.decode() or "{}")
            self.payloads.append(payload)
            return httpx.Response(200, json={"ok": True, "choices": []}, request=request)

    captured: dict[str, list[dict[str, Any]]] = {}
    for scenario in scenarios:
        scenario_id = scenario["id"]
        config = scenario_static_hint_config(scenario)
        capture_transport = CaptureTransport()
        request_count = max(1, int(scenario.get("workload_shape", {}).get("request_count") or 1))

        if scenario_id == "nat_no_hints_baseline":
            async with httpx.AsyncClient(transport=capture_transport, timeout=None) as client:
                await client.post(
                    "http://nat-hint-benchmark.local/v1/chat/completions",
                    json=scenario_base_payload(scenario, 0),
                )
            captured[scenario_id] = capture_transport.payloads
            continue

        cache_pin_type = CachePinType.EPHEMERAL if config["cache_control_enabled"] else None
        cache_control_mode = (
            CacheControlMode.FIRST_ONLY if config["cache_control_mode"] == "first_only" else CacheControlMode.ALWAYS
        )
        transport = _DynamoTransport(
            transport=capture_transport,
            total_requests=config["total_requests"],
            osl=config["osl"],
            iat=config["iat"],
            cache_pin_type=cache_pin_type,
            cache_control_mode=cache_control_mode,
            max_sensitivity=1000,
        )
        with DynamoPrefixContext.scope(config["prefix_id"]):
            with Context.get().push_latency_sensitivity(config["latency_sensitivity"]):
                async with httpx.AsyncClient(transport=transport, timeout=None) as client:
                    for index in range(request_count):
                        await client.post(
                            "http://nat-hint-benchmark.local/v1/chat/completions",
                            json=scenario_base_payload(scenario, index),
                        )
        captured[scenario_id] = capture_transport.payloads

    return captured


class ClaudeCaptureHandler(BaseHTTPRequestHandler):
    server: "ClaudeCaptureServer"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def do_POST(self) -> None:
        content_length = int(self.headers.get("content-length", "0") or "0")
        raw_body = self.rfile.read(content_length) if content_length else b""
        try:
            body = json.loads(raw_body.decode() or "{}")
        except json.JSONDecodeError:
            body = {"_raw_body": raw_body.decode(errors="replace")}
        payload = dict(body) if isinstance(body, dict) else {"body": body}
        payload["_capture"] = {
            "method": "POST",
            "path": self.path,
            "headers": {key.lower(): value for key, value in self.headers.items()},
            "received_at_unix": time.time(),
        }
        self.server.payloads.append(payload)

        response = {
            "id": "msg_hint_benchmark",
            "type": "message",
            "role": "assistant",
            "model": body.get("model", "claude-hint-benchmark-model") if isinstance(body, dict) else "claude-hint-benchmark-model",
            "content": [{"type": "text", "text": "hint benchmark capture ok"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 1,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 1,
            },
        }
        encoded = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class ClaudeCaptureServer(ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), ClaudeCaptureHandler)
        self.payloads: list[dict[str, Any]] = []


def claude_cli_command(scenario: dict[str, Any], base_command: str, invocation_index: int = 0) -> list[str]:
    setup = scenario.get("client_setup") or scenario.get("synthetic_setup", {})
    cli_args = setup.get("cli_args")
    command = shlex.split(base_command)
    if isinstance(cli_args, list) and cli_args:
        command.extend(str(arg).replace("{invocation_index}", str(invocation_index)) for arg in cli_args)
        return command
    prompt = setup.get("cli_prompt") or f"Claude hint benchmark scenario {scenario['id']}. Reply with one short sentence."
    prompt = str(prompt).replace("{invocation_index}", str(invocation_index))
    command.extend(["-p", str(prompt), "--output-format", "json"])
    return command


def capture_claude_native_payloads(
    scenarios: list[dict[str, Any]],
    *,
    command: str,
    timeout_seconds: float,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    captured: dict[str, list[dict[str, Any]]] = {}
    client_runs: list[dict[str, Any]] = []
    for scenario in scenarios:
        server = ClaudeCaptureServer()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_port}"
            env = os.environ.copy()
            env.update(
                {
                    "ANTHROPIC_BASE_URL": base_url,
                    "ANTHROPIC_API_KEY": env.get("ANTHROPIC_API_KEY", "sk-hint-benchmark"),
                    "ANTHROPIC_AUTH_TOKEN": env.get("ANTHROPIC_AUTH_TOKEN", "sk-hint-benchmark"),
                }
            )
            setup = scenario.get("client_setup") or scenario.get("synthetic_setup", {})
            env_overrides = setup.get("env", {}) if isinstance(setup, dict) else {}
            if not isinstance(env_overrides, dict):
                env_overrides = {}
            for key, value in env_overrides.items():
                env[str(key)] = str(value).replace("{base_url}", base_url)
            request_count = max(1, int(scenario.get("workload_shape", {}).get("request_count") or 1))
            for invocation_index in range(request_count):
                cmd = claude_cli_command(scenario, command, invocation_index)
                started_at = time.time()
                try:
                    completed = subprocess.run(
                        cmd,
                        cwd=REPO_ROOT.parent,
                        env=env,
                        text=True,
                        capture_output=True,
                        timeout=timeout_seconds,
                        check=False,
                    )
                    client_runs.append(
                        {
                            "scenario_id": scenario["id"],
                            "invocation_index": invocation_index,
                            "command": cmd,
                            "returncode": completed.returncode,
                            "stdout_tail": completed.stdout[-2000:],
                            "stderr_tail": completed.stderr[-2000:],
                            "duration_seconds": time.time() - started_at,
                            "capture_base_url": base_url,
                            "captured_payload_count": len(server.payloads),
                        }
                    )
                except FileNotFoundError as exc:
                    raise HintBenchmarkConfigError(
                        f"Claude native capture requires the Claude CLI command {cmd[0]!r}. "
                        "Install/configure Claude Code or pass --claude-command."
                    ) from exc
                except subprocess.TimeoutExpired as exc:
                    client_runs.append(
                        {
                            "scenario_id": scenario["id"],
                            "invocation_index": invocation_index,
                            "command": cmd,
                            "returncode": "timeout",
                            "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
                            "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
                            "duration_seconds": time.time() - started_at,
                            "capture_base_url": base_url,
                            "captured_payload_count": len(server.payloads),
                        }
                    )
        finally:
            captured[scenario["id"]] = list(server.payloads)
            server.shutdown()
            thread.join(timeout=1)
    return captured, client_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", default="nemo_agent_toolkit", choices=tuple(DEFAULT_CONFIGS))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--scenario-file", type=Path, default=None)
    parser.add_argument("--knob-file", type=Path, default=None)
    parser.add_argument(
        "--knob-profile",
        default=None,
        help="Named knob profile from the knob file. Overrides --scenarios when provided.",
    )
    parser.add_argument(
        "--scenarios",
        default="smoke",
        help="Scenario group, scenario id, comma-separated selectors, or 'all'. Default: smoke.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Record benchmark recipes and expected evidence without executing NAT.",
    )
    parser.add_argument(
        "--fixture-observations",
        action="store_true",
        help=(
            "Generate fixture observations from scenario expectations. "
            "This proves benchmark plumbing only, not native harness emission."
        ),
    )
    parser.add_argument(
        "--nat-dynamo-transport-capture",
        action="store_true",
        help="Capture real NAT _DynamoTransport hint emission without forwarding to SGLang.",
    )
    parser.add_argument(
        "--claude-native-capture",
        action="store_true",
        help="Run the real Claude CLI/client against a local capture endpoint and validate emitted request fields.",
    )
    parser.add_argument(
        "--anthropic-api-payload-capture",
        action="store_true",
        help=(
            "Generate documented direct Anthropic API request/response payloads for Claude capability probes. "
            "This does not count as native Claude Code CLI emission."
        ),
    )
    parser.add_argument(
        "--claude-command",
        default=os.environ.get("CLAUDE_CODE_BIN", "claude"),
        help="Claude CLI command to execute for --claude-native-capture. Default: claude.",
    )
    parser.add_argument(
        "--claude-timeout-seconds",
        type=float,
        default=60.0,
        help="Per-scenario timeout for --claude-native-capture.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to artifacts/results/hint_benchmark/<run_id>.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--observed-jsonl",
        type=Path,
        default=None,
        help="Optional observed hint evidence JSONL. Later real runners will write this file.",
    )
    args = parser.parse_args()

    mode_count = sum(
        bool(mode)
        for mode in (
            args.dry_run,
            args.fixture_observations,
            args.nat_dynamo_transport_capture,
            args.claude_native_capture,
            args.anthropic_api_payload_capture,
        )
    )
    if mode_count != 1:
        parser.error(
            "Choose exactly one of --dry-run, --fixture-observations, "
            "--nat-dynamo-transport-capture, --claude-native-capture, "
            "or --anthropic-api-payload-capture."
        )
    if args.nat_dynamo_transport_capture and args.harness != "nemo_agent_toolkit":
        parser.error("--nat-dynamo-transport-capture requires --harness nemo_agent_toolkit.")
    if args.claude_native_capture and args.harness != "claude_code":
        parser.error("--claude-native-capture requires --harness claude_code.")
    if args.anthropic_api_payload_capture and args.harness != "claude_code":
        parser.error("--anthropic-api-payload-capture requires --harness claude_code.")

    defaults = DEFAULT_CONFIGS[args.harness]
    manifest_path = args.manifest or defaults["manifest"]
    scenario_path = args.scenario_file or defaults["scenarios"]
    knob_path = args.knob_file or defaults["knobs"]

    try:
        manifest, scenario_file = load_benchmark_inputs(manifest_path, scenario_path)
        if manifest["harness"]["id"] != args.harness:
            raise HintBenchmarkConfigError(
                f"--harness {args.harness!r} does not match manifest harness {manifest['harness']['id']!r}"
            )
        knob_profile = None
        if args.knob_profile:
            knobs = load_knob_profiles(knob_path)
            knob_profile = select_knob_profile(knobs, args.knob_profile)
            selected = select_scenarios(scenario_file, knob_profile["scenario_selectors"])
        else:
            selected = select_scenarios(scenario_file, args.scenarios)
        execution_mode = (
            "nat_dynamo_transport_capture"
            if args.nat_dynamo_transport_capture
            else "claude_native_capture"
            if args.claude_native_capture
            else "anthropic_api_payload_capture"
            if args.anthropic_api_payload_capture
            else "fixture_smoke"
            if args.fixture_observations
            else "dry_run"
        )
        result = build_dry_run(manifest, selected, run_id=args.run_id, execution_mode=execution_mode)
        if knob_profile:
            result["run"]["knob_profile"] = knob_profile["id"]
            result["run"]["knob_profile_name"] = knob_profile.get("display_name", knob_profile["id"])
            result["knob_profile"] = knob_profile
        generated_observation_modes = [
            args.fixture_observations,
            args.nat_dynamo_transport_capture,
            args.claude_native_capture,
            args.anthropic_api_payload_capture,
        ]
        if any(generated_observation_modes) and args.observed_jsonl:
            raise HintBenchmarkConfigError("Generated observation modes cannot be combined with --observed-jsonl")
        if args.fixture_observations:
            observations = build_fixture_observations(result["scenario_records"])
        elif args.nat_dynamo_transport_capture:
            captured_payloads = asyncio.run(capture_nat_dynamo_payloads(selected))
            observations = build_nat_payload_observations(manifest, result["scenario_records"], captured_payloads)
            result["captured_payload_counts"] = {
                scenario_id: len(payloads) for scenario_id, payloads in captured_payloads.items()
            }
        elif args.claude_native_capture:
            captured_payloads, client_runs = capture_claude_native_payloads(
                selected,
                command=args.claude_command,
                timeout_seconds=args.claude_timeout_seconds,
            )
            observations = build_payload_observations(
                manifest,
                result["scenario_records"],
                captured_payloads,
                evidence_source="claude_native_capture",
            )
            result["client_runs"] = client_runs
            result["captured_payload_counts"] = {
                scenario_id: len(payloads) for scenario_id, payloads in captured_payloads.items()
            }
        elif args.anthropic_api_payload_capture:
            captured_payloads = build_direct_api_payloads(selected)
            observations = build_payload_observations(
                manifest,
                result["scenario_records"],
                captured_payloads,
                evidence_source="anthropic_api_payload_capture",
            )
            result["captured_payload_counts"] = {
                scenario_id: len(payloads) for scenario_id, payloads in captured_payloads.items()
            }
        else:
            observations = load_observations_jsonl(args.observed_jsonl) if args.observed_jsonl else []
        if observations:
            result["observations"] = observations
        result["validation"] = validate_hint_evidence(
            manifest,
            result["scenario_records"],
            observations,
            execution_mode=execution_mode if observations else "dry_run",
        )
        out_dir = args.out_dir or DEFAULT_OUT_ROOT / result["run"]["run_id"]
        write_dry_run_outputs(result, out_dir)
    except HintBenchmarkConfigError as exc:
        parser.error(str(exc))

    summary = {
        **result["run"],
        "out_dir": str(out_dir),
        "selected_scenarios": [scenario["id"] for scenario in selected],
        "validation_rows": len(result["validation"]["validation_rows"]),
        "unknown_hint_rows": len(result["validation"]["unknown_hint_rows"]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
