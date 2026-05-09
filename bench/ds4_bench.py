#!/usr/bin/env python3
"""Reproducible local benchmark harness for ds4.

The harness intentionally drives the public ds4 CLI/server binaries.  That keeps
the benchmark close to how the engine is used while still collecting structured
JSON from --bench-json and compact server KV-cache log metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def make_prompt(chars: int) -> str:
    seed = (
        "You are auditing a local DeepSeek V4 Flash inference engine. "
        "Track exact cache behavior, latency, correctness, and Metal execution. "
        "The workload contains repeated structured notes so prefix reuse and "
        "long prefill paths are exercised deterministically.\n"
    )
    block = (
        "\n[NOTE]\n"
        "component=ds4 metal_graph=layer_major kv=compressed_disk "
        "goal=measure_prefill_decode_quality\n"
        "detail=Keep this paragraph stable so token boundaries remain comparable "
        "across benchmark variants. Report concrete timings and no speculation.\n"
    )
    out = [seed]
    while sum(len(x) for x in out) < chars:
        out.append(block)
    out.append("\nSummarize the main performance risk in one short sentence.\n")
    return "".join(out)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def run_cmd(name: str, cmd: list[str], env: dict[str, str], out_dir: Path,
            timeout: int | None = None) -> dict:
    stdout_path = out_dir / f"{name}.stdout.txt"
    stderr_path = out_dir / f"{name}.stderr.txt"
    t0 = time.monotonic()
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=out,
            stderr=err,
            timeout=timeout,
        )
    elapsed = time.monotonic() - t0
    return {
        "name": name,
        "returncode": proc.returncode,
        "wall_seconds": elapsed,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "command": cmd,
    }


def run_cli_case(args: argparse.Namespace, out_dir: Path, prompt_file: Path,
                 name: str, env_extra: dict[str, str],
                 extra_args: list[str] | None = None) -> dict:
    bench_json = out_dir / f"{name}.json"
    env = os.environ.copy()
    env.update(env_extra)
    cmd = [
        str(ROOT / "ds4"),
        "-m", args.model,
        "--prompt-file", str(prompt_file),
        "--ctx", str(args.ctx),
        "--tokens", str(args.tokens),
        "--temp", "0",
        "--top-p", "1",
        "--nothink",
        "--quiet",
        "--bench-json", str(bench_json),
    ]
    if args.warm_weights:
        cmd.append("--warm-weights")
    if extra_args:
        cmd.extend(extra_args)
    result = run_cmd(name, cmd, env, out_dir, timeout=args.timeout)
    metrics = {}
    if bench_json.exists():
        metrics = json.loads(bench_json.read_text())
    result["metrics"] = metrics
    result["bench_json"] = str(bench_json)
    return result


def wait_http(port: int, timeout_s: int) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except OSError as exc:
            last_error = exc
        except urllib.error.URLError as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"server did not become ready on port {port}: {last_error}")


def post_chat(port: int, prompt: str, timeout_s: int) -> tuple[int, float]:
    body = json.dumps({
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = resp.read()
        status = resp.status
    return status, time.monotonic() - t0


def parse_kv_log(path: Path) -> list[dict]:
    rows = []
    patterns = [
        ("miss", re.compile(r"kv cache miss prompt_tokens=(\d+) entries=(\d+) quant=(\d+) scan=([0-9.]+) ms")),
        ("stored", re.compile(r"kv cache stored tokens=(\d+) trimmed=(\d+) reason=([^ ]+) size=([0-9.]+) MiB save=([0-9.]+) ms")),
        ("hit", re.compile(r"kv cache hit tokens=(\d+) quant=(\d+) load=([0-9.]+) ms")),
    ]
    if not path.exists():
        return rows
    text = path.read_text(errors="replace")
    for line in text.splitlines():
        for kind, pat in patterns:
            m = pat.search(line)
            if m:
                rows.append({"kind": kind, "line": line, "groups": list(m.groups())})
    return rows


def start_server(args: argparse.Namespace, out_dir: Path, kv_dir: Path,
                 name: str) -> tuple[subprocess.Popen, Path]:
    log_path = out_dir / f"{name}.server.log"
    env = os.environ.copy()
    env["DS4_METAL_MEMORY_REPORT"] = env.get("DS4_METAL_MEMORY_REPORT", "1")
    cmd = [
        str(ROOT / "ds4-server"),
        "-m", args.model,
        "--ctx", str(args.ctx),
        "--tokens", "1",
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--kv-disk-dir", str(kv_dir),
        "--kv-disk-space-mb", str(args.kv_mb),
        "--kv-cache-min-tokens", str(args.kv_min_tokens),
        "--kv-cache-cold-max-tokens", str(args.kv_cold_max_tokens),
    ]
    if args.warm_weights:
        cmd.append("--warm-weights")
    log = log_path.open("wb")
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    proc._ds4_log_file = log  # type: ignore[attr-defined]
    return proc, log_path


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=30)
    log = getattr(proc, "_ds4_log_file", None)
    if log:
        log.close()


def run_kv_case(args: argparse.Namespace, out_dir: Path, prompt: str) -> dict:
    kv_dir = out_dir / "kv-cache"
    if kv_dir.exists():
        shutil.rmtree(kv_dir)
    kv_dir.mkdir(parents=True)

    records = []
    for phase in ("cold-store", "restart-hit"):
        proc, log_path = start_server(args, out_dir, kv_dir, phase)
        try:
            wait_http(args.port, args.server_timeout)
            status, latency = post_chat(args.port, prompt, args.server_timeout)
            records.append({
                "phase": phase,
                "status": status,
                "request_seconds": latency,
                "log": str(log_path),
            })
        finally:
            stop_server(proc)
        records[-1]["kv_events"] = parse_kv_log(log_path)
    return {"name": "kv-disk", "records": records, "kv_dir": str(kv_dir)}


def summarize(results: list[dict], out_dir: Path) -> None:
    lines = ["# ds4 benchmark summary", ""]
    for r in results:
        if "metrics" in r:
            m = r.get("metrics") or {}
            lines.append(
                f"- {r['name']}: rc={r['returncode']} "
                f"prompt={m.get('prompt_tokens')} gen={m.get('generated_tokens')} "
                f"prefill={m.get('prefill_tokens_per_second', 0):.2f} t/s "
                f"decode={m.get('decode_tokens_per_second', 0):.2f} t/s "
                f"wall={r['wall_seconds']:.2f}s"
            )
        elif r.get("name") == "kv-disk":
            for rec in r["records"]:
                events = ",".join(e["kind"] for e in rec.get("kv_events", []))
                lines.append(
                    f"- kv {rec['phase']}: status={rec['status']} "
                    f"request={rec['request_seconds']:.2f}s events={events or 'none'}"
                )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description="Run ds4 local benchmark variants")
    p.add_argument("--model", default="ds4flash.gguf")
    p.add_argument("--ctx", type=int, default=32768)
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--prompt-chars", type=int, default=60000)
    p.add_argument("--out-dir", default="")
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--warm-weights", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--mtp", default="")
    p.add_argument("--skip-prefill", action="store_true")
    p.add_argument("--skip-gpu-top1", action="store_true")
    p.add_argument("--skip-mtp", action="store_true")
    p.add_argument("--run-kv", action="store_true")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--kv-mb", type=int, default=32768)
    p.add_argument("--kv-min-tokens", type=int, default=512)
    p.add_argument("--kv-cold-max-tokens", type=int, default=30000)
    p.add_argument("--server-timeout", type=int, default=1800)
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "bench" / "results" / now_stamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = make_prompt(args.prompt_chars)
    prompt_file = out_dir / "prompt.txt"
    prompt_file.write_text(prompt)

    results: list[dict] = []
    profile_env = {
        "DS4_METAL_MEMORY_REPORT": "1",
        "DS4_METAL_GRAPH_PREFILL_PROFILE": "1",
        "DS4_METAL_GRAPH_TOKEN_PROFILE": "1",
    } if args.profile else {"DS4_METAL_MEMORY_REPORT": "1"}

    if not args.skip_prefill:
        for chunk in (4096, 8192):
            env = dict(profile_env)
            env["DS4_METAL_PREFILL_CHUNK"] = str(chunk)
            env["DS4_METAL_GRAPH_RAW_CAP"] = "8192"
            results.append(run_cli_case(args, out_dir, prompt_file, f"prefill-chunk-{chunk}", env))

    if not args.skip_gpu_top1:
        env = dict(profile_env)
        env["DS4_METAL_PREFILL_CHUNK"] = "4096"
        results.append(run_cli_case(args, out_dir, prompt_file, "decode-baseline-full-logits", env))
        env_fast = dict(env)
        env_fast["DS4_GREEDY_GPU_TOP1"] = "1"
        results.append(run_cli_case(args, out_dir, prompt_file, "decode-gpu-top1", env_fast))

    if not args.skip_mtp:
        mtp = Path(args.mtp) if args.mtp else None
        if mtp and mtp.exists():
            env = dict(profile_env)
            env["DS4_MTP_SPEC_DISABLE"] = "1"
            results.append(run_cli_case(
                args, out_dir, prompt_file, "mtp-disabled",
                env, ["--mtp", str(mtp), "--mtp-draft", "2"]))
            env_mtp = dict(profile_env)
            env_mtp["DS4_MTP_TIMING"] = "1"
            env_mtp["DS4_MTP_CONF_LOG"] = "1"
            results.append(run_cli_case(
                args, out_dir, prompt_file, "mtp-enabled",
                env_mtp, ["--mtp", str(mtp), "--mtp-draft", "2"]))
        else:
            results.append({"name": "mtp", "skipped": True, "reason": "no --mtp GGUF path"})

    if args.run_kv:
        results.append(run_kv_case(args, out_dir, prompt))

    write_json(out_dir / "results.json", results)
    summarize(results, out_dir)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
