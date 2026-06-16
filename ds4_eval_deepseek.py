#!/usr/bin/env python3
"""Run the ds4_eval.c benchmark cases against DeepSeek's hosted API.

The benchmark data intentionally remains in ds4_eval.c.  This script parses the
embedded eval_cases[] table, builds prompts that match ds4-eval, calls the
OpenAI-compatible DeepSeek chat completions endpoint, and writes a table with
the same row order and column layout as ds4_eval_local_result.txt.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are solving a hard benchmark question. Reason carefully. "
    "The final answer must follow the requested format exactly."
)
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_OUTPUT = "ds4_eval_deepseek_v4_flash_result.txt"
DEFAULT_TRACE = "ds4_eval_deepseek_v4_flash_trace.jsonl"
ANSWER_MAX = 32


class RetryableAPIError(RuntimeError):
    pass


@dataclass
class EvalCase:
    source: str = ""
    case_id: str = ""
    domain: str = ""
    title: str = ""
    question: str = ""
    choices: list[str] = field(default_factory=list)
    answer: str = ""


def decode_c_string_literal(token: str) -> str:
    if len(token) < 2 or token[0] != '"' or token[-1] != '"':
        raise ValueError(f"not a C string literal: {token[:40]}")
    out: list[str] = []
    i = 1
    end = len(token) - 1
    simple = {
        "a": "\a",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
        "v": "\v",
        "\\": "\\",
        "'": "'",
        '"': '"',
        "?": "?",
    }
    while i < end:
        ch = token[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        i += 1
        if i >= end:
            out.append("\\")
            break
        esc = token[i]
        i += 1
        if esc in simple:
            out.append(simple[esc])
        elif esc == "x":
            j = i
            while j < end and token[j] in "0123456789abcdefABCDEF":
                j += 1
            if j == i:
                out.append("x")
            else:
                out.append(chr(int(token[i:j], 16)))
                i = j
        elif esc in "01234567":
            j = i - 1
            while i < end and i - j < 3 and token[i] in "01234567":
                i += 1
            out.append(chr(int(token[j:i], 8)))
        else:
            out.append(esc)
    return "".join(out)


def skip_ws_and_comments(src: str, pos: int) -> int:
    n = len(src)
    while pos < n:
        if src[pos].isspace():
            pos += 1
        elif src.startswith("//", pos):
            end = src.find("\n", pos + 2)
            pos = n if end < 0 else end + 1
        elif src.startswith("/*", pos):
            end = src.find("*/", pos + 2)
            pos = n if end < 0 else end + 2
        else:
            break
    return pos


def parse_c_string_expr(src: str, pos: int) -> tuple[str, int]:
    parts: list[str] = []
    n = len(src)
    pos = skip_ws_and_comments(src, pos)
    while pos < n and src[pos] == '"':
        start = pos
        pos += 1
        escaped = False
        while pos < n:
            ch = src[pos]
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                pos += 1
                break
            pos += 1
        parts.append(decode_c_string_literal(src[start:pos]))
        pos = skip_ws_and_comments(src, pos)
    if not parts:
        raise ValueError(f"expected C string literal near offset {pos}")
    return "".join(parts), pos


def find_matching_brace(src: str, open_pos: int) -> int:
    depth = 0
    pos = open_pos
    n = len(src)
    in_string = False
    escaped = False
    in_line_comment = False
    in_block_comment = False
    while pos < n:
        ch = src[pos]
        nxt = src[pos + 1] if pos + 1 < n else ""
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
        elif in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                pos += 1
        elif in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == "/" and nxt == "/":
            in_line_comment = True
            pos += 1
        elif ch == "/" and nxt == "*":
            in_block_comment = True
            pos += 1
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return pos
        pos += 1
    raise ValueError("unclosed brace while parsing eval_cases")


def parse_case_block(block: str) -> EvalCase:
    case = EvalCase()
    pos = 0
    field_re = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?\s*=\s*", re.S)
    while True:
        m = field_re.search(block, pos)
        if not m:
            break
        name = m.group(1)
        choice_index = int(m.group(2)) if m.group(2) is not None else None
        value, pos = parse_c_string_expr(block, m.end())
        if name == "source":
            case.source = value
        elif name == "id":
            case.case_id = value
        elif name == "domain":
            case.domain = value
        elif name == "title":
            case.title = value
        elif name == "question":
            case.question = value
        elif name == "answer":
            case.answer = value
        elif name == "choice" and choice_index is not None:
            while len(case.choices) <= choice_index:
                case.choices.append("")
            case.choices[choice_index] = value
    case.choices = [choice for choice in case.choices if choice]
    missing = [
        name
        for name, value in (
            ("source", case.source),
            ("id", case.case_id),
            ("question", case.question),
            ("answer", case.answer),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"case missing {', '.join(missing)}")
    return case


def load_eval_cases(path: Path) -> list[EvalCase]:
    src = path.read_text(encoding="utf-8")
    m = re.search(r"static\s+const\s+eval_case\s+eval_cases\[\]\s*=\s*\{", src)
    if not m:
        raise ValueError(f"{path} does not contain eval_cases[]")
    start = m.end() - 1
    end = find_matching_brace(src, start)
    body = src[start + 1 : end]
    cases: list[EvalCase] = []
    pos = 0
    while True:
        pos = skip_ws_and_comments(body, pos)
        if pos >= len(body):
            break
        if body[pos] == ",":
            pos += 1
            continue
        if body[pos] != "{":
            raise ValueError(f"expected case initializer near offset {pos}")
        close = find_matching_brace(body, pos)
        cases.append(parse_case_block(body[pos + 1 : close]))
        pos = close + 1
    return cases


def is_multiple_choice(case: EvalCase) -> bool:
    return bool(case.choices)


def is_compsec(case: EvalCase) -> bool:
    return case.source == "COMPSEC"


def build_question_prompt(case: EvalCase) -> str:
    parts = [case.question, "\n"]
    if is_multiple_choice(case):
        parts.append("\nChoices:\n")
        for i, choice in enumerate(case.choices):
            parts.append(f"{chr(ord('A') + i)}. {choice}\n")
        parts.append(
            "\nSolve the question. At the end, write exactly one final line in this "
            "format and do not write anything after it:\n"
            "Answer: <letter>"
        )
    elif is_compsec(case):
        parts.append(
            "\nAt the end, write exactly one final line in this format and do not "
            "write anything after it:\n"
            "Answer: <line number or comma-separated line numbers>"
        )
    else:
        parts.append(
            "\nSolve the problem. At the end, write exactly one final line in this "
            "format and do not write anything after it:\n"
            "Answer: <integer>"
        )
    return "".join(parts)


def is_letter_boundary(before: str, after: str) -> bool:
    return not before.isalpha() and not after.isalpha()


def find_last_answer_marker(visible: str) -> int:
    last = -1
    lower = visible.lower()
    pos = 0
    while True:
        pos = lower.find("answer", pos)
        if pos < 0:
            break
        before = " " if pos == 0 else visible[pos - 1]
        after = visible[pos + len("answer")] if pos + len("answer") < len(visible) else ""
        if is_letter_boundary(before, after):
            q = pos + len("answer")
            while q < len(visible) and visible[q].isspace():
                q += 1
            if q < len(visible) and visible[q] == ":":
                last = pos
        pos += 1
    if last >= 0:
        return last
    return lower.find("answer")


def visible_generated(generated: str) -> str:
    marker = generated.find("</think>")
    return generated[marker + len("</think>") :] if marker >= 0 else generated


def find_answer_letter(generated: str, nchoices: int) -> str:
    if nchoices <= 0:
        return "?"
    visible = visible_generated(generated)
    max_answer = chr(ord("A") + nchoices - 1)
    answer_pos = find_last_answer_marker(visible)
    if answer_pos >= 0:
        end = min(len(visible), answer_pos + 96)
        for i in range(answer_pos, end):
            c = visible[i].upper()
            if "A" <= c <= max_answer:
                before = " " if i == 0 else visible[i - 1]
                after = visible[i + 1] if i + 1 < len(visible) else ""
                if is_letter_boundary(before, after):
                    return c
    for i in range(len(visible) - 1, -1, -1):
        c = visible[i].upper()
        if "A" <= c <= max_answer:
            before = " " if i == 0 else visible[i - 1]
            after = visible[i + 1] if i + 1 < len(visible) else ""
            if is_letter_boundary(before, after):
                return c
    return "?"


def normalize_integer_answer(text: str) -> str:
    text = text.lstrip("0")
    return text if text else "0"


def find_integer_answer(generated: str) -> str:
    visible = visible_generated(generated)
    answer_pos = find_last_answer_marker(visible)
    if answer_pos >= 0:
        chunk = visible[answer_pos : answer_pos + 160]
        m = re.search(r"\d+", chunk)
        if m:
            return normalize_integer_answer(m.group(0))
    matches = list(re.finditer(r"\d+", visible))
    if matches:
        return normalize_integer_answer(matches[-1].group(0))
    return "?"


def normalize_compsec_line_spec(text: str) -> str:
    parts: list[str] = []
    for m in re.finditer(r"\d+(?:\s*-\s*\d+)?", text):
        part = re.sub(r"\s+", "", m.group(0))
        parts.append(part)
    return ",".join(parts) if parts else "?"


def find_compsec_answer(generated: str) -> str:
    visible = visible_generated(generated)
    answer_pos = find_last_answer_marker(visible)
    if answer_pos >= 0:
        chunk = visible[answer_pos : answer_pos + 160].splitlines()[0]
        got = normalize_compsec_line_spec(chunk)
        if got != "?":
            return got
    return find_integer_answer(generated)


def parse_line_spec(spec: str) -> set[int]:
    lines: set[int] = set()
    for m in re.finditer(r"\d+(?:-\d+)?", spec):
        token = m.group(0)
        if "-" in token:
            a_s, b_s = token.split("-", 1)
            a, b = int(a_s), int(b_s)
            if a > b:
                a, b = b, a
            lines.update(range(max(a, 0), min(b, 255) + 1))
        else:
            v = int(token)
            if 0 <= v <= 255:
                lines.add(v)
    return lines


def find_case_answer(case: EvalCase, generated: str) -> str:
    if is_multiple_choice(case):
        return find_answer_letter(generated, len(case.choices))
    if is_compsec(case):
        return find_compsec_answer(generated)
    return find_integer_answer(generated)


def answer_matches(case: EvalCase, got: str) -> bool:
    if is_multiple_choice(case):
        return bool(got and case.answer and got[0] == case.answer[0])
    if is_compsec(case):
        expected = parse_line_spec(case.answer)
        actual = parse_line_spec(got)
        return bool(expected and actual and actual.issubset(expected))
    return got == normalize_integer_answer(case.answer)


def load_trace(trace_path: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not trace_path.exists():
        return done
    with trace_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            has_usage = int(row.get("prompt_tokens") or 0) > 0 or int(row.get("completion_tokens") or 0) > 0
            if row.get("status") in {"PASSED", "FAILED"} and has_usage:
                done[str(row["index"])] = row
    return done


def append_trace(trace_path: Path, row: dict[str, Any]) -> None:
    with trace_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def row_from_trace(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": int(row["index"]),
        "status": row["status"],
        "prompt_tokens": int(row.get("prompt_tokens") or 0),
        "completion_tokens": int(row.get("completion_tokens") or 0),
        "given": str(row.get("given") or "-"),
    }


def call_deepseek(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    thinking: str,
    reasoning_effort: str,
    stream: bool,
    stream_include_usage: bool,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
    }
    if stream and stream_include_usage:
        payload["stream_options"] = {"include_usage": True}
    if thinking == "enabled":
        payload["thinking"] = {"type": "enabled"}
        payload["reasoning_effort"] = reasoning_effort
    elif thinking == "disabled":
        payload["thinking"] = {"type": "disabled"}

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    attempt = 0
    while True:
        attempt += 1
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if stream:
                    return read_sse_response(resp)
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            retryable = exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
            if not retryable or attempt > retries:
                raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except (http.client.IncompleteRead, RetryableAPIError) as exc:
            if attempt > retries:
                raise RuntimeError(str(exc)) from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt > retries:
                raise RuntimeError(str(exc)) from exc

        delay = min(60.0, 2.0**attempt) + random.random()
        print(f"retrying API call in {delay:.1f}s after attempt {attempt}", file=sys.stderr)
        time.sleep(delay)


def read_sse_response(resp: Any) -> dict[str, Any]:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    response_id: str | None = None
    created: int | None = None
    model: str | None = None
    chunks = 0
    seen_done = False

    for raw_line in resp:
        line = raw_line.decode("utf-8", "replace").strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            seen_done = True
            break
        chunk = json.loads(data)
        chunks += 1
        if chunk.get("error"):
            raise RetryableAPIError(json.dumps(chunk["error"], ensure_ascii=False))
        response_id = response_id or chunk.get("id")
        created = created or chunk.get("created")
        model = model or chunk.get("model")
        if chunk.get("usage"):
            usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        if choice.get("finish_reason") is not None:
            finish_reason = choice.get("finish_reason")
        delta = choice.get("delta") or {}
        if delta.get("reasoning_content"):
            reasoning_parts.append(delta["reasoning_content"])
        if delta.get("content"):
            content_parts.append(delta["content"])

    if not seen_done or chunks == 0 or (
        not content_parts and not reasoning_parts and not usage and finish_reason is None
    ):
        raise RetryableAPIError("incomplete streaming response")

    return {
        "id": response_id,
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": "".join(content_parts),
                    "reasoning_content": "".join(reasoning_parts),
                },
            }
        ],
        "usage": usage,
    }


def completion_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    return f"{reasoning}</think>{content}" if reasoning else content


def usage_counts(response: dict[str, Any]) -> tuple[int, int, int]:
    usage = response.get("usage") or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    total = int(usage.get("total_tokens") or (prompt + completion))
    if completion == 0 and total >= prompt:
        completion = total - prompt
    return prompt, completion, total


def write_result(path: Path, cases: list[EvalCase], rows: list[dict[str, Any]]) -> None:
    by_index = {row["index"]: row for row in rows}
    with path.open("w", encoding="utf-8") as fp:
        fp.write(f"{'#':<3} {'state':<8} {'prompt':>8} {'gen':>8} {'total':>8} {'given':<8} {'correct':<8} test\n")
        for i, case in enumerate(cases, start=1):
            row = by_index.get(i, {})
            status = row.get("status", "PENDING")
            prompt_tokens = int(row.get("prompt_tokens") or 0)
            completion_tokens = int(row.get("completion_tokens") or 0)
            total_tokens = prompt_tokens + completion_tokens
            given = str(row.get("given") or "-")[: ANSWER_MAX - 1]
            fp.write(
                f"{i:3d} {status:<8} {prompt_tokens:8d} {completion_tokens:8d} "
                f"{total_tokens:8d} {given:<8} {case.answer:<8} {case.source}/{case.case_id}\n"
            )


def parse_case_sequence(spec: str, ncases: int) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a_s, b_s = part.split("-", 1)
            a, b = int(a_s), int(b_s)
            step = 1 if a <= b else -1
            values = range(a, b + step, step)
        else:
            values = [int(part)]
        for value in values:
            if value < 1 or value > ncases:
                raise ValueError(f"case {value} out of range 1..{ncases}")
            if value not in seen:
                selected.append(value)
                seen.add(value)
    return selected


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate ds4_eval.c cases with DeepSeek V4 Flash API."
    )
    parser.add_argument("--source", default="ds4_eval.c", help="path to ds4_eval.c")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="result table path")
    parser.add_argument("--trace", default=DEFAULT_TRACE, help="append-only JSONL trace path")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=None, help="defaults to DEEPSEEK_API_KEY")
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--thinking", choices=["enabled", "disabled", "auto"], default="enabled")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--no-stream", action="store_true", help="disable SSE streaming")
    parser.add_argument(
        "--no-stream-usage",
        action="store_true",
        help="do not request stream_options.include_usage",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N selected cases")
    parser.add_argument("--case-sequence", default="", help="1-based case list/ranges, e.g. 1,3-5")
    parser.add_argument("--force", action="store_true", help="rerun cases already in the trace")
    args = parser.parse_args(argv)

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        parser.error("set DEEPSEEK_API_KEY or pass --api-key")

    source_path = Path(args.source)
    output_path = Path(args.output)
    trace_path = Path(args.trace)
    cases = load_eval_cases(source_path)
    selected = (
        parse_case_sequence(args.case_sequence, len(cases))
        if args.case_sequence
        else list(range(1, len(cases) + 1))
    )
    if args.limit > 0:
        selected = selected[: args.limit]

    done = {} if args.force else load_trace(trace_path)
    rows: list[dict[str, Any]] = [row_from_trace(row) for row in done.values()]
    write_result(output_path, cases, rows)
    print(f"loaded {len(cases)} cases from {source_path}", file=sys.stderr)
    print(f"writing result table to {output_path}", file=sys.stderr)

    for index in selected:
        case = cases[index - 1]
        if str(index) in done:
            print(f"[{index}/{len(cases)}] resume {case.source}/{case.case_id}", file=sys.stderr)
            continue
        prompt = build_question_prompt(case)
        print(f"[{index}/{len(cases)}] call {case.source}/{case.case_id}", file=sys.stderr)
        started = time.time()
        response = call_deepseek(
            base_url=args.base_url,
            api_key=api_key,
            model=args.model,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            thinking=args.thinking,
            reasoning_effort=args.reasoning_effort,
            stream=not args.no_stream,
            stream_include_usage=not args.no_stream_usage,
            timeout=args.timeout,
            retries=args.retries,
        )
        generated = completion_text(response)
        got = find_case_answer(case, generated)
        passed = answer_matches(case, got)
        prompt_tokens, completion_tokens, total_tokens = usage_counts(response)
        row = {
            "index": index,
            "status": "PASSED" if passed else "FAILED",
            "source": case.source,
            "id": case.case_id,
            "domain": case.domain,
            "title": case.title,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "given": got[: ANSWER_MAX - 1],
            "correct": case.answer,
            "elapsed_sec": round(time.time() - started, 3),
            "model": args.model,
            "thinking": args.thinking,
            "reasoning_effort": args.reasoning_effort if args.thinking == "enabled" else None,
            "finish_reason": (response.get("choices") or [{}])[0].get("finish_reason"),
            "content": (response.get("choices") or [{}])[0].get("message", {}).get("content", ""),
            "reasoning_content": (response.get("choices") or [{}])[0].get("message", {}).get("reasoning_content", ""),
            "usage": response.get("usage") or {},
        }
        append_trace(trace_path, row)
        rows = [r for r in rows if r["index"] != index]
        rows.append(row_from_trace(row))
        write_result(output_path, cases, rows)
        print(
            f"[{index}/{len(cases)}] {row['status']} got={row['given']} "
            f"expected={case.answer} prompt={prompt_tokens} gen={completion_tokens}",
            file=sys.stderr,
        )

    final_rows = [row_from_trace(row) for row in load_trace(trace_path).values()]
    write_result(output_path, cases, final_rows)
    passed = sum(1 for row in final_rows if row["status"] == "PASSED")
    failed = sum(1 for row in final_rows if row["status"] == "FAILED")
    print(f"complete rows={len(final_rows)} passed={passed} failed={failed}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
