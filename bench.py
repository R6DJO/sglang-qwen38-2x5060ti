#!/usr/bin/env python3
"""Однопоточный бенчмарк скорости генерации для SGLang (OpenAI API).

Меряет: TTFT (первый токен), decode tok/s = (токены - 1) / (t_total - t_tft),
сквозные tok/s = токены / t_total. Совместимо с методикой из README
(39-41.5 tok/s у основного сервиса).

Использование: bench.py [url] [model] [n_iters] [max_tokens]
"""
import http.client
import json
import sys
import time
from urllib.parse import urlparse

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:30001"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-27b-dspark-test"


def _int_arg(idx: int, default: int) -> int:
    try:
        return int(sys.argv[idx])
    except (ValueError, IndexError):
        return default


ITERS = _int_arg(3, 3)
MAX_TOKENS = _int_arg(4, 512)

_parsed = urlparse(URL)
if _parsed.scheme == "https":
    _conn_cls = http.client.HTTPSConnection
elif _parsed.scheme == "http":
    _conn_cls = http.client.HTTPConnection
else:
    sys.exit(f"unsupported URL scheme: {_parsed.scheme!r} (http/https only)")
_HOST: str = _parsed.hostname or ""
if not _HOST:
    sys.exit(f"URL without host: {URL!r}")
_PORT = _parsed.port or (443 if _parsed.scheme == "https" else 80)
_PATH = _parsed.path.rstrip("/") + "/v1/chat/completions"

PROMPT = (
    "Напиши подробный технический разбор гибридной архитектуры "
    "трансформеров с линейным вниманием: что такое Gated DeltaNet, "
    "зачем комбинировать его с полным attention, какие trade-offs "
    "по памяти и скорости на длинном контексте."
)

PAYLOAD = json.dumps(
    {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": True,
        # Без этого SGLang не присылает usage в стриме; при спекулятивном
        # декодинге один SSE-чанк может нести несколько токенов сразу,
        # поэтому считать чанки нельзя -- нужен completion_tokens.
        "stream_options": {"include_usage": True},
    }
).encode()


def one_run(i):
    conn = _conn_cls(_HOST, _PORT, timeout=600)
    t0 = time.perf_counter()
    ttft = None
    usage = None
    chunks = 0
    try:
        conn.request(
            "POST",
            _PATH,
            body=PAYLOAD,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        if resp.status != 200:
            sys.exit(f"HTTP {resp.status}: {resp.read().decode()[:500]}")
        for raw in resp:
            line = raw.strip()
            if not line.startswith(b"data: "):
                continue
            data = line[6:]
            if data == b"[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            delta = obj.get("choices") and obj["choices"][0].get("delta", {})
            if ttft is None and delta and (
                delta.get("content") or delta.get("reasoning_content")
            ):
                ttft = time.perf_counter() - t0
            chunks += 1
    finally:
        conn.close()
    total = time.perf_counter() - t0
    comp = usage["completion_tokens"] if usage else chunks
    tft = ttft if ttft is not None else total
    dec = (comp - 1) / (total - tft) if comp > 1 and total > tft else 0.0
    e2e = comp / total if total > 0 else 0.0
    print(
        f"run {i}: {comp} tok | ttft {tft:.2f}s | "
        f"decode {dec:.1f} tok/s | e2e {e2e:.1f} tok/s",
        flush=True,
    )
    return dec, e2e


decs = []
e2es = []
for i in range(1, ITERS + 1):
    d, e = one_run(i)
    decs.append(d)
    e2es.append(e)

decs_s = sorted(decs)
print(
    f"\nMEDIAN decode: {decs_s[len(decs_s) // 2]:.1f} tok/s | "
    f"e2e: {sorted(e2es)[len(e2es) // 2]:.1f} tok/s | "
    f"runs: {[f'{d:.1f}' for d in decs]}"
)
