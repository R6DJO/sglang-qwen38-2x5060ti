#!/usr/bin/env python3
"""Параллельный бенчмарк: 2 перекрывающихся запроса.

Краш-сценарий mamba4 -- чанковый префилл запроса B во время декода A.
Запрос B стартует через STAGGER_MS мс после A, то есть попадает в декод A.
Мерит: TTFT, decode tok/s и e2e каждого запроса, суммарный throughput.

Использование: bench-par2.py [url] [model] [iters] [max_tokens] [stagger_ms]
"""
import http.client
import json
import sys
import threading
import time
from urllib.parse import urlparse

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:30000"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-27b"


def _int_arg(idx: int, default: int) -> int:
    try:
        return int(sys.argv[idx])
    except (ValueError, IndexError):
        return default


ITERS = _int_arg(3, 3)
MAX_TOKENS = _int_arg(4, 512)
STAGGER_MS = _int_arg(5, 2000)

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

PROMPTS = [
    (
        "Напиши подробный технический разбор гибридной архитектуры "
        "трансформеров с линейным вниманием: что такое Gated DeltaNet, "
        "зачем комбинировать его с полным attention, какие trade-offs "
        "по памяти и скорости на длинном контексте."
    ),
    (
        "Объясни, как устроен спекулятивный декодинг с черновик-моделью: "
        "где здесь узкие места, когда он даёт ускорение, а когда вредит, "
        "и как это связано с батчингом запросов."
    ),
]


def one_run(idx: int):
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPTS[idx]}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    conn = _conn_cls(_HOST, _PORT, timeout=600)
    t0 = time.perf_counter()
    ttft = None
    usage = None
    chunks = 0
    try:
        conn.request(
            "POST",
            _PATH,
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        if resp.status != 200:
            return ("ERR", f"HTTP {resp.status}: {resp.read().decode()[:200]}")
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
    return ("OK", comp, tft, dec, e2e)


def safe_run(idx: int):
    try:
        return one_run(idx)
    except Exception as e:  # соединение/таймаут/обрыв
        return ("ERR", f"{type(e).__name__}: {e}", 0, 0.0, 0.0, 0.0)


def _run_into(out: list, idx: int) -> None:
    """Записывает результат в заранее созданный список (без замыканий)."""
    out[idx] = safe_run(idx)


for it in range(1, ITERS + 1):
    res: list = [None, None]
    t_start = time.perf_counter()
    tA = threading.Thread(target=_run_into, args=(res, 0))
    tA.start()
    time.sleep(STAGGER_MS / 1000.0)  # запрос B стартует в декоде A
    tB = threading.Thread(target=_run_into, args=(res, 1))
    tB.start()
    tA.join()
    tB.join()
    wall = time.perf_counter() - t_start

    parts = []
    total_tok = 0
    ok = True
    for tag, r in zip(("A", "B"), res, strict=True):
        if r[0] != "OK":
            parts.append(f"{tag}: ОШИБКА {r[1]}")
            ok = False
        else:
            _, comp, tft, dec, e2e = r
            total_tok += comp
            parts.append(
                f"{tag}: {comp} tok | ttft {tft:.2f}s | "
                f"decode {dec:.1f} | e2e {e2e:.1f}"
            )
    summ = f" | суммарно {total_tok/wall:.1f} tok/s за {wall:.1f}s" if ok else ""
    print(f"iter {it} (stagger {STAGGER_MS}ms): " + " ; ".join(parts) + summ, flush=True)
