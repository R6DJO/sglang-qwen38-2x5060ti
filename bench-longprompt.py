#!/usr/bin/env python3
"""TTFT и скорость префилла на длинном промпте (замер chunked-prefill).

Каждая итерация использует свой поворот абзацев, чтобы префикс не попадал
в radix-кэш предыдущей итерации.

Использование: bench-longprompt.py [url] [model] [n_chars] [max_tokens] [iters]
"""
import http.client
import json
import sys
import time
from urllib.parse import urlparse

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:30000"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-27b"


def _int_arg(idx: int, default: int) -> int:
    try:
        return int(sys.argv[idx])
    except (ValueError, IndexError):
        return default


N_CHARS = _int_arg(3, 100_000)
MAX_TOKENS = _int_arg(4, 256)
ITERS = _int_arg(5, 2)

_parsed = urlparse(URL)
if _parsed.scheme == "https":
    _conn_cls = http.client.HTTPSConnection
else:
    _conn_cls = http.client.HTTPConnection
_HOST: str = _parsed.hostname or ""
if not _HOST:
    sys.exit(f"URL without host: {URL!r}")
_PORT = _parsed.port or (443 if _parsed.scheme == "https" else 80)
_PATH = _parsed.path.rstrip("/") + "/v1/chat/completions"

TASK = (
    "Проанализируй текст ниже и перечисли 5 ключевых тезисов "
    "одним предложением каждый.\n\n=== ТЕКСТ ===\n"
)

PARAS = [
    (
        "Gated DeltaNet хранит матричное состояние фиксированного размера, "
        "обновляемое рекуррентно: память не растёт с длиной контекста, а "
        "стоимость шага постоянна. Это даёт линейную сложность по длине "
        "последовательности, но ограничивает точный доступ к далёким токенам: "
        "состояние сжимает историю, а не индексирует её."
    ),
    (
        "Гибридная архитектура чередует слои линейного внимания с полными "
        "слоями attention: рекуррентные слои держат локальную и сглаженную "
        "динамику, а редкие полные слои обеспечивают точный поиск по всей "
        "истории. Такой компромисс сохраняет большую часть выгоды по памяти "
        "при приемлемом качестве на длинных документах."
    ),
    (
        "Кэширование состояний для рекуррентных слоёв отличается от KV-кэша: "
        "слот состояния занимает фиксированный объём на запрос независимо от "
        "длины, но его нельзя дёшево разделить между запросами с общим "
        "префиксом, поэтому radix-кэш для таких моделей требует отдельных "
        "стратегий вытеснения и буферизации."
    ),
    (
        "Спекулятивный декодинг с черновик-моделью проверяет несколько "
        "токенов за один проход целевой модели: выигрыш зависит от доли "
        "принятых токенов и от отношения стоимости чернового прохода к "
        "целевому. При батче из нескольких запросов verify-проходы удлиняются, "
        "и выигрыш спекуляции обычно снижается."
    ),
]


def build_prompt(shift: int) -> str:
    rotated = PARAS[shift % len(PARAS):] + PARAS[: shift % len(PARAS)]
    body = ""
    cycle = 0
    while len(body) < N_CHARS:
        for j, p in enumerate(rotated):
            body += f"\n\nРаздел {cycle * len(rotated) + j}. {p}"
            if len(body) >= N_CHARS:
                break
        cycle += 1
    return TASK + body


def one_iter(i: int):
    prompt = build_prompt(i)
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
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
    try:
        conn.request(
            "POST",
            _PATH,
            body=payload,
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
    finally:
        conn.close()
    total = time.perf_counter() - t0
    if not usage:
        sys.exit("нет usage в стриме — считывать нечего")
    ptok = usage["prompt_tokens"]
    comp = usage["completion_tokens"]
    tft = ttft if ttft is not None else total
    prefill = ptok / tft if tft > 0 else 0.0
    dec = (comp - 1) / (total - tft) if comp > 1 and total > tft else 0.0
    print(
        f"iter {i}: prompt {ptok} tok | ttft {tft:.2f}s | "
        f"prefill {prefill:.0f} tok/s | decode {dec:.1f} tok/s | {comp} tok out",
        flush=True,
    )
    return ptok, tft, prefill, dec


for it in range(ITERS):
    one_iter(it)
