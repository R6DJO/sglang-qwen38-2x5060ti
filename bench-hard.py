#!/usr/bin/env python3
"""Сложный бенчмарк профилей SGLang (OpenAI API, стриминг).

Три задачи разной природы: анализ реального проекта (README передаётся
в промпте целиком), алгоритмическое кодирование, математическая логика.
Замеряет TTFT / decode tok/s / e2e, сохраняет JSON с началами ответов
для качественного сравнения профилей.

Использование: bench-hard.py <url> <model> <out.json>
"""
import http.client
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:30000"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-27b"
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("results/run.json")

# Реальный контекст проекта: README целиком (~3k токенов префилла)
try:
    README = Path("README.md").read_text()
except OSError:
    README = "(README.md недоступен)"

PROMPTS = [
    # (имя, промпт, max_tokens, в статистику)
    ("warmup", "Ответь одним словом: столица Японии?", 64, False),
    (
        "project-analysis",
        "Ниже — README реального продакшн-деплоя SGLang на 2x RTX 5060 Ti "
        "(гибридная GDN-модель, DSPARK-спекуляция, профили памяти).\n"
        "Проанализируй конфигурацию как инженер на ревью:\n"
        "1) три самых рискованных решения и почему;\n"
        "2) топ-3 улучшения через trade-offs память/скорость;\n"
        "3) что сломается первым при росте нагрузки.\n"
        "Пиши конкретно, только по фактам из текста.\n\n"
        "=== README ===\n"
        + README
        + "\n=== КОНЕЦ README ===",
        1024,
        True,
    ),
    (
        "algo-code",
        "Реализуй на Python класс LRUKCache (LRU с K=2) с методами get/put, "
        "инвалидацией по префиксу ключа и вытеснением по TTL "
        "(время без обращения). Требования: амортизированный O(log n) на "
        "операцию, потокобезопасность, генерик-типы. Добавь юнит-тесты и "
        "краткий разбор сложности каждого метода.",
        1024,
        True,
    ),
    (
        "math-logic",
        "Задача о 12 монетах: одна фальшивая, неизвестно, легче она или "
        "тяжелее настоящих; есть чашечные весы без гирь. За 3 взвешивания "
        "найди фальшивую монету И определи, легче она или тяжелее. "
        "Опиши решение как дерево решений с разбором исходов и докажи "
        "информационным аргументом, что двух взвешиваний недостаточно.",
        1024,
        True,
    ),
]


def one_run(name, prompt, max_tokens):
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            # Без этого SGLang не присылает usage в стриме; при спекуляции
            # один SSE-чанк несёт несколько токенов -- нужен completion_tokens.
            "stream_options": {"include_usage": True},
        }
    ).encode()
    parsed = urlparse(URL)
    host = parsed.hostname or ""
    if not host:
        sys.exit(f"URL without host: {URL!r}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    conn_cls = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    conn = conn_cls(host, port, timeout=900)
    t0 = time.perf_counter()
    ttft = None
    usage = None
    finish = None
    content = []
    reasoning = []
    try:
        conn.request(
            "POST",
            (parsed.path.rstrip("/") or "") + "/v1/chat/completions",
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
            choices = obj.get("choices") or []
            if choices:
                ch = choices[0]
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                delta = ch.get("delta", {})
                if ttft is None and (
                    delta.get("content") or delta.get("reasoning_content")
                ):
                    ttft = time.perf_counter() - t0
                if delta.get("content"):
                    content.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
    finally:
        conn.close()
    total = time.perf_counter() - t0
    comp = usage["completion_tokens"] if usage else 0
    tft = ttft if ttft is not None else total
    dec = (comp - 1) / (total - tft) if comp > 1 and total > tft else 0.0
    e2e = comp / total if comp > 0 else 0.0
    text = "".join(content)
    think = "".join(reasoning)
    return {
        "name": name,
        "completion_tokens": comp,
        "thinking_tokens": len(think.split()),
        "ttft_s": round(tft, 2),
        "decode_tok_s": round(dec, 1),
        "e2e_tok_s": round(e2e, 1),
        "finish": finish,
        "answer_head": text[:500],
        "answer_len_chars": len(text),
    }


OUT.parent.mkdir(parents=True, exist_ok=True)
runs = []
for name, prompt, max_tokens, in_stats in PROMPTS:
    r = one_run(name, prompt, max_tokens)
    r["in_stats"] = in_stats
    runs.append(r)
    print(
        f"{name:18s} tok={r['completion_tokens']:5d} ttft={r['ttft_s']:6.2f}s "
        f"decode={r['decode_tok_s']:6.1f} e2e={r['e2e_tok_s']:6.1f} "
        f"finish={r['finish']}",
        flush=True,
    )

stats = [r for r in runs if r["in_stats"]]
if stats:
    med = sorted(r["decode_tok_s"] for r in stats)
    print(
        f"\nMEDIAN decode (3 задач): {med[len(med) // 2]:.1f} tok/s | "
        f"runs: {[r['decode_tok_s'] for r in stats]}",
        flush=True,
    )
OUT.write_text(
    json.dumps({"url": URL, "model": MODEL, "runs": runs}, ensure_ascii=False, indent=2)
)
print(f"saved: {OUT}", flush=True)
