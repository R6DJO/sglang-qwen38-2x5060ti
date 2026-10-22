#!/bin/zsh
# Серия 2026-09-26: CONTEXT_LENGTH=128000 x PREFILL_CHUNK 1024/2048/4096
# (профили dspark-ctx128-pc{1024,2048,4096}.env в этом же каталоге).
# На каждом шаге: рестарт, ожидание health, пул max_total_num_tokens,
# bench.py 5x512, bench-longprompt.py (TTFT/префилл 27k), проверка OOM.
# В конце -- автоматический откат на боевой dspark-ctx144.env.
#
# РЕЗУЛЬТАТЫ 26.09: чанк не влияет ни на пул (149,414 везде), ни на
# decode (78.6-78.7 медиана); prefill 27k: 1764 / 1786-1791 / 1798-1799
# ток/с для 1024/2048/4096 (+2%, шум). OOM/крашей нет на 0.87.
# Аномалия: пул 149,414 при mamba8 (прежде 141.9-144.4k), мониторить.

set -e
cd "$(dirname "$0")/.."

for pc in 1024 2048 4096; do
  prof="profiles/dspark-ctx128-pc${pc}.env"
  echo "=============================== PREFILL_CHUNK=${pc} ==============================="
  docker compose --env-file "$prof" up -d --force-recreate >/dev/null
  echo "ожидание health (тёплый старт)..."
  for i in {1..120}; do
    curl -sf http://localhost:30000/health >/dev/null 2>&1 && break
    sleep 5
  done
  curl -sf http://localhost:30000/health >/dev/null || { echo "HEALTH TIMEOUT pc=${pc}"; exit 1; }
  sleep 5
  echo "--- пул ---"
  docker logs sglang-qwen38-27b-nvfp4 2>&1 | grep -oE "max_total_num_tokens[=: ]+[0-9]+" | tail -1
  echo "--- bench 5x512 ---"
  python3 bench.py http://localhost:30000 qwen3.8-27b 5 512
  echo "--- bench-longprompt (27k) ---"
  python3 bench-longprompt.py http://localhost:30000 qwen3.8-27b
  echo "--- аллокатор/краши в логах ---"
  docker logs sglang-qwen38-27b-nvfp4 2>&1 | grep -ciE "out of memory|can not alloc|CUDA error" || true
done
echo "================================ СЕРИЯ ЗАВЕРШЕНА ================================"
# откат на боевой профиль
docker compose --env-file profiles/dspark-ctx144.env up -d --force-recreate >/dev/null
echo "откат на dspark-ctx144 выполнен"
