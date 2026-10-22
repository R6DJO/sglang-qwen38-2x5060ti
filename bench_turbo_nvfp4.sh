#!/usr/bin/env bash
# Эксперименты DSPARK: (1) turbo = vision ON, (2) nvfp4 KV-кеш в песочнице.
# В конце восстанавливается Baseline.
set -u
cd "$(dirname "$0")" || exit 1

MAIN=sglang-qwen38-27b-nvfp4
TEST=sglang-qwen38-nvfp4-test

wait_health() {
  local port=$1 deadline=$((SECONDS + $2))
  until curl -sf "http://localhost:$port/health" >/dev/null 2>&1; do
    if [ "$SECONDS" -ge "$deadline" ]; then return 1; fi
    sleep 3
  done
}

pool_info() {
  docker logs "$1" 2>&1 | grep -oE "max_total_num_tokens[=: ]+[0-9]+" | tail -1
  docker logs "$1" 2>&1 | grep -oE "KV Cache is allocated[^\"]*" | tail -1
  docker logs "$1" 2>&1 | grep -iE "accept.{0,12}len" | tail -3
}

mkdir -p results

echo "===== 1/3 DSPARK-TURBO (vision ON) ====="
SECONDS=0
docker compose --env-file profiles/dspark-turbo.env up -d --force-recreate
if wait_health 30000 600; then
  echo "healthy (${SECONDS}s)"; sleep 3
  echo "--- пул/accept ---"; pool_info "$MAIN"
  echo "--- bench ---"
  python3 bench-hard.py http://localhost:30000 qwen3.8-27b results/dspark-turbo.json
  echo "--- accept после нагрузки ---"
  docker logs "$MAIN" 2>&1 | grep -iE "accept.{0,12}len" | tail -3
else
  echo "FAIL: turbo не поднялся"; docker logs "$MAIN" 2>&1 | tail -25
fi

echo ""
echo "===== 2/3 DSPARK + NVFP4 KV (песочница 30001) ====="
docker stop "$MAIN" >/dev/null 2>&1
SECONDS=0
docker compose -p qwen38-nvfp4-test --env-file profiles/dspark-nvfp4.env \
  up -d --force-recreate
if wait_health 30001 600; then
  echo "healthy (${SECONDS}s)"; sleep 3
  echo "--- пул/accept ---"; pool_info "$TEST"
  echo "--- bench ---"
  python3 bench-hard.py http://localhost:30001 qwen3.8-27b results/dspark-nvfp4.json
  echo "--- accept после нагрузки ---"
  docker logs "$TEST" 2>&1 | grep -iE "accept.{0,12}len" | tail -3
  echo "--- предупреждения о degradation, если есть ---"
  docker logs "$TEST" 2>&1 | grep -iE "accuracy drop|degrad|disabled.*graph" | head -5
else
  echo "FAIL: nvfp4 не поднялся -- последние логи:"
  docker logs "$TEST" 2>&1 | tail -30
fi
docker compose -p qwen38-nvfp4-test down >/dev/null 2>&1

echo ""
echo "===== 3/3 восстановление Baseline ====="
SECONDS=0
docker compose --env-file profiles/baseline.env up -d --force-recreate
if wait_health 30000 720; then echo "healthy (${SECONDS}s)"; fi
echo "ALL DONE"
