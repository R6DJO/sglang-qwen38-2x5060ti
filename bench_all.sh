#!/usr/bin/env bash
# Прогон трёх профилей: старт -> ожидание health -> пул/accept из логов
# -> bench-hard.py (3 сложные задачи). В конце восстанавливается Baseline.
set -euo pipefail
cd "$(dirname "$0")" || exit 1

C=sglang-qwen38-27b-nvfp4

wait_health() {
  local deadline=$((SECONDS + 720))
  until curl -sf http://localhost:30000/health >/dev/null 2>&1; do
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "TIMEOUT: сервер не поднялся за 12 мин"; exit 1
    fi
    sleep 5
  done
  echo "healthy (${SECONDS}s)"
}

pool_info() {
  docker logs "$C" 2>&1 | grep -oE "max_total_num_tokens[=: ]+[0-9]+" | tail -1 || true
  docker logs "$C" 2>&1 | grep -iE "accept.{0,12}len" | tail -3 || true
}

mkdir -p results
for spec in "dspark:.env" "baseline:profiles/baseline.env" "longctx:profiles/longctx.env"; do
  name="${spec%%:*}"; envf="${spec#*:}"
  echo ""
  echo "===== PROFILE: $name ($envf) ====="
  SECONDS=0
  docker compose --env-file "$envf" up -d --force-recreate
  wait_health
  sleep 3
  echo "--- пул/accept ---"
  pool_info
  echo "--- bench ---"
  python3 bench-hard.py http://localhost:30000 qwen3.8-27b "results/$name.json"
  echo "--- accept после нагрузки ---"
  pool_info
done

echo ""
echo "===== восстановление Baseline (текущий профиль) ====="
SECONDS=0
docker compose --env-file profiles/baseline.env up -d --force-recreate
wait_health
echo "ALL DONE"
