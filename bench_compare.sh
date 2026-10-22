#!/usr/bin/env bash
# 2026-09-30 (ночь): сравнение трёх спекуляций на одном таргете gittensor NVFP4:
#   ngram  : текущий dflash-test.env (NGRAM, пул 253k, ctx 240k)
#   dflash : тот же профиль, но SPEC_ARGS -> DFLASH (пул ~93k, ctx 90k)
#   dspark : profiles/dspark-ctx144.env (DSPARK-NVFP4, пул ~149k, ctx 145k)
# Для каждого: up -d -> health -> пул -> bench.py -> bench-hard.py -> accept из логов.
# В конце восстанавливается ngram (текущий дефолт .env).
set -uo pipefail
cd "$(dirname "$0")" || exit 1

C=sglang-qwen38-27b-nvfp4
mkdir -p results

# временный профиль DFLASH на базе текущего (отличается только SPEC_ARGS)
sed 's/^SPEC_ARGS=.*/SPEC_ARGS=--speculative-algorithm DFLASH --speculative-draft-model-path incoai\/Qwen3.8-27B-DFlash2 --speculative-num-draft-tokens 8/; s/^CONTEXT_LENGTH=.*/CONTEXT_LENGTH=85000/' \
  profiles/dflash-test.env > profiles/.dflash-bench.env

wait_health() {
  local deadline=$((SECONDS + 900))
  until curl -sf http://localhost:30000/health >/dev/null 2>&1; do
    if ! docker ps --format '{{.Names}}' | grep -qx "$C"; then echo "CONTAINER DIED"; return 1; fi
    if [ "$SECONDS" -ge "$deadline" ]; then echo "TIMEOUT health 15min"; return 1; fi
    sleep 5
  done
  echo "healthy ($((SECONDS/60))m$((SECONDS%60))s)"
}

run_one() {
  local name="$1" envf="$2"
  echo ""
  echo "===== $name ($envf) ====="
  SECONDS=0
  docker compose --env-file "$envf" up -d --force-recreate >/dev/null || { echo "UP FAILED"; return 1; }
  wait_health || return 1
  sleep 3
  echo "--- пул ---"
  docker logs "$C" 2>&1 | grep -oE "max_total_num_tokens=[0-9]+" | tail -1
  echo "--- bench.py (3 x 512 tok) ---"
  python3 bench.py http://localhost:30000 qwen3.8-27b 3 512
  echo "--- bench-hard.py ---"
  python3 bench-hard.py http://localhost:30000 qwen3.8-27b "results/$name.json" || true
  echo "--- accept после нагрузки ---"
  docker logs --since 10m "$C" 2>&1 | grep -oE "accept len: [0-9.]+" | tail -5
}

run_one ngram  profiles/dflash-test.env
run_one dflash profiles/.dflash-bench.env
run_one dspark profiles/dspark-ctx144.env

echo ""
echo "===== восстановление ngram (дефолт) ====="
docker compose --env-file profiles/dflash-test.env up -d --force-recreate >/dev/null
wait_health
echo "ALL DONE"
