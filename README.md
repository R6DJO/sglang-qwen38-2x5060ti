# Qwen3.8-27B-NVFP4 на SGLang (2× RTX 5060 Ti 16GB)

Чекпойнт: `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` — NVFP4 W4A4 (ModelOpt),
FP8 KV-cache, мультимодальная модель (текст + картинки/видео), гибридная
архитектура: 64 слоя = 48×GatedDeltaNet + 16×full attention, нативный контекст 262k.

NVFP4 работает **только на Blackwell** (ваши 5060 Ti = sm_120 — поддерживаются).
На одной 16GB карте веса (~18.8GB) не помещаются, поэтому запуск через **TP=2**.

Обычный режим — **Baseline**: без спекуляции, с vision, полным
контекстом 192k и конкурентностью 2 (пул ~229–257k токенов,
~40 tok/s decode). Для одиночной интерактивной работы есть
**DSPARK-ускорялка** — она же текущий дефолт: контекст 155k,
пул 157.9k, single-stream, vision-off, медиана 69 tok/s decode
(живой трафик, см. «Протокол 2026-10-22»). Профили переключаются
env-файлом из `profiles/`, см. «Профили»; голый `docker compose up`
запускает `.env` — это симлинк на `profiles/dspark-ctx145.env`.

## Запуск

```bash
cd sglang-fp8                 # каталог, куда распакован репозиторий
docker compose up -d
docker compose logs -f        # ждать строку "The server is fired up and ready to roll!"
```

JIT-кэш кернелов персистентен (`./sgl-cache` → `SGLANG_CACHE_DIR=/sgl-cache`):
рестарт без перекомпиляции — warm-старт 129с против холодных 136–160с
(замер 2026-09-18, Baseline; кэш заполняется при первом старте).

Проверка:

```bash
curl -s http://localhost:30000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Привет!"}]}'
```

## Профили: Baseline, DSPARK-ускорялка, Longctx

Профили на одном порту 30000, mutually-exclusive (GPU заняты одним
сервером — одновременно нельзя). Каждый профиль = env-файл в `profiles/`;
механический дефолт `.env` = DSPARK:

| | **Baseline** | **DSPARK ctx145** (текущ., дефолт `.env`) | **Longctx** |
| --- | --- | --- | --- |
| Спекуляция | нет | DSPARK NVFP4-драфтер | нет |
| Decode, 1 клиент | ~40.6 tok/s | **медиана 69 tok/s** (p10 49, p90 102; живой трафик 15.10) | ~43.5 tok/s |
| 2 парал. клиента | ~38.8×2 = **77 tok/s** | ❌ очередь, e2e 14–20 tok/s + риск краша mamba-пула | ❌ single-stream |
| Контекст | **192k** (196,608; пул ~229–257k) | **155k** (пул 157.9k, стабильно между рестартами) | 262k (пул ~302k) |
| Vision (картинки/видео) | ✅ | ❌ (vision-off) | ❌ (vision-off) |
| Mamba | 8 слотов, MAX_RUNNING=2 | 8 слотов + skip-decode-lock, MAX_RUNNING=1 | 4 слота, MAX_RUNNING=1 |
| Образ | `latest-cu130` (v0.5.20) | `dev-cu13` (сборка 26.09) | `latest-cu130` (v0.5.20) |
| Конфиг | `profiles/baseline.env` | `.env` → `profiles/dspark-ctx145.env` | `profiles/longctx.env` |

⚠️ Пулы Baseline/Longctx («~229–257k», «~302k») мерялись **без**
`SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK`, который теперь стоит в дефолте
compose (см. «Протокол 2026-10-22») — при этих профилях пул станет чуть
больше, цифры требуют перепроверки.

Ещё два профиля DSPARK-семейства: **Turbo** (`profiles/dspark-turbo.env`) —
то же, что DSPARK, но vision включён (пул ~121k; в файле CONTEXT_LENGTH=88000);
по замеру 2026-09-18
на 0.87 скорости НЕ добавил (64.3 vs 67.9 медиана) — старые 95.4 tok/s
были при 0.86, см. «Vision и контекст». И **dspark-nvfp4** — эксперимент
с FP4 KV-кешем, пока заблокирован upstream (см. соответствующий раздел).

**Longctx на gittensor** (`profiles/gittensor-longctx.env`, замер 2026-09-29) —
тот же gittensor NVFP4-чекпойнт, но без драфтера (SPEC_ARGS= гасит DSpark-дефолт):
освободившиеся 1.4 GB идут в KV, пул 293,938, контекст 260k (лимит чекпойнта
262,144 — выше ValueError). Decode 43.6 tok/s (медиана 5×512) — быстрее
longctx (~43.5 на baseline-весах) при большем контексте. Целевой профиль
под длинный контекст на 2×16GB. См. также «Протокол 2026-09-29» про то,
почему нативный MTP сюда не влезает.

```bash
docker compose --env-file profiles/gittensor-longctx.env up -d --force-recreate
```

Включить Baseline (vision, 192k, conc 2):

```bash
docker compose --env-file profiles/baseline.env up -d --force-recreate
```

Включить DSPARK-ускорялку:

```bash
docker compose up -d --force-recreate
```

Включить Longctx:

```bash
docker compose --env-file profiles/longctx.env up -d --force-recreate
```

Проверить, что активно:

```bash
docker logs sglang-qwen38-27b-nvfp4 2>&1 | grep -oE "'speculative_algorithm': [A-Za-z']+" | tail -1
# None → Baseline; 'DSPARK' → ускорялка
docker logs sglang-qwen38-27b-nvfp4 2>&1 | grep max_total_num_tokens | tail -1
```

⚠️ На DSPARK-профиле не слать параллельные запросы (MAX_RUNNING=1,
MAMBA_CACHE=8 + `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=true`): второй
одновременный клиент ронял шедулер ассертом `Can not alloc mamba cache`
(замер 2026-09-18, на mamba4; с 8 слотами не воспроизводится).
Пул на ctx145 стабилен между рестартами (157,879–157,882),
`CONTEXT_LENGTH=155000` даёт запас ~2.9k; если `max_total_num_tokens`
в логах окажется ниже 155000 — опустить CONTEXT_LENGTH (напр. 150000).

Замер 2026-09-18 (вечер, `bench.py`, temp=0, 512 ток., медиана из 5):
DSPARK 57.0 / Baseline 40.6 одиночный поток; при 2 клиентах Baseline
38.8+38.8, DSPARK суммарно ~29 tok/s effective (очередь). Утренний замер
DSPARK давал до 78 tok/s — разброс между прогонами большой, но порядок
«DSPARK ×1.4–1.8» стабилен.

## ⚠️ Диск

Нужно суммарно **~55GB** (образ ~14GB + модель ~21GB + распаковка); проверьте
свободное место перед запуском (`df -h /`). Если места впритык — освободите его
(`docker system prune -a`) или вынесите кэш модели на другой диск:
`HF_CACHE_DIR=/mnt/.../hf-cache` в `.env`.

## Что настроено и почему (профиль DSPARK; остальные — profiles/*.env)

| Параметр | Значение | Причина |
| --- | --- | --- |
| `--tp-size 2` | — | Веса 18.8GB не влезают в одну 16GB карту |
| `--attention-backend flashinfer` | — | Единственный валидный бэкенд для SM120 (trtllm_mha — только SM100) |
| `--kv-cache-dtype fp8_e4m3` | — | Чекпойнт откалиброван под FP8 KV |
| `--json-model-override-args {"language_model_only":true}` | — | Не грузить vision-башню (0.92GB, реплицирована на обе карты): +пул ценой off-distribution драфтера; см. «Vision и контекст» |
| `--speculative-algorithm DSPARK` | + NVFP4-драфтер | Ускорялка: ×1.4–1.8 к decode; см. раздел ниже |
| `--speculative-draft-model-quantization modelopt_fp4` | — | Драфтер 0.7GB/карту вместо 2.2 (bf16): +55k токенов KV и даже быстрее |
| `--speculative-dspark-block-size 7` | — | Как в конфиге драфтера (gamma=7, verify 8 ток.) |
| `--mem-fraction-static 0.883` | .env | Бюджет "веса target+draft + KV" (v0.5.x); на ctx145 (mamba8 + skip-decode-lock + GRAPH_BS=1) 0.883 держит скорость: пул 144.4k@0.87 → 157.9k, «Disable DSpark draft cuda graph» в логах нет. Без этих рычагов 0.88 давал −25…−33% (см. «Vision и контекст») |
| `--chunked-prefill-size 1024` | .env | На старых билдах >1024 подвешивал декод на GDN; на v0.5.20 не воспроизводится (замер 21.09: 2048/4096 — пул и скорость идентичны). 1024 оставлен — рецепт карточки драфтера |
| `--mamba-ssm-dtype bfloat16` | — | GDN state-слот 78MB вместо 154MB (fp32) — почти вдвое больше KV |
| `--mamba-radix-cache-strategy extra_buffer_lazy` | — | Обязательна для DSPARK; 4 слота на запрос |
| `--max-mamba-cache-size 8` | .env | Single-stream: 8 слотов (стратегия extra_buffer_lazy + skip-decode-lock). Раньше было 4 — на нём падал `Can not alloc mamba cache`; каждый +1 слот ≈ −1.8k пула |
| `--context-length 155000` | .env | Лимит на запрос = реальный пул (157.9k на ctx145, запас ~2.9k; при пуле ниже лимита — опускать, см. «Профили») |
| `--cuda-graph-max-bs-decode/prefill 1` | .env | На ctx145 GRAPH_BS=1 (при 1024-чанке 30.09 раздувало decode-графы и роняло prefill-захват по OOM); на пул 1↔4 не влияет (замер 21.09) |
| `--reasoning-parser qwen3` + `--tool-call-parser qwen3_coder` | — | Парсеры thinking и tool-calls; hermes не подойдёт — формат другой |
| `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=true` | compose-дефолт | Cookbook-оптимизация: −1 mamba-слот на запрос (extra_buffer_lazy 4→3) → +пул. В образе dev-cu13 это `EnvBool(False)`, парсер принимает `true/1/yes` — проверено. **Действует на все профили сразу** |
| `SGLANG_TRITON_LOAD_WARNING_THRESHOLD_GB=0.3` | compose-дефолт | Triton late-load watchdog (дефолт 1.0 GB) трещал при профиле 0.87+: в момент late-loads free 0.84–0.88 GB. Порог снижен, OOM-риск не меняется (загрузка ядер с 0.84+ free комфортна, замер 30.09) |

## Тюнинг после первого запуска

Смотрите в логах строки `max_total_num_tokens` и `max_running_requests`
(при DSPARK — ещё `Disable DSpark draft cuda graph`: если появилась,
слэк мал, спекуляция деградировала — верните `MEM_FRACTION` ниже):

- **Рестарт-луп при старте** → в v0.5.x бюджет включает веса драфтера;
  `MEM_FRACTION` ниже ~0.73 не стартует совсем
- **Нужно больше контекста** → `MEM_FRACTION=0.87–0.883` (по +25k токенов
  на +0.01, следить за draft cuda graph). 0.883 подтверждён на ctx145:
  `available_gpu_mem` после аллокации пула 0.93 GB, скорость не просела
- **Нужна бОльшая скорость без длинного контекста** → уже включено (DSPARK)
- **Хотите длинный контекст вместо скорости** → убрать все
  `--speculative-*` флаги и `--mamba-radix-cache-strategy`: пул станет
  265k, но 43.5 tok/s и `MAMBA_CACHE=16`, `MAX_RUNNING=4`

## Спекулятивный декодинг — DSPARK (текущий конфиг)

> Замеры ниже — 18.09, профиль того дня (0.86–0.87, mamba4, без
> skip-decode-lock). Текущий ctx145: 0.883, контекст 155k, пул 157.9k —
> см. «Протокол 2026-10-22».

Замер 2026-09-18 (sglang v0.5.20, `latest-cu130`, бенч `bench.py`:
стриминг, подсчёт по `usage.completion_tokens`, temp=0, 512 ток.,
одиночный поток):

| Режим | Decode tok/s | E2E tok/s | KV-пул | Accept len |
| --- | --- | --- | --- | --- |
| **DSPARK NVFP4 + vision-off (осн.)** | **78.4** | 75.2 | 121.8k | 2.74 |
| DSPARK NVFP4 + vision | 95.4 | 84.3 | 93.5k | 3.26 |
| DSPARK NVFP4 + vision-off @0.88 | 59.2 | 57.3 | 136.8k | 2.75 |
| DSPARK bf16-драфтер (RadixArk) | 75.1 | 72.1 | 35.5k | 2.80 |
| Без спекуляции | 43.5 | 42.3 | 265.7k | — |

Выбор NVFP4-драфтера (`gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4`,
1.4GB) неочевиден, но выигрывает по всем метрикам: быстрее И accept выше,
и пул вдвое с лишним больше, чем с bf16-драфтером RadixArk (3.7GB).

⚠️ Ранний вывод «спекуляция медленнее» (DSPARK 33.5 tok/s) был артефактом
замера: при спекулятивном декодинге один SSE-чанк несёт сразу несколько
токенов, и подсчёт чанков вместо `usage.completion_tokens` занижает
скорость почти вдвое. Без спекуляции чанк = токен, поэтому старая база
39–41.5 была честной (сегодняшняя — 43.5).

Цена DSPARK против режима без спекуляции: пул 265k → 90k, concurrency
4 → 2. Если когда-нибудь станет критичен контекст >90k — откат описан
в «Тюнинг» выше.

Эксперименты (bf16-драфтер, другие ручки) — песочница
`profiles/dspark-test.env` (порт 30001, отдельный проект/контейнер;
спекуляция настраивается строкой `SPEC_ARGS` целиком, примеры в самом
файле; сначала остановить основной сервис — GPU общие). Запуск:
`docker compose -p qwen38-dspark-test --env-file profiles/dspark-test.env
up -d`, останов: `docker compose -p qwen38-dspark-test down`. Бенч:
`python3 bench.py [url] [model] [iters] [max_tokens]`.

Если когда-нибудь переедет на одиночную карту 32GB (RTX 5090/PRO 6000) —
дополнительно попробовать EAGLE/MTP без межкарточного allreduce:

```yaml
      --speculative-algorithm EAGLE
      --speculative-num-steps 3
      --speculative-eagle-topk 1
      --speculative-num-draft-tokens 4
```

## Vision и контекст (тройной trade-off)

Три измеримых режима (замер 2026-09-18, одинаковый бенч):

| Режим | Пул | Decode | Vision | Примечание |
| --- | --- | --- | --- | --- |
| vision @0.86 | 93.5k | 95.4 tok/s | да | accept len 3.26, максимум скорости |
| **vision-off @0.86 (база замеров 18.09)** | 121.8k | 78.4 tok/s | нет | accept len 2.74, +28k контекста |
| vision-off @0.88 | 136.8k | 59.2 tok/s | нет | без mamba8/skip-lock: аллокатору тесно |

Текущий ctx145 стоит на **0.883** и этот пункт не опровергает: там
пачка рычагов (mamba8 + skip-decode-lock + GRAPH_BS=1), пул 157.9k,
`available_gpu_mem` 0.93 GB, медиана decode 69 tok/s по живому трафику,
`Disable DSpark draft cuda graph` за 6 дней работы не появлялся.

- **vision-off**: `LANGUAGE_MODEL_ONLY=true` (дефолт compose; рендерится
  в `--json-model-override-args {"language_model_only":true}`).
  Vision-башня (0.92GB, на каждой карте целиком)
  не грузится, mm-резерв возвращается в KV. Картинки/видео запросы падают.
  Нюанс: флаг также отключает mrope -> текстовое распределение слегка
  отличается от reference (драфтер, калиброванный под mrope, угадывает
  хуже: accept 3.26 -> 2.74, отсюда -18% скорости). Верификация всё равно
  гарантирует корректность выходов, но поведение чуть off-distribution.
- **0.88 не работает** (замерено без рычагов mamba8/skip-lock/GRAPH_BS=1,
  18.09): при ~0.85GB свободных шаг верификации дорожает
  34 -> 46 мс (аллокатор PyTorch начинает реаллокации), скорость -25%.
  С рычагами 0.883 работает (см. таблицу выше и «Протокол 2026-10-22»).
- **Повтор 2026-09-18 при 0.87**: turbo (vision ON) — 64.3 tok/s медиана,
  пул 121k, accept 2.8-3.5. «+18% скорости» из таблицы выше НЕ
  воспроизвелось: тот замер был при 0.86. С vision-башней при 0.87
  бюджет аллокатора входит в «тесную» зону (аналог эффекта 0.88).
  Если нужен turbo — пробовать MEM_FRACTION=0.86 (пул упадёт до ~95k).
- Вернуть vision: `LANGUAGE_MODEL_ONLY=false` в `.env`/профиле
  (пул станет ~121k при 0.87, скорость — как у дефолта).

## NVFP4 KV-кеш — заблокирован upstream (проверено 2026-09-18)

Пробовали `--kv-cache-dtype nvfp4` с DSPARK (цель: пул ×1.4–1.8
при сохранении спекуляции). Пул действительно аллоцируется
(209 973 токена против 150 150 на fp8), но ни один доступный
на SM120 бэкенд MHA-путь не поддерживает (`arg_groups/kv_cache_hook.py`):
flashinfer — assert (разрешён только MLA-моделям), triton —
`KeyError: 'float4_e2m1fn_x2'` в extend-кернеле, trtllm_mha — нет
сборки под SM120. На main ограничение то же. **Перепроверено 2026-09-29**
на том же образе (v0.5.20, gittensor + triton): пул аллоцируется
(**470,301 токен** — вот цена блокировки: до ~460k контекста),
и тот же `KeyError: 'float4_e2m1fn_x2'`. Задел оставлен
в `profiles/dspark-nvfp4.env`: когда MHA-путь доведут — снять пометку
и прогнать бенч заново.

## API

- OpenAI: `http://localhost:30000/v1` (model = `qwen3.8-27b`)
- Anthropic-совместимый: `http://localhost:30000` (`/v1/messages`)
- Thinking включён по умолчанию; выключение per-request:
  `"chat_template_kwargs": {"enable_thinking": false}`
- Модель понимает картинки/видео (vision tower активна)
  *(в DSPARK-профилях башня не грузится — картинки не принимает; это норма для них)*

---

## Замеры 2026-09-21: новые профили, intel_iommu=off

Тот же bench.py (5×512, temp=0, медиана), после `intel_iommu=off`:

| Профиль | Decode 1 поток | 2 парал. | Пул |
| --- | --- | --- | --- |
| **dspark-ctx150** (новый; сейчас mamba8, см. аддендум) | **78.1** (e2e 75.0) | ⛔ | 151,683 |
| baseline | 43.2 (e2e 42.3) | 75.8 | 256,740 |
| dspark-maxctx (новый, эксперимент) | 59.1 (−24%) | ⛔ | 170,431 |

- DSPARK 78.1 — выше прежней вилки 57–72; вероятный вклад +7% от iommu-off
  (воспроизведено и на vLLM: 35.6 → 38.1 ток/с).

## Замеры 2026-09-29: длинный контекст и нативный MTP

Попытка поднять контекст через нативный MTP (чекпойнт со встроенной MTP-головой,
`--speculative-algorithm EAGLE 3/1/4` — рецепт из официального SGLang cookbook):

| Конфигурация | Пул | Decode 5×512 | Контекст |
| --- | --- | --- | --- |
| dspark-ctx144 (точка отсчёта) | 148,989 | 78.2 | 144k |
| nvidia/Qwen3.8-27B-NVFP4 + EAGLE | — | — | рестарт-луп: `Loaded weights leave no GPU memory` |
| nvidia/Qwen3.8-27B-NVFP4, без спек. | 196,786 | 38.2 | 185k |
| **gittensor RTX5090, без спек.** (gittensor-longctx) | **293,938** | **43.6** | **260k** |

- MTP-голова nvidia-чекпойнта в bf16 весит **2.82 GB/карту** (замер из логов
  загрузки; это полный слой+эмбеддинги, не «маленькая голова»). При 0.87 после
  весов (target 9.72 + draft 2.82 = 12.54 из 13.05 GB бюджета) на KV остаётся
  ~0.5 GB — рестарт-луп. Арифметика потолка: даже при 0.95 на пул ~1.7 GB/карту
  ≈ 50–60k контекста. **Вывод: нативный MTP и контекст >144k на 2×16GB
  несовместимы.**
- nvidia-чекпойнт без спекуляции проигрывает gittensor по всему: пул −97k
  и decode −12% (38.2 vs 43.6) — тяжелее веса (9.72 против 8.96 GB/карту;
  вероятно, vision-часть и bf16-хвосты).
- gittensor без драфтера: +1.4 GB → пул 293,938; CONTEXT_LENGTH ограничен
  **производным лимитом чекпойнта 262,144** (при 280000 — ValueError без
  `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`; не поднимать). Итог 260k.
- Цена: decode 78.2 → 43.6 (−44%, ушла спекуляция ×1.4–1.8). Обратный путь —
  `profiles/dspark-ctx144.env`.
- Файлы: профиль `profiles/gittensor-longctx.env` (переименован из
  nvfp4-mtp-ctx.env в процессе), compose теперь берет модель из
  `MODEL_PATH` (дефолт = gittensor, остальные профили не затронуты).
- **Аддендум (тот же день, 10:28 UTC):** шедулер упал на реальном трафике
  (`Can not alloc mamba cache` — чанковый префилл во время декода, 3/4
  mamba-слотов заняты кэшем). Профиль переведён на mamba8: каждый +1 слот
  ≈ −1.8k пула, итого пул 151.7k → 144.4k, CONTEXT_LENGTH 150k → 140k
  (запас ~4.4k на плавание пула). Скорость не изменилась: 78.2 (e2e 74.9),
  перемеряно тем же bench.py.
- **Аддендум (тот же день, вечер): резервы по официальной документации**
  (docs.sglang.io + карточки HF `gittensor-model-hub`). Итог: конфигурация
  прижата к официальному потолку. Веса в кэше — обновлённый main-чекпойнт
  `5b7a687f` (09-10): NVFP4 `lm_head`, без MTP-головы, 2 шарда 17.92GB
  (это и есть «strictly better» вариант из карточки LMHead4: +8–15% decode
  на 5090, точность без деградации — скачан ещё 09-18). Живые замеры:
  MAX_RUNNING=2 на mamba8 краша не даёт, но пул −14.6k (144,387→129,795)
  и суммарный throughput −13% (67.6 против 78.1) — оставлен MAX_RUNNING=1;
  `GRAPH_BS` 1↔4 и `PREFILL_CHUNK` 1024/2048/4096 — мёртвые ручки (пул и
  скорость идентичны; «>1024 подвешивает декод» на v0.5.20 не
  воспроизводится); `--speculative-attention-mode decode` −1.3%;
  flashinfer allreduce fusion на SM120 отклоняется v0.5.20 (гейт
  SM90/SM10X); образ dev-cu13 (09-01) по скорости идентичен latest.
  No-MTP вариант чекпойнта VRAM не экономит (mtp.* не аллоцируется).
  Клиентский рычаг для pi: `reasoning_effort=low` −25% / thinking off −73%
  времени на рутинном шаге. accept len в наших логах 2.40 против 2.89
  в карточке — нижняя граница диапазона, мониторить. Новые инструменты:
  `bench-par2.py`, `bench-longprompt.py`; в compose — no-op плейсхолдер
  `${EXTRA_ARGS-}`. Профиль 21.09 переименован:
  `dspark-ctx150.env` → `dspark-ctx140.env` (имя «ctx150» было
  историческим, контекст давно 140k).
  26.09: контекст поднят 140k → 144k, профиль переименован
  `dspark-ctx140.env` → `dspark-ctx144.env` (имя теперь совпадает
  с контекстом).
- **`profiles/dspark-ctx150.env`** — DSPARK c CONTEXT_LENGTH=150000 при 0.87:
  +2k контекста бесплатно (пул 151.7k ≥ контекста). Риск: пул плавает
  (148.7–151.7k), после рестарта проверять `max_total_num_tokens` ≥ 150000,
  иначе понизить CONTEXT_LENGTH (инструкция в шапке профиля).
  *(С момента аддендума: mamba8, CONTEXT_LENGTH=140000 — см. выше.)*
- **`profiles/dspark-maxctx.env`** — mem-fraction 0.895 (+400 МиБ/карту):
  пул +18.7k и контекст 155k, но −24% шага (59.1). Обрыв скорости за 0.87
  подтверждён и с рычагами (ранее: 0.88 без рычагов −33%). Не включать
  без необходимости.
- **intel_iommu=off** в GRUB: +7% decode обоим стекам. Побочка: нет
  PCI-passthrough в VM.
- **P2P между 5060 Ti закрыт**: при выключенных IOMMU (dmesg «DMAR: IOMMU
  disabled») и ACS (ACSCtl все «-») `cudaDeviceCanAccessPeer` всё равно
  False — политика драйвера 610.57.04 для этой пары GeForce. Custom
  allreduce остаётся выключен у обоих стеков; тема закрыта.
- Четвёртый стек, vLLM с DFlash2 (см. `../vllm-sm12x-nvfp4-dflash2/AGENTS.md`):
  12.8 ток/с против 38.1 у того же vLLM без спеки — их спек-цикл eager/вне
  графов; на 4 параллельных у vLLM лучший агрегат дня — 133 ток/с.

---

## Профиль ctx145 @ 0.883 (2026-10)

Дефолт `.env` → `profiles/dspark-ctx145.env`: контекст 155,000 @ `MEM_FRACTION=0.883`,
mamba 8 слотов + `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=true`, `GRAPH_BS=1`.

- Пул `max_total_num_tokens` 157,879–157,882 (стабильно между рестартами),
  `available_gpu_mem` после аллокации пула 0.93 GB.
- Медиана decode 69.0 ток/с (p10 49, p90 102, n=3488), accept len драфтера 2.34
  против 2.89 в карточке — нижняя граница диапазона, мониторить.
- Ограничение «mem-fraction >0.87 = обрыв скорости» (замер 21.09) к этой
  конфигурации не применимо: оно снималось без mamba8, skip-decode-lock и
  GRAPH_BS=1. С ними 0.87→0.883 даёт +13.5k пула (144.4k → 157.9k).
- `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=true` в дефолте compose действует на все
  профили, поэтому пулы Baseline (256,740) и gittensor-longctx (293,938),
  задокументированные ранее, мерялись без него и требуют перепроверки.
- Порог `SGLANG_TRITON_LOAD_WARNING_THRESHOLD_GB` опущен 0.7 → 0.3.
- Дефолтный профиль на образе `dev-cu13`, остальные — на `latest-cu130`
  (v0.5.20); в замерах 29.09 образы по скорости совпадали.
- Чистого `bench.py` на этом профиле не прогано: цифры взяты из логов
  работающего сервиса, для сравнения с базой 78.2 (ctx144) нужен прогон.
