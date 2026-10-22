# AGENTS.md — sglang qwen3.8-27b на 2× RTX 5060 Ti

Оперативная шпаргалка для агентов/человека. Полный разбор — README.md.

## Текущий профиль и замеры (2026-10-22)

- **Текущий профиль: dspark-ctx145** (`.env` → симлинк на
  `profiles/dspark-ctx145.env`). Порт **30000**, модель
  `qwen3.8-27b` (OpenAI `/v1`, Anthropic `/v1/messages`). DSPARK + vision-off,
  `MEM_FRACTION=0.883`, `CONTEXT_LENGTH=155000`, `MAMBA_CACHE=8`, `GRAPH_BS=1`,
  `MAX_RUNNING=1`, образ `dev-cu13`.
- Замеры (логи живого прогона): пул **157,879–157,882** (стабильно между
  рестартами), `available_gpu_mem` после аллокации **0.93 GB**; медиана decode
  **69.0 ток/с** (p10 49, p90 102, max 214, n=3488), accept len **2.34**,
  максимум занятого KV 100,058. Watchdog / OOM / `Can not alloc mamba cache` /
  «Disable DSpark draft cuda graph» в логах нет. Чистого `bench.py` на 155k
  не прогано (для сравнения с базой 78.2 нужен прогон).
- **gittensor-longctx больше не текущий** — остался целевым профилем под
  длинный контекст: контекст 260,000 (ЛИМИТ чекпойнта 262,144 — выше ValueError,
  не поднимать), пул `max_total_num_tokens=293,938` (замер 29.09, запас ~34k),
  decode **43.6 ток/с** (медиана 5×512). Цена длинного контекста: −44% decode
  против DSpark (78.2 → 43.6). Пул 293,938 мерян без skip-decode-lock — перемерять.
- **Нативный MTP не лезет** (замер 29.09): nvidia/Qwen3.8-27B-NVFP4 + EAGLE —
  MTP-голова bf16 = 2.82GB/карту, рестарт-луп при 0.87 ("Loaded weights leave
  no GPU memory"), потолок при 0.95 ~ 50–60k контекста. На 2×16GB нативный
  MTP и >144k несовместимы. Сам nvidia-чекпойнт без спекуляции тоже хуже:
  пул 196,786, decode 38.2 (тяжелее веса; см. profiles/gittensor-longctx.env).
- **Веса уже обновлены**: ревизия `5b7a687f` (09-10) — NVFP4 `lm_head`,
  без MTP-головы, 2 шарда 17.92GB. Это «strictly better» апгрейд из
  карточки LMHead4 (+8–15% decode на 5090, точность без деградации) —
  скачан 09-18, ничего делать не нужно.
- **mamba8** (`MAMBA_CACHE=8` вместо 4): 10:28 UTC шедулер падал
  (`Can not alloc mamba cache` на чанковом префилле во время декода —
  3/4 слотов были заняты кэшем). mamba8 почти бесплатен: −7.3k пула,
  контекст 150k→140k, скорость та же. Каждый +1 слот ≈ −1.8k пула.
- **Конкурентность = 1** (`max_running_requests=1`): на mamba8 перекрытие
  запросов краша НЕ даёт (проверено 21.09: 2×512 с перекрытием, health 200,
  23 записи `#running-req: 2`), но включать незачем: −14.6k пула
  (144,387→129,795) и суммарный throughput −13% (67.6 против 78.1).
  Для pi (запросы строго последовательные) смысла нет.
- **Vision выключен** (`language_model_only=true`): картинки не принимает.
- **HF_TOKEN**: проброшен в контейнер через `environment` в docker-compose.yml
  (`- HF_TOKEN=${HF_TOKEN:-}`), huggingface_hub читает его с приоритетом над
  token-файлом. Значение задаётся в shell-окружении хоста (export), в профили
  и в репозиторий **не вписывается**.
  Пул после рестарта с токеном: 149,414.

## Профили (переключение перезапускает контейнер, ~2 мин тёплый старт)

| Профиль | Команда | Контекст | Параллель | Vision |
| --- | --- | --- | --- | --- |
| `profiles/dspark-ctx145.env` ← текущий (дефолт `.env`) | `docker compose --env-file profiles/dspark-ctx145.env up -d --force-recreate` (он же = голый `docker compose up -d`) | 155k (пул 157.9k) | ❌ | ❌ |
| `profiles/gittensor-longctx.env` (без спекуляции, 260k) | `docker compose --env-file profiles/gittensor-longctx.env up -d --force-recreate` | 260k | ❌ | ❌ |
| `profiles/dspark-ctx144.env` (mamba8, быстрый 78 ток/с) | `docker compose --env-file profiles/dspark-ctx144.env up -d --force-recreate` | 144k | ❌ | ❌ (в файле `MEM_FRACTION=0.89` — дрейф, не замерен) |
| `profiles/baseline.env` | `docker compose --env-file profiles/baseline.env up -d --force-recreate` | 192k | ✅ до 2 | ✅ |
| `profiles/longctx.env` | аналогично | 262k | ❌ | ❌ |
| `profiles/dspark-turbo.env` | аналогично | ~121k пул | ❌ | ✅ (в файле `CONTEXT_LENGTH=88000`) |
| `profiles/dspark-maxctx.env` | аналогично | 155k | ❌ | ❌ (−24% скорости, см. ниже) |

⚠️ Задокументированные пулы Baseline (256,740), longctx/gittensor-longctx
(~302k / 293,938) мерялись **без** `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK`,
который теперь включён в дефолт compose — у этих профилей пул станет чуть
больше, цифры требуют перепроверки.

## Грабли

- **Пул плавает между рестартами** (при mamba4 было 148.7–151.7k, при mamba8
  141.9–144.4k; ctx145 @0.883 — 157,879–157,882, стабильно на трёх стартах).
  После каждого
  старта проверять: `docker logs sglang-qwen38-27b-nvfp4 2>&1 | grep -oE "max_total_num_tokens[=: ]+[0-9]+" | tail -1`.
  Если пул < CONTEXT_LENGTH профиля — понизить CONTEXT_LENGTH (иначе краш
  аллокации на длинном запросе).
- **mem-fraction: «>0.87 = обрыв скорости» снят при рычагах.** 0.883 на ctx145
  (mamba8 + skip-decode-lock + GRAPH_BS=1) держит скорость: пул 144.4k → 157.9k,
  медиана 69 ток/с, «Disable DSpark draft cuda graph» не появлялся, free 0.93 GB.
  Без этих рычагов обрыв подтверждён: 0.895 дал 59.1 против 78.1
  ток/с (−24%) при +18.7k пула; 0.88 ранее −25…33%.
- **`SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=true` включён в дефолт compose** —
  освобождает 1 mamba-слот на запрос (extra_buffer_lazy 4→3), т.е. +пул.
  В образе это `EnvBool(False)`, парсер принимает `true/1/yes/y` (проверено);
  дока-калькулятор в `docs-sglang/Qwen3.8-27B.md:2540` проверяет только `=1`,
  поэтому его расчёт слотов при `true` занижен на один слот. Дефолт бьёт по
  всем профилям сразу.
- `--chunked-prefill-size` >1024: подвисание декода на v0.5.20 НЕ
  воспроизводится (замер 21.09: 2048/4096 — пул и скорость идентичны,
  префилл ~1.75k ток/с везде); 1024 оставлен как рецепт карточки драфтера.
  `GRAPH_BS` (cuda-graph bs) 1↔4 тоже не влияет ни на пул, ни на скорость —
  4 оставить как бесплатный запас под MAX_RUNNING=2.
- `--enable-flashinfer-allreduce-fusion` на SM120 отклоняется v0.5.20
  (`ValueError: requires SM90 or SM10X`); образ dev-cu13 (09-01) по
  скорости идентичен latest-cu130. **Дрейф: дефолтный профиль ctx145 сейчас
  на `dev-cu13` (сборка 26.09, `sglang 0.0.0.dev1+g1f6ce4b06`)**, остальные
  профили — на `latest-cu130` (v0.5.20).
- **Пути**: том кэша кернелов — `./sgl-cache` (compose монтирует его в
  `/sgl-cache`, `SGLANG_CACHE_DIR=/sgl-cache`); старое имя тома `./sglang-cache`
  осиротевшее, в `.gitignore`.
- **Сырой вывод модели и бенч-скрипты, работающие с локальными проектами, в репу
  не кладутся** (пути перечислены в `.gitignore`). Коммиты проверяет хук
  `scripts/git-hooks/pre-commit` (включён: `git config core.hooksPath
  scripts/git-hooks`); обход осознанно — `git commit --no-verify`.
- **Ядро хоста: `intel_iommu=off`** (дано +7% decode; побочка — нет
  PCI-passthrough в VM). P2P между картами при этом всё равно запрещён
  драйвером NVIDIA 610.57.04 (проверено: IOMMU off + ACS off, и
  `cudaDeviceCanAccessPeer` = False) — custom allreduce не включить, тема закрыта.

## Замеры 2026-09-21 (bench.py, 5×512, temp=0, после iommu-off)

| Профиль | Decode 1 поток | 2 парал. | Пул |
| --- | --- | --- | --- |
| dspark-ctx150 (mamba4) | 78.1 (e2e 75.0) | ⛔ | 151,683 |
| dspark-ctx144 (mamba8, текущий) | 78.2 (e2e 74.9) | ⛔ | 144,387 |
| dspark-conc2 (MAX_RUNNING=2) | 78.1 | 67.6 (сумма 2×512 перекрытием) | 129,795 |
| baseline | 43.2 (e2e 42.3) | 75.8 | 256,740 |
| dspark-maxctx | 59.1 (−24%) | ⛔ | 170,431 |

**ctx145 @ 0.883/155k (22.10, по логам, не чистый бенч):** медиана decode
69.0 ток/с (p10 49, p90 102, max 214, n=3488), accept len 2.34, пул 157,879.
Для прямой сравнимости с базой 78.2 нужен `bench.py` на этом профиле.

Прежние цифры README (DSPARK 57–72, Baseline 40.6/77.6) воспроизвелись
с поправкой ≈ +7% от iommu-off.

Инструменты: `bench.py` (1 поток), `bench-par2.py` (2 запроса с
перекрытием), `bench-longprompt.py` (TTFT/префилл на ~27k промпте).
accept length драфтера (логи, `accept len:`): у нас 2.34–2.40 против 2.89
в карточке — нижняя граница доменного диапазона, мониторить. Клиентский
рычаг латентности pi: `reasoning_effort=low` −25% / `enable_thinking:
false` −73% времени на рутинном шаге (chat_template_kwargs).
