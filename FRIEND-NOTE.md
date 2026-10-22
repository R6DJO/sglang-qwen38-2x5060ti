# Другу, который это распаковал

Что это: рабочий сетап Qwen3.8-27B NVFP4 (gittensor-model-hub/…) на двух
RTX 5060 Ti 16GB через SGLang (TP=2), с профилями под разные задачи.
Документация: README.md (полная), AGENTS.md (шпаргалка + свежие замеры).

## Быстрый старт

1. Нужно: Linux + NVIDIA driver с поддержкой Blackwell (sm_120), Docker +
   compose plugin, ~55 ГБ диска (образ ~14ГБ + модель ~21ГБ из HF + распаковка).
2. В `.env` и в каждом `profiles/*.env` поправь HF_CACHE_DIR на свой путь
   к кэшу HuggingFace (по умолчанию `./hf-cache`).
3. Модель скачается сама при первом старте (публичный репозиторий на HF).
4. Запуск (DSPARK — быстрый single-stream, 78 ток/с, контекст 140k):
   docker compose up -d
   docker compose logs -f   # ждать "The server is fired up and ready to roll!"
   API: http://localhost:30000/v1 , модель qwen3.8-27b
5. Другие профили — см. README «Профили»: baseline (2 клиента + vision + 192k),
   longctx (262k), turbo (DSPARK + vision).

## Важно (грабли, проверены на практике)

- На DSPARK-профилях шлать запросы по одному. Перекрытие больше не роняет
  шедулер (проверено при 8 слотах), но параллель всё равно невыгодна:
  суммарная скорость −13%, пул −15k. Для типичного одного клиента — строго
  по одному.
- После рестарта проверять max_total_num_tokens в логах: если пул ниже
  CONTEXT_LENGTH профиля — понизить CONTEXT_LENGTH (пул плавает ±2k).
- mem-fraction выше 0.87 = обвал скорости шага на ~25% (аллокатор), не поднимай.
- Замеры в документах сделаны на Xeon E5-2690 v4 + PCIe 3.0, intel_iommu=off.
