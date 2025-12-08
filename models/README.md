# Модель сигналов (buy/sell)

В этом каталоге два основных скрипта:

- `train_signal_model.py` — обучение LSTM-модели по историческим OHLCV и индикаторам, генерация сигналов (бинарно: sell=0, buy=1).
- `eval_signal_model.py` — повторная оценка сохранённой модели на тестовом срезе по тем же правилам, что и в обучении.

## Данные

- Источник: parquet-файлы из `data/tickers/<TICKER>/<TF>.parquet`, которые собирает `scripts/data_collector.py`.
- Таймфрейм задаётся флагом `--timeframe` (по умолчанию M30).
- Признаки берутся из колонок индикаторов (SMA/EMA, MACD, RSI, BB, ATR, паттерны, режимы рынка) + OHLCV и `ticker_id`.

## Разметка (бинарно)

- `future_ret = (future_close - close) / close` через `--horizon` баров.
- Если `future_ret > threshold` → `buy (1)`.
- Если `future_ret < -threshold` → `sell (0)`.
- Нейтральные примеры (`|future_ret| <= threshold`) исключаются, чтобы убрать hold.

## Обучение `train_signal_model.py`

Артефакты по умолчанию:
- модель: `models/signal_model.keras`
- скейлер: `models/signal_model_scaler.pkl`
- meta: `models/signal_model_meta.json`
- график обучения: `models/signal_model_learning_curve.png`

Пример запуска (бинарно):
```bash
cd /Users/denis/dev/app/dss_backend
KERAS_BACKEND=tensorflow python models/train_signal_model.py \
  --timeframe M30 --lookback 64 --horizon 4 --threshold 0.003 \
  --val-size 0.15 --test-size 0.15 --epochs 20 --batch-size 256
```

Ключевые флаги:
- `--timeframe` — таймфрейм parquet (например M30, H1, H4, D1).
- `--lookback` — длина окна в барах.
- `--horizon` — через сколько баров считаем доходность.
- `--threshold` — порог для buy/sell (0.003 = 0.3%).
- `--val-size`, `--test-size` — доли валидации и теста (по времени, хвост).
- `--max-rows` — ограничить хвост данных для быстрых прогонов.

Процесс:
1. Загрузка и объединение данных всех тикеров по заданному таймфрейму; замена inf на NaN, удаление NaN.
2. Разметка бинарных сигналов: считаем future_ret на горизонте `--horizon`, применяем порог `--threshold`; оставляем только buy/sell, нейтральные строки выбрасываются.
3. Формирование временных окон длиной `--lookback`; метка относится к последней свече окна.
4. Временное разбиение последовательно по оси времени: train → val → test (проценты `--val-size`, `--test-size`); тест — самый свежий хвост.
5. Масштабирование признаков StandardScaler, обученного на train; применяем к val/test.
6. Обучение модели: двухслойная LSTM + Dense, softmax на 2 класса; class_weight для компенсации дисбаланса.
7. Регуляризация обучения: EarlyStopping (патience=5) + ReduceLROnPlateau (patience=2, factor=0.5).
8. Оценка на val/test, сохранение отчётов в meta, построение кривых обучения (loss/acc).
9. Сохранение артефактов: модель (.keras), scaler (.pkl), meta (.json), график (.png).

Meta хранит: список признаков, mapping классов, tf, lookback, horizon, threshold, доли val/test и отчёты по val/test.

## Оценка `eval_signal_model.py`

Использует сохранённые `meta`, `model`, `scaler`:
```bash
KERAS_BACKEND=tensorflow python models/eval_signal_model.py
```

Артефакты:
- `models/eval_signal_metrics.json` — classification_report по тесту.
- `models/eval_signal_confusion.png` — confusion matrix.

Опциональные флаги:
- `--timeframe`, `--tickers` — переопределить tf/список тикеров.
- `--max-rows` — укоротить хвост.
- Пути к артефактам: `--model-path`, `--scaler-path`, `--meta-path`, `--metrics-path`, `--cm-path`.

## Требования

- `tensorflow-macos` + `tensorflow-metal` для M1/M2/M3 (GPU через Metal).
- Зависимости из `requirements.txt` (pandas, numpy, sklearn, matplotlib и т.д.).
- Переменная `KERAS_BACKEND=tensorflow` (в скриптах стоит по умолчанию).

## Замечания по качеству

- Класс hold убран. Если нужно больше баланса, регулировать `--threshold`, `--horizon`, `--lookback`, а также class_weight/oversampling при необходимости.
- Проверяйте, что после фильтрации по threshold данных хватает (скрипт предупредит и остановится).

