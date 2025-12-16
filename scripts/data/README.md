# Скрипты для работы с историческими данными Tinkoff Invest

Набор скриптов для загрузки и обработки исторических рыночных данных с Tinkoff Invest API.

## Структура

```
scripts/data/
├── download_archives.py   # Скачивание ZIP архивов с минутными свечами за год
├── process_archives.py    # Обработка архивов и создание Parquet файлов
└── README.md             
```

## Быстрый старт

### 1. Установите зависимости

```bash
pip install requests polars pyarrow workalendar
```

### 2. Необходимо добаваить ваш токен в settings.py

```python
# В корне проекта: settings.py
INVEST_TOKEN = "t.ваш_токен_здесь"
```

### 3. Скачайте архивы

```bash
# Все инструменты из settings.INSTRUMENTS
python scripts/data/download_archives.py

# Или конкретные тикеры
python scripts/data/download_archives.py AFKS SBER GAZP
```

### 4. Обработайте архивы

```bash
# Все тикеры
python scripts/data/process_archives.py

# Или конкретные тикеры
python scripts/data/process_archives.py AFKS SBER GAZP
```

## Описание скриптов

### download_archives.py

**Назначение:** Скачивание исторических данных с Tinkoff Invest API в формате ZIP архивов.

**Что делает:**
- Скачивает минутные свечи за каждый год (от CURRENT_YEAR до MINIMUM_YEAR)
- Использует параллельную загрузку (2 потока по умолчанию)
- Автоматически обрабатывает rate limits и ошибки сети
- Пропускает уже существующие архивы (кроме текущего года)
- Сохраняет архивы в `data/archive/{ticker}/{year}.zip`

**Использование:**
```bash
# Все инструменты
python scripts/data/download_archives.py

# Конкретные тикеры
python scripts/data/download_archives.py AFKS
python scripts/data/download_archives.py AFKS SBER GAZP

# С подробным выводом
python scripts/data/download_archives.py --verbose
python scripts/data/download_archives.py AFKS -v
```

**Параметры (в коде):**
```python
DOWNLOAD_WORKERS = 2         # Количество потоков
RATE_LIMIT_DELAY = 5         # Задержка при rate limit (секунды)
MAX_RETRY_ATTEMPTS = 3       # Попыток повтора при ошибках
```

**Результат:**
```
data/
└── archive/
    ├── AFKS/
    │   ├── 2025.zip
    │   ├── 2024.zip
    │   ├── 2023.zip
    │   └── ...
    ├── SBER/
    └── ...
```

---

### process_archives.py

**Назначение:** Обработка скачанных архивов и создание Parquet файлов для разных таймфреймов.

**Что делает:**
- Извлекает CSV файлы из ZIP архивов
- Фильтрует праздничные и выходные дни (Russian calendar)
- Фильтрует нерабочие часы (07:00-23:59:59 по МСК)
- Создаёт файлы для 8 таймфреймов: 5M, 15M, 30M, 1H, 2H, 4H, 1D, 1W
- Сохраняет в формате Parquet с сжатием Snappy

**Использование:**
```bash
# Все тикеры из settings.INSTRUMENTS
python scripts/data/process_archives.py

# Конкретные тикеры
python scripts/data/process_archives.py AFKS
python scripts/data/process_archives.py AFKS SBER GAZP

# С подробным выводом
python scripts/data/process_archives.py --verbose
python scripts/data/process_archives.py AFKS -v
```

**Параметры (в коде):**
```python
TRADING_START_TIME = time(7, 0)      # Начало торговой сессии
TRADING_END_TIME = time(23, 59, 59)  # Конец торговой сессии
```

**Результат:**
```
data/
└── tickers/
    ├── AFKS/
    │   ├── 5M.parquet
    │   ├── 15M.parquet
    │   ├── 30M.parquet
    │   ├── 1H.parquet
    │   ├── 2H.parquet
    │   ├── 4H.parquet
    │   ├── 1D.parquet
    │   └── 1W.parquet
    ├── SBER/
    └── ...
```

## Типичные сценарии

### Первоначальная загрузка всех данных

```bash
# 1. Скачиваем архивы (может занять несколько часов)
python scripts/data/download_archives.py

# 2. Обрабатываем архивы
python scripts/data/process_archives.py
```

### Добавление нового инструмента

```bash
# 1. Добавьте инструмент в settings.py
# 2. Скачайте только его данные
python scripts/data/download_archives.py НОВЫЙ_ТИКЕР

# 3. Обработайте
python scripts/data/process_archives.py НОВЫЙ_ТИКЕР
```

### Ежедневное обновление

```bash
# Скачивание обновляет только текущий год
python scripts/data/download_archives.py

# Переобработка всех данных
python scripts/data/process_archives.py
```

### Обновление конкретного тикера

```bash
python scripts/data/download_archives.py AFKS
python scripts/data/process_archives.py AFKS
```

### Восстановление после сбоя

```bash
# Если скачивание прервалось - продолжится с того места
python scripts/data/download_archives.py

# Если обработка прервалась - можно запустить заново
python scripts/data/process_archives.py
```

## Формат данных

### Входные данные (CSV в ZIP)

Формат строки:
```
UUID;Date;Open;High;Low;Close;Volume;
962e2a95-02a9-4171-abd7-aa198dbe643a;2025-01-02T07:00:00Z;148.5;149.2;148.0;148.8;1000;
```

### Выходные данные (Parquet)

Структура:
```python
date: datetime[μs, Europe/Moscow]  # Дата и время в московском часовом поясе
open: float64                      # Цена открытия
high: float64                      # Максимальная цена
low: float64                       # Минимальная цена
close: float64                     # Цена закрытия
volume: int64                      # Объём торгов
```

Пример чтения:
```python
import polars as pl

df = pl.read_parquet("data/tickers/AFKS/1D.parquet")
print(df.head())
```

## Конфигурация (settings.py)

### Обязательные параметры

```python
# Токен Tinkoff Invest API
INVEST_TOKEN = "t.xxx..."

# Диапазон годов
MINIMUM_YEAR = 2009
CURRENT_YEAR = datetime.now().year

# Таймфреймы
TIMEFRAMES = ["5M", "15M", "30M", "1H", "2H", "4H", "1D", "1W"]

# Инструменты
INSTRUMENTS = [
    {
        "ticker": "AFKS",
        "name": "АФК Система",
        "figi": "BBG004S68614",
        "class_code": "TQBR",
        "instrument_type": "share"
    },
    # ... другие инструменты
]
```

### Получение токена

1. Откройте приложение **Tinkoff Investments**
2. Настройки → **Токены**
3. Создайте токен с правами **"Только чтение"**
4. Скопируйте и добавьте в `settings.py`

## Автоматизация

### Cron job для ежедневного обновления

```bash
# Добавьте в crontab (crontab -e):

# Скачивание новых данных каждый день в 19:00
0 19 * * * cd /path/to/project && python scripts/data/download_archives.py >> logs/download.log 2>&1

# Обработка данных каждый день в 20:00
0 20 * * * cd /path/to/project && python scripts/data/process_archives.py >> logs/process.log 2>&1
```

### Bash скрипт для полного обновления

```bash
#!/bin/bash
# update_data.sh

cd /path/to/project

echo "Начало обновления данных: $(date)"

# Скачивание
python scripts/data/download_archives.py
if [ $? -ne 0 ]; then
    echo "Ошибка при скачивании"
    exit 1
fi

# Обработка
python scripts/data/process_archives.py
if [ $? -ne 0 ]; then
    echo "Ошибка при обработке"
    exit 1
fi

echo "Обновление завершено: $(date)"
```

## Логика работы

### Скачивание (download_archives.py)

1. **Инициализация**: Создаёт очередь задач для каждого инструмента
2. **Параллельная загрузка**: Запускает 2 воркера для одновременного скачивания
3. **Последовательность**: Для каждого инструмента скачивает годы от текущего к минимальному
4. **Проверка существования**: Пропускает уже скачанные файлы (кроме текущего года)
5. **Обработка ошибок**:
   - **Rate limit (429)**: Ждёт 5 секунд и повторяет
   - **Server error (500)**: Пропускает инструмент
   - **Not found (404)**: Останавливает загрузку для инструмента
   - **Network error**: До 3 попыток повтора
6. **Сохранение**: `data/archive/{ticker}/{year}.zip`

### Обработка (process_archives.py)

1. **Чтение архивов**: Последовательная обработка всех ZIP файлов тикера
2. **Фильтрация файлов**: Пропуск файлов праздничных/выходных дней
3. **Парсинг CSV**: Чтение минутных свечей с разделителем `;`
4. **Фильтрация данных**:
   - Удаление праздничных дней (Russian calendar)
   - Фильтрация по часам: 07:00-23:59:59 МСК (только для внутридневных)
5. **Объединение**: Объединение всех данных и удаление дубликатов
6. **Ресемплинг**: Создание 8 таймфреймов через `group_by_dynamic`
7. **Сохранение**: Parquet с компрессией Snappy

### Фильтрация торговых часов

**Внутридневные таймфреймы** (5M, 15M, 30M, 1H, 2H, 4H):
- Применяется фильтр 07:00-23:59:59 МСК
- Убирает ночные часы и внебиржевую торговлю

**Дневные/недельные** (1D, 1W):
- Фильтр по часам НЕ применяется
- Агрегируются все данные за день/неделю

## Примеры использования данных

### Загрузка и анализ

```python
import polars as pl

# Загрузка дневных данных
df = pl.read_parquet("data/tickers/AFKS/1D.parquet")

# Базовая информация
print(f"Строк: {len(df)}")
print(f"Период: {df['date'].min()} - {df['date'].max()}")
print(df.describe())

# Расчёт доходности
df = df.with_columns([
    ((pl.col("close") - pl.col("close").shift(1)) / pl.col("close").shift(1) * 100)
    .alias("return_pct")
])

print(f"Средняя дневная доходность: {df['return_pct'].mean():.2f}%")
```

### Объединение нескольких тикеров

```python
import polars as pl
from pathlib import Path

tickers = ["AFKS", "SBER", "GAZP"]
timeframe = "1D"

dfs = []
for ticker in tickers:
    file_path = Path(f"data/tickers/{ticker}/{timeframe}.parquet")
    if file_path.exists():
        df = pl.read_parquet(file_path)
        df = df.with_columns(pl.lit(ticker).alias("ticker"))
        dfs.append(df)

# Объединение
combined = pl.concat(dfs)

# Анализ по тикерам
stats = combined.group_by("ticker").agg([
    pl.col("close").mean().alias("avg_price"),
    pl.col("volume").mean().alias("avg_volume"),
])
print(stats)
```

## ⚠️ Устранение проблем

### "INVEST_TOKEN не установлен"

```python
# Решение: добавьте токен в settings.py (в корне проекта)
INVEST_TOKEN = "t.ваш_токен_здесь"
```

### "Rate limit" при скачивании

```python
# Решение 1: уменьшите количество воркеров в download_archives.py
DOWNLOAD_WORKERS = 1

# Решение 2: увеличьте задержку
RATE_LIMIT_DELAY = 10
```

### Файлы не создаются в нужной директории

Скрипты автоматически определяют корень проекта (где находится `settings.py`) и создают директории относительно него:
- `data/archive/` - архивы
- `data/tickers/` - обработанные данные

## Требования

### Системные
- Python 3.12
- 2+ GB RAM
- 10+ GB свободного места (для нескольких инструментов)

### Python пакеты
```bash
requests       # HTTP клиент для API
polars         # Обработка данных (быстрее pandas)
pyarrow        # Backend для Parquet
workalendar    # Российский календарь праздников
```

## Полезные ссылки

- **Tinkoff Invest API**: https://tinkoff.github.io/investAPI/
