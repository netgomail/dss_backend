"""
Скрипт для обработки архивов исторических данных и генерации Parquet файлов.

Обрабатывает архивы из data/archive/{ticker}/, фильтрует данные по праздничным/выходным дням,
ограничивает временной диапазон (10:00-18:40) и создает файлы для каждого таймфрейма.

Использование:
    python scripts/process_archives.py [ticker]
    
Примеры:
    python scripts/process_archives.py AFKS
    python scripts/process_archives.py  # обработает все тикеры из settings.INSTRUMENTS
"""

import logging
import sys
import zipfile
from pathlib import Path
from datetime import datetime, time
from typing import List, Optional
import tempfile

import polars as pl
from workalendar.europe import Russia

# === НАСТРОЙКА ПУТИ ПРОЕКТА ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import settings

# === НАСТРОЙКА ЛОГИРОВАНИЯ ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# === КОНСТАНТЫ ===
ARCHIVE_DIR = PROJECT_ROOT / "data" / "archive"
OUTPUT_DIR = PROJECT_ROOT / "data" / "tickers"
TRADING_START_TIME = time(10, 0)  # 10:00
TRADING_END_TIME = time(18, 40)   # 18:40

# Календарь для проверки праздничных дней
cal = Russia()


def extract_date_from_filename(filename: str) -> datetime:
    """
    Извлекает дату из имени файла формата *_YYYYMMDD.csv.
    
    Args:
        filename: Имя файла (например, "962e2a95-02a9-4171-abd7-aa198dbe643a_20250102.csv")
    
    Returns:
        Объект datetime с датой из имени файла
    
    Raises:
        ValueError: Если не удалось извлечь дату
    """
    parts = filename.rsplit('_', 1)
    if len(parts) != 2:
        raise ValueError(f"Неверный формат имени файла: {filename}")
    
    date_str = parts[1].replace('.csv', '')
    
    try:
        return datetime.strptime(date_str, '%Y%m%d')
    except ValueError as e:
        raise ValueError(f"Не удалось распарсить дату из {filename}: {e}")


def is_holiday_or_weekend(date_obj: datetime) -> bool:
    """
    Проверяет, является ли дата праздничным или выходным днем.
    
    Args:
        date_obj: Объект datetime для проверки
    
    Returns:
        True если дата является праздничным или выходным днем
    """
    date_only = date_obj.date()
    return not cal.is_working_day(date_only)


def filter_trading_hours(df: pl.DataFrame) -> pl.DataFrame:
    """
    Фильтрует данные по торговым часам (10:00-18:40) в московском времени.
    
    Args:
        df: Polars DataFrame с колонкой 'date' (datetime с timezone UTC)
    
    Returns:
        Отфильтрованный DataFrame
    """
    if df.is_empty():
        return df
    
    # Конвертируем UTC в московское время
    df_local = df.with_columns(
        pl.col("date").dt.convert_time_zone("Europe/Moscow").alias("date")
    )
    
    # Фильтруем по торговым часам (10:00-18:40)
    # Используем час и минуту для сравнения
    df_filtered = df_local.filter(
        (
            (pl.col("date").dt.hour() > TRADING_START_TIME.hour) |
            ((pl.col("date").dt.hour() == TRADING_START_TIME.hour) & 
             (pl.col("date").dt.minute() >= TRADING_START_TIME.minute))
        ) &
        (
            (pl.col("date").dt.hour() < TRADING_END_TIME.hour) |
            ((pl.col("date").dt.hour() == TRADING_END_TIME.hour) & 
             (pl.col("date").dt.minute() <= TRADING_END_TIME.minute))
        )
    )
    
    return df_filtered


def resample_to_timeframe(df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
    """
    Ресемплирует данные на указанный таймфрейм используя group_by_dynamic.
    
    Args:
        df: Polars DataFrame с OHLCV данными и колонкой 'date'
        timeframe: Таймфрейм (5M, 15M, 30M, 1H, 2H, 4H, 1D, 1W)
    
    Returns:
        Ресемплированный DataFrame
    """
    if df.is_empty():
        return df
    
    # Маппинг таймфреймов на Polars интервалы
    timeframe_map = {
        "5M": "5m",   # 5 минут
        "15M": "15m", # 15 минут
        "30M": "30m", # 30 минут
        "1H": "1h",   # 1 час
        "2H": "2h",   # 2 часа
        "4H": "4h",   # 4 часа
        "1D": "1d",   # 1 день
        "1W": "1w",   # 1 неделя
    }
    
    every = timeframe_map.get(timeframe)
    if not every:
        raise ValueError(f"Неизвестный таймфрейм: {timeframe}")
    
    # Ресемплинг OHLCV данных используя group_by_dynamic
    resampled = df.group_by_dynamic(
        "date",
        every=every,
        closed="left",
        label="left"
    ).agg([
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
    ]).sort("date")
    
    return resampled


def process_csv_file(csv_path: Path, ticker: str) -> Optional[pl.DataFrame]:
    """
    Обрабатывает один CSV файл: читает, фильтрует и возвращает DataFrame.
    
    Формат CSV: UUID;Date;Open;High;Low;Close;Volume;
    
    Args:
        csv_path: Путь к CSV файлу
        ticker: Тикер инструмента
    
    Returns:
        Polars DataFrame с обработанными данными или None при ошибке
    """
    try:
        # Читаем CSV файл с разделителем точка с запятой
        # Формат: UUID;Date;Open;High;Low;Close;Volume;
        df = pl.read_csv(
            csv_path,
            separator=';',
            has_header=False,
            new_columns=['uuid', 'date', 'open', 'high', 'low', 'close', 'volume', 'empty'],
            schema_overrides={
                'uuid': pl.Utf8,
                'date': pl.Utf8,  # Сначала читаем как строку
                'open': pl.Float64,
                'high': pl.Float64,
                'low': pl.Float64,
                'close': pl.Float64,
                'volume': pl.Int64,
                'empty': pl.Utf8,
            },
            ignore_errors=True
        )
        
        # Удаляем ненужные колонки
        df = df.select(['date', 'open', 'high', 'low', 'close', 'volume'])
        
        # Преобразуем date в datetime (UTC)
        df = df.with_columns(
            pl.col("date").str.to_datetime(time_unit="us", time_zone="UTC").alias("date")
        )
        
        # Удаляем строки с пустыми значениями
        df = df.drop_nulls()
        
        if df.is_empty():
            logger.debug(f"Файл {csv_path.name} пуст после чтения")
            return None
        
        # Фильтруем по праздничным и выходным дням
        # Используем map_elements для проверки каждой даты
        df = df.with_columns(
            pl.col("date").dt.date().map_elements(
                lambda d: not cal.is_working_day(d) if d else True,
                return_dtype=pl.Boolean
            ).alias("is_holiday")
        )
        
        # Удаляем строки с праздничными/выходными днями
        df = df.filter(~pl.col("is_holiday")).drop("is_holiday")
        
        if df.is_empty():
            logger.debug(f"Файл {csv_path.name} пуст после фильтрации праздничных дней")
            return None
        
        # Фильтруем по торговым часам (10:00-18:40)
        df = filter_trading_hours(df)
        
        if df.is_empty():
            logger.debug(f"Файл {csv_path.name} пуст после фильтрации торговых часов")
            return None
        
        return df
        
    except Exception as e:
        logger.error(f"Ошибка при обработке файла {csv_path.name}: {e}")
        return None


def process_archive(archive_path: Path, ticker: str) -> Optional[pl.DataFrame]:
    """
    Обрабатывает архив: извлекает CSV файлы, фильтрует и объединяет данные.
    
    Args:
        archive_path: Путь к ZIP архиву
        ticker: Тикер инструмента
    
    Returns:
        Объединенный Polars DataFrame со всеми данными из архива или None при ошибке
    """
    if not archive_path.exists():
        logger.warning(f"Архив не найден: {archive_path}")
        return None
    
    logger.info(f"Обработка архива: {archive_path.name}")
    
    all_dataframes = []
    
    try:
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            logger.info(f"Найдено файлов в архиве: {len(file_list)}")
            
            # Создаем временную директорию для извлечения файлов
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                for filename in file_list:
                    # Пропускаем директории
                    if filename.endswith('/'):
                        continue
                    
                    # Проверяем, является ли файл праздничным/выходным днем
                    try:
                        file_date = extract_date_from_filename(filename)
                        if is_holiday_or_weekend(file_date):
                            logger.debug(f"Пропущен файл {filename}: праздничный/выходной день")
                            continue
                    except ValueError:
                        # Если не удалось извлечь дату, пропускаем файл
                        logger.warning(f"Пропущен файл {filename}: неверный формат имени")
                        continue
                    
                    # Извлекаем файл во временную директорию
                    zip_ref.extract(filename, temp_path)
                    csv_path = temp_path / filename
                    
                    # Обрабатываем CSV файл
                    df = process_csv_file(csv_path, ticker)
                    if df is not None and not df.is_empty():
                        all_dataframes.append(df)
                
                if not all_dataframes:
                    logger.warning(f"Нет данных для обработки в архиве {archive_path.name}")
                    return None
                
                # Объединяем все DataFrame
                combined_df = pl.concat(all_dataframes)
                
                # Сортируем по дате
                combined_df = combined_df.sort("date")
                
                # Удаляем дубликаты по дате (оставляем первое вхождение)
                combined_df = combined_df.unique(subset=["date"], keep="first")
                
                logger.info(f"Обработано строк: {len(combined_df)}")
                return combined_df
                
    except zipfile.BadZipFile:
        logger.error(f"Некорректный формат ZIP архива: {archive_path}")
        return None
    except Exception as e:
        logger.error(f"Ошибка при обработке архива {archive_path.name}: {e}", exc_info=True)
        return None


def process_ticker(ticker: str) -> None:
    """
    Обрабатывает все архивы для указанного тикера и создает Parquet файлы.
    
    Args:
        ticker: Тикер инструмента
    """
    ticker_archive_dir = ARCHIVE_DIR / ticker
    ticker_output_dir = OUTPUT_DIR / ticker
    
    if not ticker_archive_dir.exists():
        logger.warning(f"Директория архивов не найдена: {ticker_archive_dir}")
        return
    
    # Создаем выходную директорию
    ticker_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Обработка тикера: {ticker}")
    
    # Находим все архивы для тикера
    archive_files = sorted(ticker_archive_dir.glob("*.zip"))
    
    if not archive_files:
        logger.warning(f"Архивы не найдены для тикера {ticker}")
        return
    
    logger.info(f"Найдено архивов: {len(archive_files)}")
    
    # Обрабатываем все архивы и объединяем данные
    all_dataframes = []
    
    for archive_path in archive_files:
        df = process_archive(archive_path, ticker)
        if df is not None and not df.is_empty():
            all_dataframes.append(df)
    
    if not all_dataframes:
        logger.warning(f"Нет данных для тикера {ticker}")
        return
    
    # Объединяем все данные
    combined_df = pl.concat(all_dataframes)
    combined_df = combined_df.sort("date")
    
    # Удаляем дубликаты
    combined_df = combined_df.unique(subset=["date"], keep="first")
    
    logger.info(f"Всего обработано строк для {ticker}: {len(combined_df)}")
    
    # Создаем файлы для каждого таймфрейма
    for timeframe in settings.TIMEFRAMES:
        try:
            # Ресемплируем данные на таймфрейм
            resampled_df = resample_to_timeframe(combined_df, timeframe)
            
            if resampled_df.is_empty():
                logger.warning(f"Нет данных для {ticker}/{timeframe} после ресемплинга")
                continue
            
            # Сохраняем в Parquet
            output_file = ticker_output_dir / f"{timeframe}.parquet"
            resampled_df.write_parquet(
                output_file,
                compression="snappy",
                use_pyarrow=True
            )
            
            logger.info(f"Создан файл: {ticker}/{timeframe}.parquet ({len(resampled_df)} строк)")
            
        except Exception as e:
            logger.error(f"Ошибка при создании файла {ticker}/{timeframe}.parquet: {e}", exc_info=True)
            continue


def main() -> None:
    """Основная функция для запуска обработки."""
    # Определяем тикеры для обработки
    if len(sys.argv) > 1:
        tickers = [sys.argv[1]]
    else:
        # Обрабатываем все активные тикеры из settings
        tickers = [inst["ticker"] for inst in settings.INSTRUMENTS]
    
    logger.info(f"Начало обработки архивов для {len(tickers)} тикеров")
    logger.info(f"Таймфреймы: {', '.join(settings.TIMEFRAMES)}")
    
    for ticker in tickers:
        try:
            process_ticker(ticker)
        except Exception as e:
            logger.error(f"Критическая ошибка при обработке {ticker}: {e}", exc_info=True)
            continue
    
    logger.info("Обработка завершена")


if __name__ == "__main__":
    main()
