"""
Скрипт для проверки файлов в архиве на соответствие праздничным дням в РФ.

Извлекает архив, анализирует даты из имен файлов и выводит список файлов,
которые попадают на праздничные дни в Российской Федерации.

Использование:
    python scripts/check_holidays.py [ticker] [year]
    
Примеры:
    python scripts/check_holidays.py GAZP 2025
    python scripts/check_holidays.py SBER 2024
"""

import logging
import sys
import zipfile
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

from workalendar.europe import Russia

# === НАСТРОЙКА ПУТИ ПРОЕКТА ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# === НАСТРОЙКА ЛОГИРОВАНИЯ ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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
    # Ищем паттерн _YYYYMMDD.csv в конце имени файла
    parts = filename.rsplit('_', 1)
    if len(parts) != 2:
        raise ValueError(f"Неверный формат имени файла: {filename}")
    
    date_str = parts[1].replace('.csv', '')
    
    try:
        return datetime.strptime(date_str, '%Y%m%d')
    except ValueError as e:
        raise ValueError(f"Не удалось распарсить дату из {filename}: {e}")


def get_holiday_files(archive_path: Path) -> List[Tuple[str, datetime, str]]:
    """
    Извлекает список файлов из архива, которые попадают на праздничные дни.
    
    Args:
        archive_path: Путь к ZIP архиву
    
    Returns:
        Список кортежей (имя_файла, дата, название_праздника)
    """
    if not archive_path.exists():
        raise FileNotFoundError(f"Архив не найден: {archive_path}")
    
    # Создаем календарь для России
    cal = Russia()
    
    holiday_files = []
    
    logger.info(f"Открытие архива: {archive_path}")
    
    try:
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            logger.info(f"Найдено файлов в архиве: {len(file_list)}")
            
            for filename in file_list:
                # Пропускаем директории
                if filename.endswith('/'):
                    continue
                
                try:
                    # Извлекаем дату из имени файла
                    file_date = extract_date_from_filename(filename)
                    
                    # Проверяем, является ли дата праздничным днем
                    is_holiday, holiday_name = cal.is_holiday(file_date)
                    
                    if is_holiday:
                        holiday_files.append((filename, file_date, holiday_name))
                        logger.debug(f"Найден праздничный день: {filename} -> {file_date.date()} ({holiday_name})")
                
                except ValueError as e:
                    logger.warning(f"Пропущен файл {filename}: {e}")
                    continue
    
    except zipfile.BadZipFile:
        raise ValueError(f"Некорректный формат ZIP архива: {archive_path}")
    
    return holiday_files


def main() -> None:
    """Основная функция для запуска проверки."""
    # Параметры командной строки
    ticker = sys.argv[1] if len(sys.argv) > 1 else "GAZP"
    year = sys.argv[2] if len(sys.argv) > 2 else "2025"
    
    # Формируем путь к архиву
    archive_path = PROJECT_ROOT / "data" / "archive" / ticker / f"{year}.zip"
    
    logger.info(f"Проверка архива: {archive_path}")
    logger.info(f"Тикер: {ticker}, Год: {year}")
    
    try:
        # Получаем список файлов, попадающих на праздничные дни
        holiday_files = get_holiday_files(archive_path)
        
        if not holiday_files:
            logger.info("Файлы, попадающие на праздничные дни, не найдены.")
            return
        
        # Выводим результаты
        print(f"\n{'='*80}")
        print(f"Файлы, попадающие на праздничные дни в РФ ({len(holiday_files)} шт.):")
        print(f"{'='*80}\n")
        
        # Сортируем по дате
        holiday_files.sort(key=lambda x: x[1])
        
        for filename, file_date, holiday_name in holiday_files:
            print(f"  {filename}")
            print(f"    Дата: {file_date.strftime('%Y-%m-%d (%A)')}")
            print(f"    Праздник: {holiday_name}")
            print()
        
        print(f"{'='*80}")
        print(f"Всего найдено: {len(holiday_files)} файлов")
        
    except FileNotFoundError as e:
        logger.error(f"Ошибка: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

