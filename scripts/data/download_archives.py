"""
Скрипт для загрузки исторических рыночных данных от Tinkoff Invest API.

Скачивает исторические данные в формате ZIP архивов для всех инструментов,
указанных в settings.INSTRUMENTS, за период от CURRENT_YEAR до MINIMUM_YEAR.

Использование:
    python scripts/data/download_archives.py              # Все инструменты
    python scripts/data/download_archives.py AFKS         # Один тикер
    python scripts/data/download_archives.py AFKS SBER    # Несколько тикеров
    python scripts/data/download_archives.py --verbose    # Подробный вывод
"""

import logging
import sys
import time
from pathlib import Path
from typing import List, Optional
from queue import Queue, Empty
from threading import Thread, Event
from dataclasses import dataclass
from enum import Enum

import requests

# === НАСТРОЙКА ПУТИ ПРОЕКТА ===
# Поднимаемся на два уровня вверх от scripts/data/ к корню проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import settings

# === НАСТРОЙКА ЛОГИРОВАНИЯ ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(threadName)-12s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# === КОНСТАНТЫ ===
HISTORY_DATA_URL = "https://invest-public-api.tinkoff.ru/history-data"
ARCHIVE_DIR = PROJECT_ROOT / "data" / "archive"

# Параметры работы
RATE_LIMIT_DELAY = 5  # секунд при превышении лимита
MAX_RETRY_ATTEMPTS = 3  # максимум попыток при ошибках
DOWNLOAD_WORKERS = 1  # количество потоков для скачивания


class TaskStatus(Enum):
    """Статусы задач."""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    FAILED = "failed"


@dataclass
class InstrumentTask:
    """Задача для скачивания инструмента."""
    figi: str
    ticker: str
    name: str
    status: TaskStatus = TaskStatus.PENDING
    error: Optional[str] = None
    downloaded_years: List[int] = None
    
    def __post_init__(self):
        if self.downloaded_years is None:
            self.downloaded_years = []


class ServerError(Exception):
    """Исключение для ошибок сервера, требующих пропуска инструмента."""
    pass


# ==================== ФУНКЦИИ СКАЧИВАНИЯ ====================

def download_year(
    figi: str,
    ticker: str,
    year: int,
    token: str,
    output_dir: Path,
    retry_count: int = 0
) -> bool:
    """
    Скачивает исторические данные за указанный год для инструмента.
    
    Args:
        figi: FIGI инструмента
        ticker: Тикер инструмента
        year: Год для загрузки
        token: Токен авторизации API
        output_dir: Базовая директория для сохранения файлов
        retry_count: Счетчик попыток повтора
    
    Returns:
        True если загрузка успешна, False если данные не найдены
        
    Raises:
        ValueError: При невалидном токене (401)
        ServerError: При ошибках сервера
    """
    if year < settings.MINIMUM_YEAR:
        return False
    
    ticker_dir = output_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    
    file_name = ticker_dir / f"{year}.zip"
    
    # Пропускаем если файл существует (кроме текущего года)
    if file_name.exists() and year != settings.CURRENT_YEAR:
        logger.info(f"[{ticker}] Архив {year}.zip уже существует")
        return True
    
    logger.info(f"[{ticker}] Скачивание данных за {year} год")
    
    try:
        response = requests.get(
            HISTORY_DATA_URL,
            params={"figi": figi, "year": year},
            headers={"Authorization": f"Bearer {token}"},
            timeout=30
        )
        
        status_code = response.status_code
        
        # Rate limiting
        if status_code == 429:
            if retry_count < MAX_RETRY_ATTEMPTS:
                logger.warning(f"[{ticker}] Rate limit, повтор через {RATE_LIMIT_DELAY}с")
                time.sleep(RATE_LIMIT_DELAY)
                return download_year(figi, ticker, year, token, output_dir, retry_count + 1)
            else:
                raise ServerError(f"Превышено количество попыток для {ticker}/{year}")
        
        # Невалидный токен
        if status_code == 401:
            logger.error(f"[{ticker}] Невалидный токен API")
            raise ValueError("Невалидный токен API")
        
        # Ошибка сервера
        if status_code == 500:
            logger.warning(f"[{ticker}] Ошибка сервера (500) для {year}")
            raise ServerError(f"Ошибка сервера для {ticker}")
        
        # Данные не найдены
        if status_code == 404:
            logger.debug(f"[{ticker}] Данные не найдены за {year} год")
            if file_name.exists():
                file_name.unlink()
            return False
        
        # Другие ошибки
        if status_code != 200:
            logger.warning(f"[{ticker}] Ошибка {status_code} для {year}")
            raise ServerError(f"Ошибка {status_code} для {ticker}")
        
        # Сохраняем файл
        with open(file_name, "wb") as f:
            f.write(response.content)
        
        logger.info(f"[{ticker}] ✓ Загружен архив {year}.zip ({len(response.content) / 1024 / 1024:.1f} MB)")
        return True
        
    except requests.RequestException as e:
        if retry_count < MAX_RETRY_ATTEMPTS:
            logger.warning(f"[{ticker}] Сетевая ошибка, повтор {retry_count + 1}/{MAX_RETRY_ATTEMPTS}")
            time.sleep(RATE_LIMIT_DELAY)
            return download_year(figi, ticker, year, token, output_dir, retry_count + 1)
        else:
            logger.error(f"[{ticker}] Сетевая ошибка: {e}")
            raise ServerError(f"Сетевая ошибка для {ticker}: {e}")


def download_instrument_worker(
    task_queue: Queue,
    token: str,
    stop_event: Event
) -> None:
    """
    Воркер для скачивания данных инструментов.
    
    Args:
        task_queue: Очередь с задачами для скачивания
        token: Токен API
        stop_event: Событие для остановки воркера
    """
    while not stop_event.is_set():
        try:
            task: InstrumentTask = task_queue.get(timeout=1)
        except Empty:
            continue
        
        try:
            task.status = TaskStatus.DOWNLOADING
            logger.info(f"[{task.ticker}] Начало скачивания ({task.name})")
            
            year = settings.CURRENT_YEAR
            
            while year >= settings.MINIMUM_YEAR:
                try:
                    success = download_year(
                        task.figi,
                        task.ticker,
                        year,
                        token,
                        ARCHIVE_DIR
                    )
                    
                    if success:
                        task.downloaded_years.append(year)
                    else:
                        # Данных нет, останавливаем для этого инструмента
                        break
                        
                except ServerError as e:
                    logger.warning(f"[{task.ticker}] Пропуск: {e}")
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    break
                except ValueError as e:
                    # Критическая ошибка токена
                    logger.error(f"[{task.ticker}] Критическая ошибка: {e}")
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    stop_event.set()  # Останавливаем все воркеры
                    break
                
                year -= 1
            
            if task.status != TaskStatus.FAILED:
                task.status = TaskStatus.DOWNLOADED
                logger.info(f"[{task.ticker}] ✓ Скачивание завершено ({len(task.downloaded_years)} архивов)")
                
        except Exception as e:
            logger.error(f"[{task.ticker}] Неожиданная ошибка при скачивании: {e}", exc_info=True)
            task.status = TaskStatus.FAILED
            task.error = str(e)
        finally:
            task_queue.task_done()


# ==================== ГЛАВНАЯ ФУНКЦИЯ ====================

def main() -> None:
    """Основная функция для запуска загрузки данных."""
    start_time = time.time()
    
    # Парсинг аргументов
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Определяем тикеры
    tickers_arg = [arg for arg in sys.argv[1:] if not arg.startswith("--") and not arg.startswith("-")]
    if tickers_arg:
        instruments = [
            inst for inst in settings.INSTRUMENTS
            if inst["ticker"] in tickers_arg
        ]
        if not instruments:
            logger.error(f"Тикеры {tickers_arg} не найдены в settings.INSTRUMENTS")
            sys.exit(1)
    else:
        instruments = settings.INSTRUMENTS
    
    # Получаем токен
    token = settings.INVEST_TOKEN
    if not token:
        logger.error("INVEST_TOKEN не установлен в settings.py")
        sys.exit(1)
    
    # Создаем директорию для архивов
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info(f"СКАЧИВАНИЕ АРХИВОВ ДЛЯ {len(instruments)} ИНСТРУМЕНТОВ")
    logger.info(f"Диапазон годов: {settings.MINIMUM_YEAR} - {settings.CURRENT_YEAR}")
    logger.info(f"Директория: {ARCHIVE_DIR}")
    logger.info(f"Воркеров: {DOWNLOAD_WORKERS}")
    logger.info("=" * 70)
    
    # Создаем очереди и события
    download_queue = Queue()
    stop_event = Event()
    task_dict = {}
    
    # Добавляем задачи в очередь
    for inst in instruments:
        task = InstrumentTask(
            figi=inst["figi"],
            ticker=inst["ticker"],
            name=inst["name"]
        )
        task_dict[task.ticker] = task
        download_queue.put(task)
    
    # Запускаем воркеры
    download_threads = []
    for i in range(DOWNLOAD_WORKERS):
        t = Thread(
            target=download_instrument_worker,
            args=(download_queue, token, stop_event),
            name=f"Downloader-{i+1}"
        )
        t.daemon = True
        t.start()
        download_threads.append(t)
    
    # Ждем завершения
    try:
        download_queue.join()
    except KeyboardInterrupt:
        logger.info("Прервано пользователем")
        stop_event.set()
        sys.exit(0)
    
    # Статистика
    elapsed = time.time() - start_time
    minutes, seconds = divmod(int(elapsed), 60)
    
    successful = sum(1 for t in task_dict.values() if t.status == TaskStatus.DOWNLOADED)
    failed = sum(1 for t in task_dict.values() if t.status == TaskStatus.FAILED)
    total_archives = sum(len(t.downloaded_years) for t in task_dict.values())
    
    logger.info("=" * 70)
    logger.info("ИТОГОВАЯ СТАТИСТИКА")
    logger.info("=" * 70)
    logger.info(f"Инструментов успешно: {successful}/{len(instruments)}")
    logger.info(f"Инструментов с ошибками: {failed}")
    logger.info(f"Всего скачано архивов: {total_archives}")
    logger.info(f"Время выполнения: {minutes}м {seconds}с")
    
    # Детализация по ошибкам
    if failed > 0:
        logger.warning("Инструменты с ошибками:")
        for ticker, task in task_dict.items():
            if task.status == TaskStatus.FAILED:
                logger.warning(f"  - {ticker}: {task.error}")
    
    logger.info("=" * 70)
    logger.info(f"Архивы сохранены в: {ARCHIVE_DIR}")
    logger.info("Следующий шаг: python scripts/data/process_archives.py")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()