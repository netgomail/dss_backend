"""
Скрипт для загрузки исторических рыночных данных от Tinkoff Invest API.

Скачивает исторические данные в формате ZIP архивов для всех инструментов,
указанных в settings.INSTRUMENTS, за период от CURRENT_YEAR до MINIMUM_YEAR.

Использование:
    python scripts/download_md.py
"""

import logging
import sys
import time
from pathlib import Path

import requests


class ServerError(Exception):
    """Исключение для ошибок сервера, требующих пропуска инструмента."""
    pass

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
HISTORY_DATA_URL = "https://invest-public-api.tinkoff.ru/history-data"
RATE_LIMIT_DELAY = 5  # секунд ожидания при превышении лимита запросов


def download_year(figi: str, ticker: str, year: int, token: str, output_dir: Path) -> bool:
    """
    Скачивает исторические данные за указанный год для инструмента.
    
    Args:
        figi: FIGI инструмента
        ticker: Тикер инструмента (для создания подпапки)
        year: Год для загрузки
        token: Токен авторизации API
        output_dir: Базовая директория для сохранения файлов
    
    Returns:
        True если загрузка успешна, False если данные не найдены или произошла ошибка,
        выбрасывает ValueError при невалидном токене (401)
    """
    # Проверяем минимальный год
    if year < settings.MINIMUM_YEAR:
        return False
    
    # Создаем подпапку для тикера
    ticker_dir = output_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    
    file_name = ticker_dir / f"{year}.zip"
    
    # Проверяем, существует ли файл уже
    # Для текущего года всегда загружаем (обновляем)
    if file_name.exists() and year != settings.CURRENT_YEAR:
        logger.info(f"Архив {ticker}/{year}.zip уже существует, пропускаем")
        return True
    
    logger.info(f"Скачивание {ticker} за {year} год")
    
    try:
        response = requests.get(
            HISTORY_DATA_URL,
            params={"figi": figi, "year": year},
            headers={"Authorization": f"Bearer {token}"},
            timeout=30
        )
        
        status_code = response.status_code
        
        # Обработка rate limiting (429)
        if status_code == 429:
            logger.warning("Превышен лимит запросов. Ожидание 5 секунд...")
            time.sleep(RATE_LIMIT_DELAY)
            return download_year(figi, ticker, year, token, output_dir)
        
        # Обработка ошибок авторизации
        if status_code == 401:
            logger.error(f"Невалидный токен для {ticker} (FIGI: {figi}), year={year}")
            raise ValueError("Невалидный токен API")
        
        # Обработка ошибок сервера (500) - пропускаем весь инструмент
        if status_code == 500:
            logger.warning(f"Ошибка сервера для {ticker} (FIGI: {figi}), year={year}. Пропускаем весь инструмент.")
            raise ServerError(f"Ошибка сервера для {ticker}")
        
        # Данные не найдены (404)
        if status_code == 404:
            logger.info(f"Данные не найдены для figi={figi}, year={year}")
            # Удаляем пустой файл, если он был создан
            if file_name.exists():
                file_name.unlink()
            return False
        
        # Обработка других ошибок (кроме 404) - пропускаем весь инструмент
        if status_code != 200:
            logger.warning(f"Ошибка с кодом {status_code} для {ticker} (FIGI: {figi}), year={year}. Пропускаем весь инструмент.")
            raise ServerError(f"Ошибка {status_code} для {ticker}")
        
        # Сохраняем файл
        with open(file_name, "wb") as f:
            f.write(response.content)
        
        logger.info(f"Успешно загружено: {file_name.name}")
        return True
        
    except requests.RequestException as e:
        logger.error(f"Ошибка при запросе для {ticker} (FIGI: {figi}), year={year}: {e}")
        # Пропускаем весь инструмент при сетевых ошибках
        raise ServerError(f"Сетевая ошибка для {ticker}: {e}")


def download_instrument_history(figi: str, ticker: str, token: str, output_dir: Path) -> None:
    """
    Скачивает всю историю инструмента от текущего года до минимального.
    
    Args:
        figi: FIGI инструмента
        ticker: Тикер инструмента (для создания подпапки)
        token: Токен авторизации API
        output_dir: Базовая директория для сохранения файлов
    """
    year = settings.CURRENT_YEAR
    
    while year >= settings.MINIMUM_YEAR:
        download_year(figi, ticker, year, token, output_dir)
        year -= 1


def main() -> None:
    """Основная функция для запуска загрузки данных."""
    # Получаем токен
    token = settings.INVEST_TOKEN
    if not token:
        logger.error("INVEST_TOKEN не установлен в settings.py")
        sys.exit(1)
    
    # Создаем директорию для сохранения архивов
    output_dir = PROJECT_ROOT / "data" / "archive"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Начало загрузки данных для {len(settings.INSTRUMENTS)} инструментов")
    logger.info(f"Диапазон годов: {settings.MINIMUM_YEAR} - {settings.CURRENT_YEAR}")
    logger.info(f"Директория сохранения: {output_dir}")
    
    # Загружаем данные для каждого инструмента
    for instrument in settings.INSTRUMENTS:
        figi = instrument.get("figi")
        ticker = instrument.get("ticker", "UNKNOWN")
        
        if not figi:
            logger.warning(f"Пропущен инструмент {ticker}: отсутствует FIGI")
            continue
        
        logger.info(f"Обработка инструмента: {ticker} (FIGI: {figi})")
        
        try:
            download_instrument_history(figi, ticker, token, output_dir)
        except KeyboardInterrupt:
            logger.info("Загрузка прервана пользователем")
            sys.exit(0)
        except ValueError as e:
            # Критическая ошибка (например, невалидный токен) - останавливаем весь процесс
            logger.error(f"Критическая ошибка: {e}")
            sys.exit(1)
        except ServerError as e:
            # Ошибка сервера - пропускаем этот инструмент, переходим к следующему
            logger.warning(f"Пропущен инструмент {ticker} ({figi}): {e}")
            continue
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных для {ticker} ({figi}): {e}")
            # Продолжаем обработку других инструментов
            continue
    
    logger.info("Загрузка данных завершена")


if __name__ == "__main__":
    main()

