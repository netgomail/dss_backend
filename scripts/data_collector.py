"""
Модуль сбора исторических данных по финансовым инструментам.

Этот модуль предоставляет функциональность для загрузки исторических свечных данных
из API Tinkoff Invest, расчета технических индикаторов и сохранения результатов
в формате Parquet.

Основные компоненты:
    - DataCollectorConfig: Конфигурация для сборщика данных
    - IndicatorCalculator: Расчет технических индикаторов
    - CandleDataLoader: Загрузка свечных данных из API
    - DataCollector: Основной класс для оркестрации процесса сбора данных

Пример использования:
    >>> from scripts.data_collector import DataCollector
    >>> collector = DataCollector()
    >>> collector.run()
"""

import logging
import sys
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta
from grpc import RpcError, StatusCode
from tinkoff.invest import CandleInterval, Client
from tinkoff.invest.utils import now

# === НАСТРОЙКА ПУТИ ПРОЕКТА ===
# Добавляем корень проекта в sys.path для импорта локальных модулей
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


# ============================================================================
# ПЕРЕЧИСЛЕНИЯ И КОНСТАНТЫ
# ============================================================================


class ProcessingStatus(Enum):
    """
    Статусы обработки инструмента/таймфрейма.
    
    Используется для отслеживания результата обработки данных
    и формирования отчетов.
    """
    SUCCESS = "success"           # Данные успешно обработаны и сохранены
    NO_NEW_DATA = "no_new_data"   # Нет новых данных для загрузки
    EMPTY_AFTER_CLEAN = "empty_after_clean"  # DataFrame пуст после очистки
    EMPTY_AFTER_INDICATORS = "empty_after_indicators"  # DataFrame пуст после индикаторов
    NO_NEW_ROWS = "no_new_rows"   # Нет новых строк после объединения
    ERROR = "error"               # Произошла ошибка при обработке


class ParquetEngine(Enum):
    """
    Поддерживаемые движки для работы с Parquet файлами.
    
    Приоритет: pyarrow > fastparquet
    """
    PYARROW = "pyarrow"
    FASTPARQUET = "fastparquet"


# Базовые столбцы OHLCV, которые сохраняются из сырых данных
BASE_COLUMNS: Tuple[str, ...] = ("open", "high", "low", "close", "volume")


# ============================================================================
# КОНФИГУРАЦИОННЫЕ КЛАССЫ
# ============================================================================


@dataclass(frozen=True)
class TimeframeConfig:
    """
    Конфигурация для одного таймфрейма.
    
    Attributes:
        interval: Интервал свечей из API Tinkoff Invest
        days_back: Количество дней истории для начальной загрузки
    
    Example:
        >>> config = TimeframeConfig(
        ...     interval=CandleInterval.CANDLE_INTERVAL_HOUR,
        ...     days_back=365
        ... )
    """
    interval: CandleInterval
    days_back: int


@dataclass
class DataCollectorConfig:
    """
    Основная конфигурация сборщика данных.
    
    Этот класс инкапсулирует все настраиваемые параметры для процесса
    сбора данных, включая пути, лимиты и параметры повторных попыток.
    
    Attributes:
        data_dir: Директория для сохранения данных
        min_history_rows: Минимальное количество строк для расчета индикаторов
        max_workers: Максимальное количество параллельных потоков
        request_pause_seconds: Базовая пауза между запросами к API
        request_jitter_seconds: Случайная добавка к паузе (для разнесения запросов)
        max_api_retries: Максимальное число повторных попыток при ошибке API
        base_retry_delay_seconds: Базовая задержка между повторными попытками
        timeframes: Словарь конфигураций таймфреймов
        instruments: Список инструментов для загрузки
        api_token: Токен API Tinkoff Invest
    
    Example:
        >>> config = DataCollectorConfig(
        ...     max_workers=2,
        ...     request_pause_seconds=0.5
        ... )
    """
    # Пути и директории
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "tickers")
    
    # Лимиты и пороговые значения
    min_history_rows: int = 100
    max_workers: int = 2
    
    # Параметры запросов к API
    request_pause_seconds: float = 1.0
    request_jitter_seconds: float = 0.3
    max_api_retries: int = 5
    base_retry_delay_seconds: float = 5.0
    
    # Конфигурация таймфреймов (по умолчанию - стандартный набор)
    timeframes: Dict[str, TimeframeConfig] = field(default_factory=lambda: {
        "M5": TimeframeConfig(CandleInterval.CANDLE_INTERVAL_5_MIN, 30),
        "M15": TimeframeConfig(CandleInterval.CANDLE_INTERVAL_15_MIN, 60),
        "M30": TimeframeConfig(CandleInterval.CANDLE_INTERVAL_30_MIN, 120),
        "H1": TimeframeConfig(CandleInterval.CANDLE_INTERVAL_HOUR, 365),
        "H2": TimeframeConfig(CandleInterval.CANDLE_INTERVAL_2_HOUR, 730),
        "H4": TimeframeConfig(CandleInterval.CANDLE_INTERVAL_4_HOUR, 730),
        "D1": TimeframeConfig(CandleInterval.CANDLE_INTERVAL_DAY, 3650),
        "Week": TimeframeConfig(CandleInterval.CANDLE_INTERVAL_WEEK, 3650),
    })
    
    # Инструменты и токен (берутся из settings по умолчанию)
    instruments: List[Dict[str, str]] = field(default_factory=lambda: settings.INSTRUMENTS)
    api_token: str = field(default_factory=lambda: settings.INVEST_TOKEN)
    
    def __post_init__(self) -> None:
        """Создает директорию для данных, если она не существует."""
        self.data_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# УТИЛИТЫ
# ============================================================================


def detect_parquet_engine() -> Optional[str]:
    """
    Определяет доступный движок для работы с Parquet файлами.
    
    Проверяет наличие pyarrow (приоритетный) или fastparquet.
    
    Returns:
        Название движка ('pyarrow' или 'fastparquet') или None, если
        ни один движок не установлен.
    
    Example:
        >>> engine = detect_parquet_engine()
        >>> if engine:
        ...     df.to_parquet("data.parquet", engine=engine)
    """
    try:
        import pyarrow  # noqa: F401
        return ParquetEngine.PYARROW.value
    except ImportError:
        pass
    
    try:
        import fastparquet  # noqa: F401
        return ParquetEngine.FASTPARQUET.value
    except ImportError:
        pass
    
    return None


def cast_money(value: Any) -> float:
    """
    Конвертирует денежное значение из формата Tinkoff API в float.
    
    API Tinkoff возвращает денежные значения в виде объекта с полями
    units (целая часть) и nano (миллиардные доли).
    
    Args:
        value: Объект MoneyValue из API Tinkoff
    
    Returns:
        Числовое представление денежного значения
    
    Example:
        >>> # MoneyValue(units=100, nano=500000000) -> 100.5
        >>> price = cast_money(candle.close)
    """
    return value.units + value.nano / 1e9


# ============================================================================
# РАСЧЕТ ТЕХНИЧЕСКИХ ИНДИКАТОРОВ
# ============================================================================


class BaseIndicatorGroup(ABC):
    """
    Абстрактный базовый класс для групп индикаторов.
    
    Определяет интерфейс для добавления группы связанных индикаторов
    к DataFrame. Наследники реализуют конкретную логику расчета.
    """
    
    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> None:
        """
        Рассчитывает индикаторы и добавляет их в DataFrame (in-place).
        
        Args:
            df: DataFrame с OHLCV данными
        """
        pass


class TrendIndicators(BaseIndicatorGroup):
    """
    Группа трендовых индикаторов.
    
    Включает:
        - SMA (5, 10, 20, 50): Простые скользящие средние
        - EMA (10, 20, 50, 200): Экспоненциальные скользящие средние
        - MACD: Moving Average Convergence/Divergence
    """
    
    def calculate(self, df: pd.DataFrame) -> None:
        """
        Рассчитывает трендовые индикаторы.
        
        Добавляет столбцы: sma5, sma10, sma20, sma50, ema10, ema20, ema50,
        ema200 (если достаточно данных), macd, macd_signal, macd_hist.
        
        Args:
            df: DataFrame с OHLCV данными
        """
        close = df["close"]
        
        # SMA - Simple Moving Average (разные периоды)
        df["sma5"] = ta.sma(close, length=5)
        df["sma10"] = ta.sma(close, length=10)
        df["sma20"] = ta.sma(close, length=20)
        df["sma50"] = ta.sma(close, length=50)
        
        # EMA - Exponential Moving Average (разные периоды)
        df["ema10"] = ta.ema(close, length=10)
        df["ema20"] = ta.ema(close, length=20)
        df["ema50"] = ta.ema(close, length=50)
        
        # EMA200 требует больше данных
        if len(df) >= 200:
            df["ema200"] = ta.ema(close, length=200)
        
        # MACD - Moving Average Convergence/Divergence
        macd_result = ta.macd(close)
        if macd_result is not None and not macd_result.empty:
            df["macd"] = macd_result.get("MACD_12_26_9", macd_result.iloc[:, 0])
            df["macd_signal"] = macd_result.get("MACDs_12_26_9", macd_result.iloc[:, 1])
            df["macd_hist"] = macd_result.get("MACDh_12_26_9", macd_result.iloc[:, 2])


class OscillatorIndicators(BaseIndicatorGroup):
    """
    Группа осцилляторных индикаторов.
    
    Включает:
        - RSI: Relative Strength Index
        - Stochastic: Стохастический осциллятор
        - CCI: Commodity Channel Index
        - Williams %R
    """
    
    def calculate(self, df: pd.DataFrame) -> None:
        """
        Рассчитывает осцилляторные индикаторы.
        
        Добавляет столбцы: rsi, stoch_k, stoch_d, cci, willr.
        
        Args:
            df: DataFrame с OHLCV данными
        """
        high, low, close = df["high"], df["low"], df["close"]
        
        # RSI - Relative Strength Index (период 14)
        df["rsi"] = ta.rsi(close, length=14)
        
        # Stochastic Oscillator (%K и %D линии)
        stoch_result = ta.stoch(high, low, close)
        if stoch_result is not None and not stoch_result.empty:
            df["stoch_k"] = stoch_result.get("STOCHk_14_3_3", stoch_result.iloc[:, 0])
            df["stoch_d"] = stoch_result.get("STOCHd_14_3_3", stoch_result.iloc[:, 1])
        
        # CCI - Commodity Channel Index
        df["cci"] = ta.cci(high, low, close)
        
        # Williams %R
        df["willr"] = ta.willr(high, low, close)


class VolatilityIndicators(BaseIndicatorGroup):
    """
    Группа индикаторов волатильности.
    
    Включает:
        - ATR: Average True Range
        - Bollinger Bands: Полосы Боллинджера
    """
    
    def calculate(self, df: pd.DataFrame) -> None:
        """
        Рассчитывает индикаторы волатильности.
        
        Добавляет столбцы: atr, bb_lower, bb_middle, bb_upper, bb_pband.
        
        Args:
            df: DataFrame с OHLCV данными
        """
        high, low, close = df["high"], df["low"], df["close"]
        
        # ATR - Average True Range (период 14)
        df["atr"] = ta.atr(high, low, close, length=14)
        
        # Bollinger Bands (период 20, стандартное отклонение 2)
        bb_result = ta.bbands(close, length=20)
        if bb_result is not None and not bb_result.empty:
            df["bb_lower"] = bb_result.iloc[:, 0]   # Нижняя полоса
            df["bb_middle"] = bb_result.iloc[:, 1]  # Средняя линия (SMA)
            df["bb_upper"] = bb_result.iloc[:, 2]   # Верхняя полоса
            
            # Процентная полоса: положение цены относительно полос (0-1)
            band_width = df["bb_upper"] - df["bb_lower"]
            df["bb_pband"] = np.where(
                band_width != 0,
                (close - df["bb_lower"]) / band_width,
                0
            )


class VolumeIndicators(BaseIndicatorGroup):
    """
    Группа объемных индикаторов.
    
    Включает:
        - Volume ROC: Изменение объема в процентах
        - Volume SMA: Скользящая средняя объема
        - Relative Volume: Относительный объем к средней
    """
    
    def calculate(self, df: pd.DataFrame) -> None:
        """
        Рассчитывает объемные индикаторы.
        
        Добавляет столбцы: vol_roc, vol_sma20, vol_rel.
        
        Args:
            df: DataFrame с OHLCV данными
        """
        volume = df["volume"]
        
        # Volume Rate of Change (процентное изменение)
        df["vol_roc"] = volume.pct_change()
        
        # SMA объема за 20 периодов
        df["vol_sma20"] = ta.sma(volume, length=20)
        
        # Относительный объем (текущий / средний)
        df["vol_rel"] = np.where(
            df["vol_sma20"] > 0,
            volume / df["vol_sma20"],
            1
        )


class PatternIndicators(BaseIndicatorGroup):
    """
    Группа свечных паттернов.
    
    Включает:
        - Doji: Паттерн нерешительности
        - Hammer: Паттерн молот
        - Engulfing: Паттерн поглощения
    """
    
    # Список паттернов для расчета: (имя_функции, имя_столбца)
    PATTERNS: List[Tuple[str, str]] = [
        ("cdl_doji", "pat_doji"),
        ("cdl_hammer", "pat_hammer"),
        ("cdl_engulfing", "pat_engulfing"),
    ]
    
    def calculate(self, df: pd.DataFrame) -> None:
        """
        Рассчитывает свечные паттерны.
        
        Добавляет столбцы: pat_doji, pat_hammer, pat_engulfing.
        Значения нормализованы: исходные значения (-100/0/100) делятся на 100.
        
        Args:
            df: DataFrame с OHLCV данными
        """
        for fn_name, col_name in self.PATTERNS:
            df[col_name] = self._safe_candle_pattern(fn_name, df)
    
    @staticmethod
    def _safe_candle_pattern(fn_name: str, df: pd.DataFrame) -> pd.Series:
        """
        Безопасно рассчитывает свечной паттерн.
        
        При ошибке или отсутствии функции возвращает нулевой Series.
        
        Args:
            fn_name: Имя функции паттерна в pandas_ta
            df: DataFrame с OHLCV данными
        
        Returns:
            Series с нормализованными значениями паттерна (деленными на 100)
        """
        pattern_fn = getattr(ta, fn_name, None)
        if pattern_fn is None:
            return pd.Series(0, index=df.index)
        
        try:
            series = pattern_fn(df["open"], df["high"], df["low"], df["close"])
            if series is None:
                return pd.Series(0, index=df.index)
            return series.fillna(0) / 100
        except Exception:
            return pd.Series(0, index=df.index)


class RegimeIndicators(BaseIndicatorGroup):
    """
    Группа индикаторов рыночного режима.
    
    Определяет текущее состояние рынка по нескольким параметрам:
        - Режим волатильности (высокая/низкая)
        - Режим тренда (трендовый/боковой)
        - Режим ликвидности (высокая/низкая)
    """
    
    def calculate(self, df: pd.DataFrame) -> None:
        """
        Рассчитывает индикаторы рыночного режима.
        
        Добавляет столбцы: atr_sma50, regime_vol, adx, regime_trend,
        regime_liq, market_regime.
        
        Market_regime кодирует состояние: сотни - волатильность,
        десятки - тренд, единицы - ликвидность.
        
        Args:
            df: DataFrame с OHLCV данными (должен содержать atr, vol_sma20)
        """
        high, low, close = df["high"], df["low"], df["close"]
        
        # Режим волатильности: ATR выше своей SMA50 = высокая волатильность
        df["atr_sma50"] = ta.sma(df["atr"], length=50)
        df["regime_vol"] = np.where(df["atr"] > df["atr_sma50"], 1, 0)
        
        # Режим тренда: ADX > 25 = трендовый рынок
        adx_result = ta.adx(high, low, close)
        if adx_result is not None and not adx_result.empty:
            df["adx"] = adx_result.get("ADX_14", adx_result.iloc[:, 0])
            df["regime_trend"] = np.where(df["adx"] > 25, 1, 0)
        else:
            df["regime_trend"] = 0
        
        # Режим ликвидности: объем выше SMA20 = высокая ликвидность
        df["regime_liq"] = np.where(df["volume"] > df["vol_sma20"], 1, 0)
        
        # Комбинированный код режима рынка
        # Формат: XYZ, где X=волатильность, Y=тренд, Z=ликвидность
        df["market_regime"] = (
            df["regime_vol"] * 100 +
            df["regime_trend"] * 10 +
            df["regime_liq"]
        )


class IndicatorCalculator:
    """
    Калькулятор технических индикаторов.
    
    Оркестрирует расчет всех групп технических индикаторов
    для DataFrame с OHLCV данными.
    
    Attributes:
        min_history_rows: Минимальное количество строк для расчета
        indicator_groups: Список групп индикаторов для расчета
    
    Example:
        >>> calculator = IndicatorCalculator(min_history_rows=50)
        >>> df = calculator.calculate(df)
    """
    
    def __init__(self, min_history_rows: int = 50) -> None:
        """
        Инициализирует калькулятор индикаторов.
        
        Args:
            min_history_rows: Минимальное количество строк для расчета.
                При недостаточном количестве данных индикаторы не рассчитываются.
        """
        self.min_history_rows = min_history_rows
        
        # Порядок групп важен: некоторые зависят от результатов предыдущих
        # (например, RegimeIndicators требует atr и vol_sma20)
        self.indicator_groups: List[BaseIndicatorGroup] = [
            TrendIndicators(),
            OscillatorIndicators(),
            VolatilityIndicators(),
            VolumeIndicators(),
            PatternIndicators(),
            RegimeIndicators(),
        ]
    
    def has_sufficient_history(self, df: pd.DataFrame) -> bool:
        """
        Проверяет достаточность истории для расчета индикаторов.
        
        Args:
            df: DataFrame для проверки
        
        Returns:
            True, если строк достаточно для расчета индикаторов
        """
        return len(df) >= self.min_history_rows
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитывает все технические индикаторы.
        
        Применяет все группы индикаторов последовательно.
        При недостаточном количестве данных возвращает исходный DataFrame.
        
        Args:
            df: DataFrame с OHLCV данными
        
        Returns:
            DataFrame с добавленными индикаторами
        """
        if not self.has_sufficient_history(df):
            logger.warning(
                f"Недостаточно данных для расчета индикаторов: "
                f"{len(df)} < {self.min_history_rows}"
            )
            return df
        
        try:
            for group in self.indicator_groups:
                group.calculate(df)
        except Exception as e:
            logger.error(f"Ошибка расчета индикаторов: {e}")
            return df
        
        return df


# ============================================================================
# ЗАГРУЗКА ДАННЫХ ИЗ API
# ============================================================================


class CandleDataLoader:
    """
    Загрузчик свечных данных из API Tinkoff Invest.
    
    Обеспечивает загрузку исторических свечей с механизмом повторных
    попыток при ошибках rate limiting (429).
    
    Attributes:
        client: Клиент API Tinkoff Invest
        max_retries: Максимальное количество повторных попыток
        base_retry_delay: Базовая задержка между попытками (в секундах)
    
    Example:
        >>> with Client(token) as client:
        ...     loader = CandleDataLoader(client)
        ...     candles = loader.load(figi, from_timestamp, interval)
    """
    
    def __init__(
        self,
        client: Client,
        max_retries: int = 5,
        base_retry_delay: float = 5.0
    ) -> None:
        """
        Инициализирует загрузчик данных.
        
        Args:
            client: Авторизованный клиент API Tinkoff Invest
            max_retries: Максимальное количество повторных попыток при 429
            base_retry_delay: Базовая задержка между попытками
        """
        self.client = client
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay
    
    def load(
        self,
        figi: str,
        from_timestamp: Any,
        interval: CandleInterval
    ) -> List[Dict[str, Any]]:
        """
        Загружает свечные данные из API.
        
        Реализует механизм повторных попыток с экспоненциальной задержкой
        при ошибках rate limiting.
        
        Args:
            figi: Уникальный идентификатор финансового инструмента
            from_timestamp: Начальная дата/время для загрузки
            interval: Интервал свечей (CandleInterval)
        
        Returns:
            Список словарей с данными свечей (time, open, high, low, close, volume)
            или пустой список при ошибке
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._fetch_candles(figi, from_timestamp, interval)
            except RpcError as e:
                if not self._handle_rpc_error(e, figi, attempt):
                    return []
            except Exception as e:
                logger.error(f"Неизвестная ошибка при загрузке {figi}: {e}")
                return []
        
        logger.error(f"Не удалось получить данные после {self.max_retries} попыток")
        return []
    
    def _fetch_candles(
        self,
        figi: str,
        from_timestamp: Any,
        interval: CandleInterval
    ) -> List[Dict[str, Any]]:
        """
        Выполняет непосредственный запрос к API.
        
        Args:
            figi: FIGI инструмента
            from_timestamp: Начальная временная метка
            interval: Интервал свечей
        
        Returns:
            Список словарей с данными свечей
        """
        return [
            {
                "time": candle.time,
                "open": cast_money(candle.open),
                "high": cast_money(candle.high),
                "low": cast_money(candle.low),
                "close": cast_money(candle.close),
                "volume": candle.volume,
            }
            for candle in self.client.get_all_candles(
                figi=figi,
                from_=from_timestamp,
                interval=interval
            )
        ]
    
    def _handle_rpc_error(self, error: RpcError, figi: str, attempt: int) -> bool:
        """
        Обрабатывает RPC ошибку от API.
        
        Args:
            error: Исключение RpcError
            figi: FIGI инструмента (для логирования)
            attempt: Номер текущей попытки
        
        Returns:
            True, если следует повторить попытку; False для прекращения
        """
        if error.code() == StatusCode.RESOURCE_EXHAUSTED:
            # Rate limiting - ждем и повторяем
            wait_time = self.base_retry_delay * attempt + np.random.uniform(0, 1)
            logger.warning(
                f"Лимит запросов (429) для {figi}. "
                f"Ожидание {wait_time:.1f} сек (попытка {attempt}/{self.max_retries})"
            )
            time.sleep(wait_time)
            return True
        
        # Другие ошибки API - прекращаем попытки
        logger.error(f"Критическая ошибка API для {figi}: {error}")
        return False


# ============================================================================
# РАБОТА С ФАЙЛАМИ ДАННЫХ
# ============================================================================


class DataFileManager:
    """
    Менеджер файлов данных в формате Parquet.
    
    Управляет чтением и записью файлов с данными, включая
    определение доступного движка Parquet.
    
    Attributes:
        data_dir: Базовая директория для хранения данных
        parquet_engine: Используемый движок Parquet
    
    Example:
        >>> manager = DataFileManager(Path("data/tickers"))
        >>> df = manager.load_base_data("SBER", "H1")
        >>> manager.save_data(df, "SBER", "H1")
    """
    
    def __init__(self, data_dir: Path) -> None:
        """
        Инициализирует менеджер файлов.
        
        Args:
            data_dir: Директория для хранения данных
        
        Raises:
            RuntimeError: Если не найден движок Parquet
        """
        self.data_dir = data_dir
        self.parquet_engine = detect_parquet_engine()
        
        if self.parquet_engine is None:
            raise RuntimeError(
                "Не найден движок Parquet. Установите pyarrow: pip install pyarrow"
            )
    
    def get_file_path(self, ticker: str, timeframe: str) -> Path:
        """
        Формирует путь к файлу данных.
        
        Args:
            ticker: Тикер инструмента
            timeframe: Название таймфрейма
        
        Returns:
            Путь к файлу Parquet
        """
        return self.data_dir / ticker / f"{timeframe}.parquet"
    
    def ensure_ticker_dir(self, ticker: str) -> Path:
        """
        Создает директорию для тикера, если не существует.
        
        Args:
            ticker: Тикер инструмента
        
        Returns:
            Путь к директории тикера
        """
        ticker_dir = self.data_dir / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        return ticker_dir
    
    def load_base_data(self, ticker: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Загружает существующие базовые данные (OHLCV).
        
        Args:
            ticker: Тикер инструмента
            timeframe: Название таймфрейма
        
        Returns:
            DataFrame с базовыми столбцами или None, если файл не существует
        """
        file_path = self.get_file_path(ticker, timeframe)
        
        if not file_path.exists():
            return None
        
        try:
            df = pd.read_parquet(file_path, engine=self.parquet_engine)
            return df[list(BASE_COLUMNS)]
        except Exception as e:
            logger.warning(f"Не удалось прочитать {file_path.name}: {e}")
            return None
    
    def save_data(
        self,
        df: pd.DataFrame,
        ticker: str,
        timeframe: str,
        compression: str = "snappy"
    ) -> bool:
        """
        Сохраняет данные в Parquet файл.
        
        Args:
            df: DataFrame для сохранения
            ticker: Тикер инструмента
            timeframe: Название таймфрейма
            compression: Алгоритм сжатия
        
        Returns:
            True при успешном сохранении
        """
        file_path = self.get_file_path(ticker, timeframe)
        
        try:
            df.to_parquet(
                file_path,
                compression=compression,
                engine=self.parquet_engine
            )
            return True
        except Exception as e:
            logger.error(f"Ошибка сохранения {file_path}: {e}")
            return False


# ============================================================================
# ОБРАБОТКА ДАННЫХ
# ============================================================================


class DataFrameCleaner:
    """
    Класс для очистки и подготовки DataFrame с данными свечей.
    
    Выполняет:
        - Удаление дубликатов по индексу
        - Фильтрацию некорректных данных (нулевой объем/цена)
        - Исключение выходных дней
        - Удаление полностью пустых строк
    """
    
    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        """
        Очищает DataFrame от некорректных и дублирующихся данных.
        
        Args:
            df: DataFrame для очистки
        
        Returns:
            Очищенный и отсортированный по индексу DataFrame
        """
        # Удаляем дубликаты по индексу (оставляем первое вхождение)
        df = df[~df.index.duplicated(keep="first")]
        
        # Фильтруем строки с нулевым объемом или ценой
        df = df[df["volume"] > 0]
        df = df[df["close"] > 0]
        
        # Исключаем выходные дни (суббота=5, воскресенье=6)
        df = df[~df.index.weekday.isin([5, 6])]
        
        # Удаляем полностью пустые строки
        df = df.dropna(how="all")
        
        # Сортируем по времени
        return df.sort_index()


# ============================================================================
# ОСНОВНОЙ СБОРЩИК ДАННЫХ
# ============================================================================


class DataCollector:
    """
    Основной класс для сбора и обработки финансовых данных.
    
    Оркестрирует весь процесс:
        1. Загрузка существующих данных
        2. Получение новых данных из API
        3. Объединение и очистка данных
        4. Расчет технических индикаторов
        5. Сохранение результатов
    
    Attributes:
        config: Конфигурация сборщика
        file_manager: Менеджер файлов данных
        indicator_calculator: Калькулятор индикаторов
        data_cleaner: Очиститель данных
    
    Example:
        >>> collector = DataCollector()
        >>> collector.run()
    """
    
    def __init__(self, config: Optional[DataCollectorConfig] = None) -> None:
        """
        Инициализирует сборщик данных.
        
        Args:
            config: Конфигурация сборщика (по умолчанию создается стандартная)
        """
        self.config = config or DataCollectorConfig()
        self.file_manager = DataFileManager(self.config.data_dir)
        self.indicator_calculator = IndicatorCalculator(self.config.min_history_rows)
        self.data_cleaner = DataFrameCleaner()
    
    def run(self) -> None:
        """
        Запускает процесс сбора данных для всех инструментов.
        
        Создает клиент API, запускает параллельную обработку инструментов
        и выводит отчет о результатах.
        """
        with Client(self.config.api_token) as client:
            logger.info(
                f"Старт загрузки для {len(self.config.instruments)} тикеров"
            )
            
            self._process_all_instruments(client)
        
        logger.info("Загрузка и обработка завершены")
        self._check_missing_files()
    
    def _process_all_instruments(self, client: Client) -> None:
        """
        Обрабатывает все инструменты параллельно.
        
        Args:
            client: Авторизованный клиент API
        """
        total = len(self.config.instruments)
        workers = min(self.config.max_workers, total) or 1
        
        logger.info(f"Параллельная загрузка: {total} тикеров, потоки: {workers}")
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Создаем задачи для всех инструментов
            futures = {
                executor.submit(
                    self._process_instrument, client, instrument
                ): instrument["name"]
                for instrument in self.config.instruments
            }
            
            # Обрабатываем результаты по мере завершения
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    logger.error(f"Ошибка при обработке {ticker}: {exc}")
    
    def _process_instrument(
        self,
        client: Client,
        instrument: Dict[str, str]
    ) -> None:
        """
        Обрабатывает один инструмент (все таймфреймы).
        
        Args:
            client: Авторизованный клиент API
            instrument: Словарь с данными инструмента (name, figi)
        """
        ticker = instrument["name"]
        figi = instrument["figi"]
        
        # Создаем директорию для тикера
        self.file_manager.ensure_ticker_dir(ticker)
        
        # Создаем загрузчик данных
        candle_loader = CandleDataLoader(
            client,
            max_retries=self.config.max_api_retries,
            base_retry_delay=self.config.base_retry_delay_seconds
        )
        
        # Обрабатываем каждый таймфрейм
        for tf_name, tf_config in self.config.timeframes.items():
            status = self._process_timeframe(
                ticker, figi, tf_name, tf_config, candle_loader
            )
            self._log_status(ticker, tf_name, status)
    
    def _process_timeframe(
        self,
        ticker: str,
        figi: str,
        tf_name: str,
        tf_config: TimeframeConfig,
        candle_loader: CandleDataLoader
    ) -> ProcessingStatus:
        """
        Обрабатывает один таймфрейм инструмента.
        
        Args:
            ticker: Тикер инструмента
            figi: FIGI инструмента
            tf_name: Название таймфрейма
            tf_config: Конфигурация таймфрейма
            candle_loader: Загрузчик данных
        
        Returns:
            Статус обработки
        """
        # Загружаем существующие данные
        existing_base = self.file_manager.load_base_data(ticker, tf_name)
        
        logger.info(f"{ticker} | {tf_name} загрузка...")
        
        # Пауза между запросами (с джиттером)
        pause = (
            self.config.request_pause_seconds +
            np.random.uniform(0, self.config.request_jitter_seconds)
        )
        time.sleep(pause)
        
        # Определяем начальную точку загрузки
        from_timestamp = self._get_start_timestamp(existing_base, tf_config)
        
        # Загружаем новые данные
        candles = candle_loader.load(figi, from_timestamp, tf_config.interval)
        
        if not candles:
            return ProcessingStatus.NO_NEW_DATA
        
        # Создаем DataFrame из новых данных
        df_new = pd.DataFrame(candles).set_index("time")
        
        # Объединяем с существующими данными
        df = (
            df_new if existing_base is None
            else pd.concat([existing_base, df_new])
        )
        
        # Очищаем данные
        df = self.data_cleaner.clean(df)
        
        if df.empty:
            return ProcessingStatus.EMPTY_AFTER_CLEAN
        
        # Проверяем, есть ли новые строки
        previous_len = len(existing_base) if existing_base is not None else 0
        if len(df) <= previous_len:
            return ProcessingStatus.NO_NEW_ROWS
        
        # Рассчитываем индикаторы
        df = self.indicator_calculator.calculate(df)
        df = df.dropna()
        
        if df.empty:
            return ProcessingStatus.EMPTY_AFTER_INDICATORS
        
        # Сохраняем результат
        if self.file_manager.save_data(df, ticker, tf_name):
            return ProcessingStatus.SUCCESS
        
        return ProcessingStatus.ERROR
    
    @staticmethod
    def _get_start_timestamp(
        existing: Optional[pd.DataFrame],
        tf_config: TimeframeConfig
    ) -> Any:
        """
        Определяет начальную временную метку для загрузки.
        
        Args:
            existing: Существующий DataFrame (может быть None)
            tf_config: Конфигурация таймфрейма
        
        Returns:
            Временная метка для начала загрузки
        """
        if existing is None or existing.empty:
            # Нет существующих данных - загружаем всю историю
            return now() - timedelta(days=tf_config.days_back)
        
        # Есть данные - продолжаем с последней свечи
        last_ts = existing.index.max()
        
        # Добавляем timezone, если отсутствует
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=now().tzinfo)
        
        # +1 секунда, чтобы не загружать последнюю свечу повторно
        return last_ts + timedelta(seconds=1)
    
    def _log_status(
        self,
        ticker: str,
        tf_name: str,
        status: ProcessingStatus
    ) -> None:
        """
        Логирует статус обработки таймфрейма.
        
        Args:
            ticker: Тикер инструмента
            tf_name: Название таймфрейма
            status: Статус обработки
        """
        status_messages = {
            ProcessingStatus.SUCCESS: "сохранено",
            ProcessingStatus.NO_NEW_DATA: "нет новых данных",
            ProcessingStatus.EMPTY_AFTER_CLEAN: "пусто после очистки",
            ProcessingStatus.EMPTY_AFTER_INDICATORS: "пусто после индикаторов",
            ProcessingStatus.NO_NEW_ROWS: "нет новых строк",
            ProcessingStatus.ERROR: "ошибка",
        }
        
        message = status_messages.get(status, "неизвестный статус")
        
        if status == ProcessingStatus.SUCCESS:
            logger.info(f"✓ {ticker} | {tf_name}: {message}")
        elif status == ProcessingStatus.ERROR:
            logger.error(f"✗ {ticker} | {tf_name}: {message}")
        else:
            logger.warning(f"○ {ticker} | {tf_name}: {message}")
    
    def _check_missing_files(self) -> None:
        """
        Проверяет наличие всех ожидаемых файлов данных.
        
        Выводит список отсутствующих файлов, если они есть.
        """
        logger.info("Проверка целостности данных...")
        
        missing: List[str] = []
        
        for instrument in self.config.instruments:
            ticker = instrument["name"]
            
            for tf_name in self.config.timeframes.keys():
                file_path = self.file_manager.get_file_path(ticker, tf_name)
                if not file_path.exists():
                    missing.append(f"{ticker} - {tf_name}")
        
        if missing:
            logger.warning(f"Отсутствуют файлы ({len(missing)} шт.):")
            for m in missing:
                logger.warning(f"  - {m}")
        else:
            logger.info("Все файлы успешно загружены!")


# ============================================================================
# ТОЧКА ВХОДА
# ============================================================================


def main() -> None:
    """
    Точка входа для запуска сборщика данных.
    
    Создает экземпляр DataCollector с конфигурацией по умолчанию
    и запускает процесс сбора данных.
    """
    collector = DataCollector()
    collector.run()


if __name__ == "__main__":
    main()
