"""
Модуль генерации торговых сигналов.

Этот модуль реализует многоуровневую систему фильтрации для генерации
торговых сигналов на основе технических индикаторов.

Архитектура фильтрации (5 уровней):
    Уровень 1: Режим рынка (market regime)
    Уровень 2: Тренд (trend direction)
    Уровень 3: Волатильность (volatility)
    Уровень 4: Объёмы + Паттерны (volume + patterns)
    Уровень 5: Осцилляторы (oscillators confirmation)

Пример использования:
    >>> from scripts.signal_generator import SignalGenerator
    >>> generator = SignalGenerator()
    >>> signals = generator.generate_signals("SBER", "D1")
"""

import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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


# ============================================================================
# ПЕРЕЧИСЛЕНИЯ И ТИПЫ
# ============================================================================


class SignalType(IntEnum):
    """
    Типы торговых сигналов.
    
    Значения:
        STRONG_SELL (-2): Сильный сигнал на продажу
        SELL (-1): Обычный сигнал на продажу
        NEUTRAL (0): Нет сигнала
        BUY (1): Обычный сигнал на покупку
        STRONG_BUY (2): Сильный сигнал на покупку
    """
    STRONG_SELL = -2
    SELL = -1
    NEUTRAL = 0
    BUY = 1
    STRONG_BUY = 2


class MarketRegime(IntEnum):
    """
    Режимы рынка на основе комбинации волатильности, тренда и ликвидности.
    
    Формат кода: XYZ
        X (сотни) — волатильность: 0 = низкая, 1 = высокая
        Y (десятки) — тренд: 0 = боковой, 1 = трендовый
        Z (единицы) — ликвидность: 0 = низкая, 1 = высокая
    """
    # Низкая волатильность
    LOW_VOL_RANGE_LOW_LIQ = 0      # 000: Спокойный рынок, мало движения
    LOW_VOL_RANGE_HIGH_LIQ = 1     # 001: Спокойный рынок с объёмами
    LOW_VOL_TREND_LOW_LIQ = 10     # 010: Тренд без объёмов (слабый)
    LOW_VOL_TREND_HIGH_LIQ = 11    # 011: Хороший тренд
    
    # Высокая волатильность
    HIGH_VOL_RANGE_LOW_LIQ = 100   # 100: Волатильный боковик (опасно)
    HIGH_VOL_RANGE_HIGH_LIQ = 101  # 101: Волатильный боковик с объёмами
    HIGH_VOL_TREND_LOW_LIQ = 110   # 110: Волатильный тренд без подтверждения
    HIGH_VOL_TREND_HIGH_LIQ = 111  # 111: Сильный тренд (лучшее время для торговли)


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================


@dataclass
class SignalConfig:
    """
    Конфигурация параметров генерации сигналов.
    
    Attributes:
        rsi_oversold: Уровень перепроданности RSI
        rsi_overbought: Уровень перекупленности RSI
        stoch_oversold: Уровень перепроданности Stochastic
        stoch_overbought: Уровень перекупленности Stochastic
        cci_oversold: Уровень перепроданности CCI
        cci_overbought: Уровень перекупленности CCI
        volume_threshold: Минимальный относительный объём для подтверждения
        adx_trend_threshold: Минимальный ADX для определения тренда
        bb_squeeze_threshold: Порог сжатия полос Боллинджера
        min_filters_passed: Минимум пройденных уровней фильтрации
    """
    # Уровни осцилляторов
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    stoch_oversold: float = 20.0
    stoch_overbought: float = 80.0
    cci_oversold: float = -100.0
    cci_overbought: float = 100.0
    willr_oversold: float = -80.0
    willr_overbought: float = -20.0
    
    # Объёмы
    volume_threshold: float = 1.2  # 120% от среднего
    
    # Тренд
    adx_trend_threshold: float = 25.0
    adx_strong_trend: float = 40.0
    
    # Волатильность
    bb_squeeze_threshold: float = 0.02  # Сжатие BB < 2%
    
    # Фильтрация
    min_filters_passed: int = 3  # Минимум 3 из 5 уровней


# ============================================================================
# БАЗОВЫЕ КЛАССЫ ФИЛЬТРОВ
# ============================================================================


class BaseFilter(ABC):
    """
    Абстрактный базовый класс для фильтров сигналов.
    
    Каждый фильтр представляет один уровень системы фильтрации
    и возвращает направление сигнала или None если условия не выполнены.
    """
    
    def __init__(self, config: SignalConfig):
        """
        Инициализация фильтра.
        
        Args:
            config: Конфигурация параметров сигналов
        """
        self.config = config
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Название фильтра для логирования."""
        pass
    
    @property
    @abstractmethod
    def level(self) -> int:
        """Уровень фильтра (1-5)."""
        pass
    
    @abstractmethod
    def evaluate(self, row: pd.Series) -> Optional[int]:
        """
        Оценка условий фильтра для одной строки данных.
        
        Args:
            row: Строка DataFrame с индикаторами
            
        Returns:
            1 для BUY, -1 для SELL, 0 для NEUTRAL, None если фильтр не применим
        """
        pass


# ============================================================================
# УРОВЕНЬ 1: ФИЛЬТР РЕЖИМА РЫНКА
# ============================================================================


class MarketRegimeFilter(BaseFilter):
    """
    Уровень 1: Фильтр режима рынка.
    
    Определяет благоприятность текущего режима рынка для торговли.
    Использует комбинацию волатильности, тренда и ликвидности.
    
    Благоприятные режимы:
        - 011 (LOW_VOL_TREND_HIGH_LIQ): Низкая волатильность + тренд + объёмы
        - 111 (HIGH_VOL_TREND_HIGH_LIQ): Высокая волатильность + тренд + объёмы
    """
    
    # Режимы, благоприятные для торговли
    FAVORABLE_REGIMES = {
        MarketRegime.LOW_VOL_TREND_HIGH_LIQ,   # 011
        MarketRegime.HIGH_VOL_TREND_HIGH_LIQ,  # 111
    }
    
    # Режимы, требующие осторожности (только контртренд)
    CAUTION_REGIMES = {
        MarketRegime.LOW_VOL_RANGE_HIGH_LIQ,   # 001
        MarketRegime.HIGH_VOL_RANGE_HIGH_LIQ,  # 101
    }
    
    @property
    def name(self) -> str:
        return "Market Regime"
    
    @property
    def level(self) -> int:
        return 1
    
    def evaluate(self, row: pd.Series) -> Optional[int]:
        """
        Оценка режима рынка.
        
        Returns:
            1 — благоприятный режим для торговли
            0 — нейтральный режим (требует осторожности)
            -1 — неблагоприятный режим (не торговать)
            None — данные недоступны
        """
        regime = row.get("market_regime")
        
        if pd.isna(regime):
            return None
        
        regime = int(regime)
        
        if regime in [r.value for r in self.FAVORABLE_REGIMES]:
            return 1  # Режим благоприятен для торговли
        elif regime in [r.value for r in self.CAUTION_REGIMES]:
            return 0  # Режим нейтральный
        else:
            return -1  # Режим неблагоприятен


# ============================================================================
# УРОВЕНЬ 2: ФИЛЬТР ТРЕНДА
# ============================================================================


class TrendFilter(BaseFilter):
    """
    Уровень 2: Фильтр направления тренда.
    
    Определяет направление тренда на основе:
        - Взаимного расположения EMA (10, 20, 50)
        - Направления MACD
        - Положения цены относительно EMA
    """
    
    @property
    def name(self) -> str:
        return "Trend"
    
    @property
    def level(self) -> int:
        return 2
    
    def evaluate(self, row: pd.Series) -> Optional[int]:
        """
        Оценка направления тренда.
        
        Returns:
            1 — восходящий тренд (BUY)
            -1 — нисходящий тренд (SELL)
            0 — боковой тренд (NEUTRAL)
            None — данные недоступны
        """
        close = row.get("close")
        ema10 = row.get("ema10")
        ema20 = row.get("ema20")
        ema50 = row.get("ema50")
        macd = row.get("macd")
        macd_signal = row.get("macd_signal")
        
        # Проверка доступности данных
        required = [close, ema10, ema20, ema50, macd, macd_signal]
        if any(pd.isna(v) for v in required):
            return None
        
        # Подсчёт бычьих и медвежьих сигналов
        bullish_score = 0
        bearish_score = 0
        
        # 1. Расположение EMA: EMA10 > EMA20 > EMA50 = бычий
        if ema10 > ema20 > ema50:
            bullish_score += 2
        elif ema10 < ema20 < ema50:
            bearish_score += 2
        
        # 2. Цена выше/ниже EMA20
        if close > ema20:
            bullish_score += 1
        elif close < ema20:
            bearish_score += 1
        
        # 3. MACD выше/ниже сигнальной линии
        if macd > macd_signal:
            bullish_score += 1
        elif macd < macd_signal:
            bearish_score += 1
        
        # Итоговая оценка
        if bullish_score >= 3 and bullish_score > bearish_score:
            return 1  # Восходящий тренд
        elif bearish_score >= 3 and bearish_score > bullish_score:
            return -1  # Нисходящий тренд
        else:
            return 0  # Боковой тренд


# ============================================================================
# УРОВЕНЬ 3: ФИЛЬТР ВОЛАТИЛЬНОСТИ
# ============================================================================


class VolatilityFilter(BaseFilter):
    """
    Уровень 3: Фильтр волатильности.
    
    Оценивает текущую волатильность и её изменение:
        - ATR относительно своей средней
        - Ширина полос Боллинджера
        - Сжатие BB как предвестник движения
    """
    
    @property
    def name(self) -> str:
        return "Volatility"
    
    @property
    def level(self) -> int:
        return 3
    
    def evaluate(self, row: pd.Series) -> Optional[int]:
        """
        Оценка условий волатильности.
        
        Returns:
            1 — волатильность благоприятна для входа
            0 — нейтральная волатильность
            -1 — слишком высокая волатильность (опасно)
            None — данные недоступны
        """
        atr = row.get("atr")
        atr_sma50 = row.get("atr_sma50")
        bb_upper = row.get("bb_upper")
        bb_lower = row.get("bb_lower")
        bb_middle = row.get("bb_middle")
        close = row.get("close")
        
        required = [atr, atr_sma50, bb_upper, bb_lower, bb_middle, close]
        if any(pd.isna(v) for v in required):
            return None
        
        # Относительная волатильность (ATR / ATR_SMA50)
        atr_ratio = atr / atr_sma50 if atr_sma50 > 0 else 1
        
        # Ширина BB относительно средней
        bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0
        
        # Сжатие полос (потенциальный пробой)
        is_squeeze = bb_width < self.config.bb_squeeze_threshold
        
        # Оценка
        if is_squeeze:
            # Сжатие BB — ожидается движение, благоприятно для входа
            return 1
        elif atr_ratio > 2.0:
            # Слишком высокая волатильность — опасно
            return -1
        elif 0.8 <= atr_ratio <= 1.5:
            # Нормальная волатильность — благоприятно
            return 1
        else:
            return 0


# ============================================================================
# УРОВЕНЬ 4: ФИЛЬТР ОБЪЁМОВ И ПАТТЕРНОВ
# ============================================================================


class VolumePatternFilter(BaseFilter):
    """
    Уровень 4: Фильтр объёмов и свечных паттернов.
    
    Проверяет подтверждение движения объёмами:
        - Относительный объём выше порога
        - Наличие свечных паттернов (doji, hammer, engulfing)
    """
    
    @property
    def name(self) -> str:
        return "Volume + Patterns"
    
    @property
    def level(self) -> int:
        return 4
    
    def evaluate(self, row: pd.Series) -> Optional[int]:
        """
        Оценка объёмов и паттернов.
        
        Returns:
            1 — объёмы подтверждают покупку + бычий паттерн
            -1 — объёмы подтверждают продажу + медвежий паттерн
            0 — нет подтверждения
            None — данные недоступны
        """
        vol_rel = row.get("vol_rel")
        pat_doji = row.get("pat_doji", 0)
        pat_hammer = row.get("pat_hammer", 0)
        pat_engulfing = row.get("pat_engulfing", 0)
        
        if pd.isna(vol_rel):
            return None
        
        # Проверка объёма
        high_volume = vol_rel >= self.config.volume_threshold
        
        # Суммарный сигнал паттернов
        pattern_signal = pat_hammer + pat_engulfing  # hammer и engulfing дают направление
        
        # Оценка
        if high_volume:
            if pattern_signal > 0:
                return 1  # Бычий сигнал с подтверждением объёмом
            elif pattern_signal < 0:
                return -1  # Медвежий сигнал с подтверждением объёмом
            else:
                # Высокий объём без паттерна — нейтрально
                return 0
        else:
            # Низкий объём — слабый сигнал
            if abs(pattern_signal) > 0:
                return int(np.sign(pattern_signal))
            return 0


# ============================================================================
# УРОВЕНЬ 5: ФИЛЬТР ОСЦИЛЛЯТОРОВ
# ============================================================================


class OscillatorFilter(BaseFilter):
    """
    Уровень 5: Фильтр осцилляторов.
    
    Финальное подтверждение на основе осцилляторов:
        - RSI (перекупленность/перепроданность)
        - Stochastic (пересечения)
        - CCI
        - Williams %R
    """
    
    @property
    def name(self) -> str:
        return "Oscillators"
    
    @property
    def level(self) -> int:
        return 5
    
    def evaluate(self, row: pd.Series) -> Optional[int]:
        """
        Оценка осцилляторов.
        
        Returns:
            1 — осцилляторы указывают на покупку
            -1 — осцилляторы указывают на продажу
            0 — нет единого мнения
            None — данные недоступны
        """
        rsi = row.get("rsi")
        stoch_k = row.get("stoch_k")
        stoch_d = row.get("stoch_d")
        cci = row.get("cci")
        willr = row.get("willr")
        
        required = [rsi, stoch_k, stoch_d, cci, willr]
        if any(pd.isna(v) for v in required):
            return None
        
        bullish_count = 0
        bearish_count = 0
        
        # RSI
        if rsi < self.config.rsi_oversold:
            bullish_count += 1  # Перепроданность — сигнал на покупку
        elif rsi > self.config.rsi_overbought:
            bearish_count += 1  # Перекупленность — сигнал на продажу
        
        # Stochastic: %K пересекает %D снизу = бычий
        if stoch_k < self.config.stoch_oversold and stoch_k > stoch_d:
            bullish_count += 1
        elif stoch_k > self.config.stoch_overbought and stoch_k < stoch_d:
            bearish_count += 1
        
        # CCI
        if cci < self.config.cci_oversold:
            bullish_count += 1
        elif cci > self.config.cci_overbought:
            bearish_count += 1
        
        # Williams %R
        if willr < self.config.willr_oversold:
            bullish_count += 1
        elif willr > self.config.willr_overbought:
            bearish_count += 1
        
        # Итоговая оценка (нужно минимум 2 из 4)
        if bullish_count >= 2 and bullish_count > bearish_count:
            return 1
        elif bearish_count >= 2 and bearish_count > bullish_count:
            return -1
        else:
            return 0


# ============================================================================
# ГЕНЕРАТОР СИГНАЛОВ
# ============================================================================


@dataclass
class SignalResult:
    """
    Результат генерации сигнала для одной свечи.
    
    Attributes:
        time: Время свечи
        ticker: Тикер инструмента
        timeframe: Таймфрейм
        signal: Итоговый сигнал
        strength: Сила сигнала (количество пройденных фильтров)
        filter_results: Результаты каждого фильтра
        price: Цена закрытия
    """
    time: datetime
    ticker: str
    timeframe: str
    signal: SignalType
    strength: int
    filter_results: Dict[str, Optional[int]]
    price: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "time": self.time,
            "ticker": self.ticker,
            "timeframe": self.timeframe,
            "signal": self.signal.name,
            "signal_value": self.signal.value,
            "strength": self.strength,
            "price": self.price,
            **{f"filter_{k}": v for k, v in self.filter_results.items()},
        }


class SignalGenerator:
    """
    Генератор торговых сигналов.
    
    Применяет 5-уровневую систему фильтрации для генерации
    надёжных торговых сигналов.
    
    Архитектура:
        Уровень 1: Режим рынка → Проходит ли рынок для торговли?
        Уровень 2: Тренд → Какое направление движения?
        Уровень 3: Волатильность → Подходящие ли условия?
        Уровень 4: Объёмы + Паттерны → Есть ли подтверждение?
        Уровень 5: Осцилляторы → Финальное подтверждение
    
    Example:
        >>> generator = SignalGenerator()
        >>> signals = generator.generate_signals("SBER", "D1")
        >>> # Получить только сильные сигналы
        >>> strong = [s for s in signals if s.strength >= 4]
    """
    
    def __init__(self, config: Optional[SignalConfig] = None):
        """
        Инициализация генератора сигналов.
        
        Args:
            config: Конфигурация параметров (используется дефолтная если не задана)
        """
        self.config = config or SignalConfig()
        self.data_path = PROJECT_ROOT / "data" / "tickers"
        
        # Инициализация фильтров (порядок важен!)
        self.filters: List[BaseFilter] = [
            MarketRegimeFilter(self.config),
            TrendFilter(self.config),
            VolatilityFilter(self.config),
            VolumePatternFilter(self.config),
            OscillatorFilter(self.config),
        ]
    
    def load_data(self, ticker: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Загрузка данных инструмента.
        
        Args:
            ticker: Тикер инструмента
            timeframe: Таймфрейм (D1, H1, M15 и т.д.)
            
        Returns:
            DataFrame с данными или None при ошибке
        """
        file_path = self.data_path / ticker / f"{timeframe}.parquet"
        
        if not file_path.exists():
            logger.error(f"Файл не найден: {file_path}")
            return None
        
        try:
            df = pd.read_parquet(file_path)
            logger.info(f"Загружено {len(df)} свечей для {ticker}/{timeframe}")
            return df
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            return None
    
    def _evaluate_row(self, row: pd.Series, ticker: str, timeframe: str) -> SignalResult:
        """
        Оценка одной строки через все фильтры.
        
        Args:
            row: Строка DataFrame
            ticker: Тикер инструмента
            timeframe: Таймфрейм
            
        Returns:
            Результат генерации сигнала
        """
        filter_results: Dict[str, Optional[int]] = {}
        
        # Проход по всем фильтрам
        for f in self.filters:
            result = f.evaluate(row)
            filter_results[f.name] = result
        
        # Подсчёт пройденных фильтров и направления
        bullish_count = 0
        bearish_count = 0
        passed_count = 0
        
        for name, result in filter_results.items():
            if result is not None and result != 0:
                passed_count += 1
                if result > 0:
                    bullish_count += 1
                else:
                    bearish_count += 1
        
        # Определение итогового сигнала
        if passed_count < self.config.min_filters_passed:
            signal = SignalType.NEUTRAL
        elif bullish_count > bearish_count:
            if bullish_count >= 4:
                signal = SignalType.STRONG_BUY
            else:
                signal = SignalType.BUY
        elif bearish_count > bullish_count:
            if bearish_count >= 4:
                signal = SignalType.STRONG_SELL
            else:
                signal = SignalType.SELL
        else:
            signal = SignalType.NEUTRAL
        
        return SignalResult(
            time=row.name,
            ticker=ticker,
            timeframe=timeframe,
            signal=signal,
            strength=passed_count,
            filter_results=filter_results,
            price=row.get("close", 0),
        )
    
    def generate_signals(
        self,
        ticker: str,
        timeframe: str,
        last_n: Optional[int] = None,
    ) -> List[SignalResult]:
        """
        Генерация сигналов для инструмента.
        
        Args:
            ticker: Тикер инструмента
            timeframe: Таймфрейм
            last_n: Количество последних свечей для анализа (None = все)
            
        Returns:
            Список результатов сигналов
        """
        df = self.load_data(ticker, timeframe)
        
        if df is None or df.empty:
            return []
        
        # Ограничение количества свечей
        if last_n is not None:
            df = df.tail(last_n)
        
        signals: List[SignalResult] = []
        
        for idx, row in df.iterrows():
            result = self._evaluate_row(row, ticker, timeframe)
            signals.append(result)
        
        # Статистика
        active_signals = [s for s in signals if s.signal != SignalType.NEUTRAL]
        logger.info(
            f"Сгенерировано сигналов: {len(active_signals)} из {len(signals)} свечей"
        )
        
        return signals
    
    def generate_signals_df(
        self,
        ticker: str,
        timeframe: str,
        last_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Генерация сигналов в формате DataFrame.
        
        Args:
            ticker: Тикер инструмента
            timeframe: Таймфрейм
            last_n: Количество последних свечей для анализа
            
        Returns:
            DataFrame с сигналами
        """
        signals = self.generate_signals(ticker, timeframe, last_n)
        
        if not signals:
            return pd.DataFrame()
        
        df = pd.DataFrame([s.to_dict() for s in signals])
        df.set_index("time", inplace=True)
        
        return df
    
    def get_active_signals(
        self,
        ticker: str,
        timeframe: str,
        min_strength: int = 3,
    ) -> List[SignalResult]:
        """
        Получение только активных (не нейтральных) сигналов.
        
        Args:
            ticker: Тикер инструмента
            timeframe: Таймфрейм
            min_strength: Минимальная сила сигнала
            
        Returns:
            Список активных сигналов
        """
        signals = self.generate_signals(ticker, timeframe)
        
        return [
            s for s in signals
            if s.signal != SignalType.NEUTRAL and s.strength >= min_strength
        ]
    
    def scan_all_instruments(
        self,
        timeframe: str = "D1",
        min_strength: int = 3,
    ) -> pd.DataFrame:
        """
        Сканирование всех инструментов на наличие сигналов.
        
        Args:
            timeframe: Таймфрейм для анализа
            min_strength: Минимальная сила сигнала
            
        Returns:
            DataFrame с последними сигналами по каждому инструменту
        """
        results = []
        
        for instrument in settings.INSTRUMENTS:
            ticker = instrument["name"]
            
            # Получаем только последнюю свечу
            signals = self.generate_signals(ticker, timeframe, last_n=1)
            
            if signals and signals[0].signal != SignalType.NEUTRAL:
                if signals[0].strength >= min_strength:
                    results.append(signals[0].to_dict())
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df.sort_values("strength", ascending=False, inplace=True)
        
        return df


# ============================================================================
# ТОЧКА ВХОДА
# ============================================================================


def main():
    """Демонстрация работы генератора сигналов."""
    generator = SignalGenerator()
    
    # Генерация сигналов для SBER
    print("\n" + "=" * 60)
    print("ГЕНЕРАЦИЯ ТОРГОВЫХ СИГНАЛОВ")
    print("=" * 60)
    
    # Последние 10 свечей SBER D1
    signals_df = generator.generate_signals_df("SBER", "D1", last_n=10)
    
    if not signals_df.empty:
        print(f"\nПоследние сигналы SBER (D1):")
        print(signals_df[["signal", "strength", "price"]].to_string())
    
    # Сканирование всех инструментов
    print("\n" + "-" * 60)
    print("СКАНИРОВАНИЕ ВСЕХ ИНСТРУМЕНТОВ (D1)")
    print("-" * 60)
    
    scan_results = generator.scan_all_instruments("D1", min_strength=3)
    
    if not scan_results.empty:
        print(f"\nНайдено {len(scan_results)} активных сигналов:")
        print(scan_results[["ticker", "signal", "strength", "price"]].to_string(index=False))
    else:
        print("\nАктивных сигналов не найдено")


if __name__ == "__main__":
    main()




