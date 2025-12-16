"""
Базовый класс для торговых стратегий.

Все пользовательские стратегии должны наследоваться от Strategy
и реализовывать методы init() и next().
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import polars as pl


@dataclass
class Signal:
    """Торговый сигнал."""
    action: str  # 'BUY', 'SELL', 'HOLD'
    price: float
    timestamp: datetime
    confidence: float = 1.0  # Уверенность в сигнале (0-1)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    size: float = 1.0  # Размер позиции (доля от капитала)
    reason: str = ""  # Причина сигнала


@dataclass
class Position:
    """Открытая позиция."""
    ticker: str
    entry_price: float
    entry_time: datetime
    size: float  # Количество акций
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    current_price: float = 0.0
    
    @property
    def pnl(self) -> float:
        """Текущая прибыль/убыток."""
        return (self.current_price - self.entry_price) * self.size
    
    @property
    def pnl_percent(self) -> float:
        """Прибыль/убыток в процентах."""
        return ((self.current_price - self.entry_price) / self.entry_price) * 100
    
    def update_price(self, price: float) -> None:
        """Обновить текущую цену."""
        self.current_price = price
    
    def should_close(self, current_price: float) -> tuple[bool, str]:
        """
        Проверить, нужно ли закрыть позицию.
        
        Returns:
            (should_close, reason)
        """
        if self.stop_loss and current_price <= self.stop_loss:
            return True, "stop_loss"
        
        if self.take_profit and current_price >= self.take_profit:
            return True, "take_profit"
        
        return False, ""


class Strategy(ABC):
    """
    Базовый класс для всех торговых стратегий.
    
    Attributes:
        data: DataFrame с историческими данными
        params: Словарь с параметрами стратегии
        indicators: Рассчитанные индикаторы
        current_bar: Индекс текущего бара
    """
    
    def __init__(self, data: pl.DataFrame, params: Dict[str, Any]):
        """
        Инициализация стратегии.
        
        Args:
            data: DataFrame с колонками [date, open, high, low, close, volume]
            params: Параметры стратегии
        """
        self.data = data
        self.params = params
        self.indicators: Dict[str, pl.Series] = {}
        self.current_bar: int = 0
        self._position: Optional[Position] = None
        
        # Валидация данных
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Отсутствует обязательная колонка: {col}")
        
        # Инициализация индикаторов
        self.init()
    
    @abstractmethod
    def init(self) -> None:
        """
        Инициализация индикаторов и переменных стратегии.
        Вызывается один раз перед началом бэктестинга.
        
        Пример:
            self.indicators['sma'] = self.sma(period=20)
            self.indicators['rsi'] = self.rsi(period=14)
        """
        pass
    
    @abstractmethod
    def next(self) -> Signal:
        """
        Логика стратегии для текущего бара.
        Вызывается на каждом баре данных.
        
        Returns:
            Signal: Торговый сигнал (BUY, SELL, HOLD)
            
        Пример:
            if self.indicators['rsi'][self.current_bar] < 30:
                return Signal('BUY', self.close[self.current_bar], ...)
            return Signal('HOLD', 0, ...)
        """
        pass
    
    # === Удобные геттеры для доступа к данным ===
    
    @property
    def open(self) -> pl.Series:
        """Цены открытия."""
        return self.data['open']
    
    @property
    def high(self) -> pl.Series:
        """Максимальные цены."""
        return self.data['high']
    
    @property
    def low(self) -> pl.Series:
        """Минимальные цены."""
        return self.data['low']
    
    @property
    def close(self) -> pl.Series:
        """Цены закрытия."""
        return self.data['close']
    
    @property
    def volume(self) -> pl.Series:
        """Объёмы."""
        return self.data['volume']
    
    @property
    def date(self) -> pl.Series:
        """Даты."""
        return self.data['date']
    
    # === Вспомогательные методы для индикаторов ===
    
    def sma(self, period: int, source: Optional[pl.Series] = None) -> pl.Series:
        """
        Простая скользящая средняя (SMA).
        
        Args:
            period: Период усреднения
            source: Источник данных (по умолчанию close)
            
        Returns:
            Series с значениями SMA
        """
        if source is None:
            source = self.close
        return source.rolling_mean(window_size=period)
    
    def ema(self, period: int, source: Optional[pl.Series] = None) -> pl.Series:
        """
        Экспоненциальная скользящая средняя (EMA).
        
        Args:
            period: Период усреднения
            source: Источник данных (по умолчанию close)
            
        Returns:
            Series с значениями EMA
        """
        if source is None:
            source = self.close
        return source.ewm_mean(span=period, adjust=False)
    
    def rsi(self, period: int = 14, source: Optional[pl.Series] = None) -> pl.Series:
        """
        Индекс относительной силы (RSI).
        
        Args:
            period: Период расчёта
            source: Источник данных (по умолчанию close)
            
        Returns:
            Series с значениями RSI (0-100)
        """
        if source is None:
            source = self.close
        
        # Вычисляем изменения
        delta = source.diff()
        
        # Разделяем на рост и падение
        gain = delta.clip_min(0)
        loss = (-delta).clip_min(0)
        
        # Экспоненциальное скользящее среднее
        avg_gain = gain.ewm_mean(span=period, adjust=False)
        avg_loss = loss.ewm_mean(span=period, adjust=False)
        
        # Избегаем деления на ноль
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def bbands(self, period: int = 20, std_dev: float = 2.0, 
               source: Optional[pl.Series] = None) -> tuple[pl.Series, pl.Series, pl.Series]:
        """
        Полосы Боллинджера (Bollinger Bands).
        
        Args:
            period: Период расчёта
            std_dev: Количество стандартных отклонений
            source: Источник данных (по умолчанию close)
            
        Returns:
            (upper, middle, lower): Верхняя полоса, средняя линия, нижняя полоса
        """
        if source is None:
            source = self.close
        
        middle = source.rolling_mean(window_size=period)
        std = source.rolling_std(window_size=period)
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9,
             source: Optional[pl.Series] = None) -> tuple[pl.Series, pl.Series, pl.Series]:
        """
        MACD (Moving Average Convergence Divergence).
        
        Args:
            fast: Период быстрой EMA
            slow: Период медленной EMA
            signal: Период сигнальной линии
            source: Источник данных (по умолчанию close)
            
        Returns:
            (macd, signal_line, histogram)
        """
        if source is None:
            source = self.close
        
        fast_ema = self.ema(fast, source)
        slow_ema = self.ema(slow, source)
        
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm_mean(span=signal, adjust=False)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def atr(self, period: int = 14) -> pl.Series:
        """
        Average True Range (ATR) - средний истинный диапазон.
        
        Args:
            period: Период расчёта
            
        Returns:
            Series с значениями ATR
        """
        high_low = self.high - self.low
        high_close = (self.high - self.close.shift(1)).abs()
        low_close = (self.low - self.close.shift(1)).abs()
        
        true_range = pl.concat([
            high_low,
            high_close,
            low_close
        ], how="horizontal").max(axis=1)
        
        atr = true_range.ewm_mean(span=period, adjust=False)
        return atr
    
    def crossover(self, series1: pl.Series, series2: pl.Series) -> pl.Series:
        """
        Проверка пересечения series1 над series2.
        
        Args:
            series1: Первая серия (например, быстрая MA)
            series2: Вторая серия (например, медленная MA)
            
        Returns:
            Boolean Series: True где произошло пересечение вверх
        """
        return (series1 > series2) & (series1.shift(1) <= series2.shift(1))
    
    def crossunder(self, series1: pl.Series, series2: pl.Series) -> pl.Series:
        """
        Проверка пересечения series1 под series2.
        
        Args:
            series1: Первая серия
            series2: Вторая серия
            
        Returns:
            Boolean Series: True где произошло пересечение вниз
        """
        return (series1 < series2) & (series1.shift(1) >= series2.shift(1))
    
    # === Управление позицией ===
    
    def has_position(self) -> bool:
        """Проверка наличия открытой позиции."""
        return self._position is not None
    
    def get_position(self) -> Optional[Position]:
        """Получить текущую позицию."""
        return self._position
    
    def set_position(self, position: Optional[Position]) -> None:
        """Установить позицию (используется движком бэктестинга)."""
        self._position = position
    
    # === Информация о стратегии ===
    
    @classmethod
    def get_param_space(cls) -> Dict[str, List[Any]]:
        """
        Возвращает пространство параметров для оптимизации.
        Должен быть переопределён в наследниках.
        
        Returns:
            Словарь с параметрами и их возможными значениями
            
        Пример:
            return {
                'period': [10, 20, 30, 50],
                'threshold': [0.01, 0.02, 0.05],
                'stop_loss': [0.02, 0.03, 0.05]
            }
        """
        return {}
    
    def __repr__(self) -> str:
        """Строковое представление стратегии."""
        return f"{self.__class__.__name__}(params={self.params})"
