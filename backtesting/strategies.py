"""
Примеры торговых стратегий для бэктестинга.

Содержит готовые стратегии:
- SMA Crossover (пересечение скользящих средних)
- RSI Strategy (стратегия на RSI)
- Bollinger Bands (стратегия на полосах Боллинджера)
- MACD Strategy (стратегия на MACD)
"""

from typing import Dict, Any, List
from datetime import datetime

from strategy_base import Strategy, Signal


class SMACrossoverStrategy(Strategy):
    """
    Стратегия пересечения скользящих средних (SMA Crossover).
    
    Покупка: быстрая SMA пересекает медленную SMA снизу вверх
    Продажа: быстрая SMA пересекает медленную SMA сверху вниз
    
    Параметры:
        fast_period: Период быстрой SMA (по умолчанию 10)
        slow_period: Период медленной SMA (по умолчанию 30)
        stop_loss_pct: Стоп-лосс в процентах (по умолчанию 2%)
        take_profit_pct: Тейк-профит в процентах (по умолчанию 5%)
    """
    
    def init(self) -> None:
        """Инициализация индикаторов."""
        fast_period = self.params.get('fast_period', 10)
        slow_period = self.params.get('slow_period', 30)
        
        self.indicators['sma_fast'] = self.sma(fast_period)
        self.indicators['sma_slow'] = self.sma(slow_period)
    
    def next(self) -> Signal:
        """Логика стратегии."""
        i = self.current_bar
        
        # Проверяем достаточность данных
        if i < 1:
            return Signal('HOLD', 0, self.date[i])
        
        sma_fast = self.indicators['sma_fast'][i]
        sma_slow = self.indicators['sma_slow'][i]
        sma_fast_prev = self.indicators['sma_fast'][i-1]
        sma_slow_prev = self.indicators['sma_slow'][i-1]
        
        current_price = self.close[i]
        current_date = self.date[i]
        
        # Пересечение вверх (buy signal)
        if sma_fast > sma_slow and sma_fast_prev <= sma_slow_prev:
            if not self.has_position():
                stop_loss_pct = self.params.get('stop_loss_pct', 0.02)
                take_profit_pct = self.params.get('take_profit_pct', 0.05)
                
                stop_loss = current_price * (1 - stop_loss_pct)
                take_profit = current_price * (1 + take_profit_pct)
                
                return Signal(
                    'BUY',
                    current_price,
                    current_date,
                    confidence=1.0,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason=f"SMA {self.params.get('fast_period')} crossed above {self.params.get('slow_period')}"
                )
        
        # Пересечение вниз (sell signal)
        elif sma_fast < sma_slow and sma_fast_prev >= sma_slow_prev:
            if self.has_position():
                return Signal(
                    'SELL',
                    current_price,
                    current_date,
                    reason=f"SMA {self.params.get('fast_period')} crossed below {self.params.get('slow_period')}"
                )
        
        return Signal('HOLD', 0, current_date)
    
    @classmethod
    def get_param_space(cls) -> Dict[str, List[Any]]:
        """Пространство параметров для оптимизации."""
        return {
            'fast_period': [5, 10, 15, 20],
            'slow_period': [20, 30, 50, 100],
            'stop_loss_pct': [0.01, 0.02, 0.03, 0.05],
            'take_profit_pct': [0.03, 0.05, 0.10, 0.15]
        }


class RSIStrategy(Strategy):
    """
    Стратегия на основе индикатора RSI.
    
    Покупка: RSI < oversold_level (перепроданность)
    Продажа: RSI > overbought_level (перекупленность)
    
    Параметры:
        rsi_period: Период RSI (по умолчанию 14)
        oversold_level: Уровень перепроданности (по умолчанию 30)
        overbought_level: Уровень перекупленности (по умолчанию 70)
        stop_loss_pct: Стоп-лосс в процентах (по умолчанию 3%)
    """
    
    def init(self) -> None:
        """Инициализация индикаторов."""
        rsi_period = self.params.get('rsi_period', 14)
        self.indicators['rsi'] = self.rsi(rsi_period)
        
        # Добавим SMA для дополнительного фильтра
        self.indicators['sma'] = self.sma(50)
    
    def next(self) -> Signal:
        """Логика стратегии."""
        i = self.current_bar
        
        rsi = self.indicators['rsi'][i]
        sma = self.indicators['sma'][i]
        current_price = self.close[i]
        current_date = self.date[i]
        
        oversold = self.params.get('oversold_level', 30)
        overbought = self.params.get('overbought_level', 70)
        
        # Покупка при перепроданности (и цена выше SMA для фильтра тренда)
        if rsi < oversold and current_price > sma:
            if not self.has_position():
                stop_loss_pct = self.params.get('stop_loss_pct', 0.03)
                stop_loss = current_price * (1 - stop_loss_pct)
                
                # Тейк-профит на уровне overbought
                take_profit = current_price * 1.05  # +5%
                
                return Signal(
                    'BUY',
                    current_price,
                    current_date,
                    confidence=(oversold - rsi) / oversold,  # Чем ниже RSI, тем выше уверенность
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason=f"RSI oversold: {rsi:.1f} < {oversold}"
                )
        
        # Продажа при перекупленности
        elif rsi > overbought:
            if self.has_position():
                return Signal(
                    'SELL',
                    current_price,
                    current_date,
                    reason=f"RSI overbought: {rsi:.1f} > {overbought}"
                )
        
        return Signal('HOLD', 0, current_date)
    
    @classmethod
    def get_param_space(cls) -> Dict[str, List[Any]]:
        """Пространство параметров для оптимизации."""
        return {
            'rsi_period': [7, 14, 21, 28],
            'oversold_level': [20, 25, 30, 35],
            'overbought_level': [65, 70, 75, 80],
            'stop_loss_pct': [0.02, 0.03, 0.05, 0.07]
        }


class BollingerBandsStrategy(Strategy):
    """
    Стратегия на полосах Боллинджера.
    
    Покупка: цена касается нижней полосы
    Продажа: цена касается верхней полосы
    
    Параметры:
        bb_period: Период для Bollinger Bands (по умолчанию 20)
        bb_std: Количество стандартных отклонений (по умолчанию 2.0)
        stop_loss_pct: Стоп-лосс в процентах (по умолчанию 3%)
    """
    
    def init(self) -> None:
        """Инициализация индикаторов."""
        bb_period = self.params.get('bb_period', 20)
        bb_std = self.params.get('bb_std', 2.0)
        
        upper, middle, lower = self.bbands(bb_period, bb_std)
        self.indicators['bb_upper'] = upper
        self.indicators['bb_middle'] = middle
        self.indicators['bb_lower'] = lower
    
    def next(self) -> Signal:
        """Логика стратегии."""
        i = self.current_bar
        
        current_price = self.close[i]
        current_date = self.date[i]
        
        bb_lower = self.indicators['bb_lower'][i]
        bb_upper = self.indicators['bb_upper'][i]
        bb_middle = self.indicators['bb_middle'][i]
        
        # Покупка при касании нижней полосы
        if current_price <= bb_lower:
            if not self.has_position():
                stop_loss_pct = self.params.get('stop_loss_pct', 0.03)
                stop_loss = current_price * (1 - stop_loss_pct)
                
                # Тейк-профит на средней линии
                take_profit = bb_middle
                
                return Signal(
                    'BUY',
                    current_price,
                    current_date,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason=f"Price touched lower band: {current_price:.2f} <= {bb_lower:.2f}"
                )
        
        # Продажа при касании верхней полосы
        elif current_price >= bb_upper:
            if self.has_position():
                return Signal(
                    'SELL',
                    current_price,
                    current_date,
                    reason=f"Price touched upper band: {current_price:.2f} >= {bb_upper:.2f}"
                )
        
        return Signal('HOLD', 0, current_date)
    
    @classmethod
    def get_param_space(cls) -> Dict[str, List[Any]]:
        """Пространство параметров для оптимизации."""
        return {
            'bb_period': [10, 15, 20, 30],
            'bb_std': [1.5, 2.0, 2.5, 3.0],
            'stop_loss_pct': [0.02, 0.03, 0.05]
        }


class MACDStrategy(Strategy):
    """
    Стратегия на индикаторе MACD.
    
    Покупка: MACD пересекает сигнальную линию снизу вверх
    Продажа: MACD пересекает сигнальную линию сверху вниз
    
    Параметры:
        fast_period: Период быстрой EMA (по умолчанию 12)
        slow_period: Период медленной EMA (по умолчанию 26)
        signal_period: Период сигнальной линии (по умолчанию 9)
        stop_loss_pct: Стоп-лосс в процентах (по умолчанию 2%)
    """
    
    def init(self) -> None:
        """Инициализация индикаторов."""
        fast = self.params.get('fast_period', 12)
        slow = self.params.get('slow_period', 26)
        signal = self.params.get('signal_period', 9)
        
        macd, signal_line, histogram = self.macd(fast, slow, signal)
        self.indicators['macd'] = macd
        self.indicators['signal'] = signal_line
        self.indicators['histogram'] = histogram
    
    def next(self) -> Signal:
        """Логика стратегии."""
        i = self.current_bar
        
        if i < 1:
            return Signal('HOLD', 0, self.date[i])
        
        macd = self.indicators['macd'][i]
        signal_line = self.indicators['signal'][i]
        macd_prev = self.indicators['macd'][i-1]
        signal_prev = self.indicators['signal'][i-1]
        
        current_price = self.close[i]
        current_date = self.date[i]
        
        # Пересечение вверх (bullish)
        if macd > signal_line and macd_prev <= signal_prev:
            if not self.has_position():
                stop_loss_pct = self.params.get('stop_loss_pct', 0.02)
                atr_value = self.atr(14)[i]
                
                stop_loss = current_price - (atr_value * 2)  # 2 ATR stop
                take_profit = current_price + (atr_value * 3)  # 3 ATR target
                
                return Signal(
                    'BUY',
                    current_price,
                    current_date,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason="MACD crossed above signal line"
                )
        
        # Пересечение вниз (bearish)
        elif macd < signal_line and macd_prev >= signal_prev:
            if self.has_position():
                return Signal(
                    'SELL',
                    current_price,
                    current_date,
                    reason="MACD crossed below signal line"
                )
        
        return Signal('HOLD', 0, current_date)
    
    @classmethod
    def get_param_space(cls) -> Dict[str, List[Any]]:
        """Пространство параметров для оптимизации."""
        return {
            'fast_period': [8, 12, 16],
            'slow_period': [20, 26, 32],
            'signal_period': [7, 9, 11],
            'stop_loss_pct': [0.015, 0.02, 0.03]
        }
