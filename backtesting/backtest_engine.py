"""
Движок бэктестинга для симуляции торговли на исторических данных.

Основной класс Backtest отвечает за:
- Загрузку исторических данных
- Симуляцию исполнения сделок
- Учёт комиссий и проскальзывания
- Управление капиталом
- Сбор статистики
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Type
from dataclasses import dataclass, field
from datetime import datetime

import polars as pl

from strategy_base import Strategy, Signal, Position

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Завершённая сделка."""
    ticker: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_percent: float
    commission: float
    exit_reason: str  # 'signal', 'stop_loss', 'take_profit'
    
    @property
    def duration(self) -> float:
        """Длительность сделки в днях."""
        return (self.exit_time - self.entry_time).total_seconds() / 86400
    
    @property
    def is_winning(self) -> bool:
        """Прибыльная ли сделка."""
        return self.pnl > 0


@dataclass
class BacktestResult:
    """Результаты бэктестинга."""
    # Основные метрики
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_percent: float
    
    # Сделки
    trades: List[Trade] = field(default_factory=list)
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # Прибыль/убыток
    total_pnl: float = 0.0
    total_commission: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Метрики риска
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    
    # Временные метрики
    avg_trade_duration: float = 0.0  # в днях
    total_days: int = 0
    exposure_time: float = 0.0  # процент времени в позиции
    
    # История капитала
    equity_curve: List[tuple[datetime, float]] = field(default_factory=list)
    
    def calculate_metrics(self) -> None:
        """Рассчитать все метрики на основе сделок."""
        if not self.trades:
            return
        
        self.total_trades = len(self.trades)
        self.winning_trades = sum(1 for t in self.trades if t.is_winning)
        self.losing_trades = self.total_trades - self.winning_trades
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        wins = [t.pnl for t in self.trades if t.is_winning]
        losses = [t.pnl for t in self.trades if not t.is_winning]
        
        self.total_pnl = sum(t.pnl for t in self.trades)
        self.total_commission = sum(t.commission for t in self.trades)
        
        self.avg_win = sum(wins) / len(wins) if wins else 0
        self.avg_loss = sum(losses) / len(losses) if losses else 0
        self.largest_win = max(wins) if wins else 0
        self.largest_loss = min(losses) if losses else 0
        
        # Profit factor
        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        self.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Средняя длительность сделки
        self.avg_trade_duration = sum(t.duration for t in self.trades) / len(self.trades)
        
        # Максимальная просадка
        self._calculate_drawdown()
        
        # Sharpe и Sortino
        self._calculate_risk_metrics()
    
    def _calculate_drawdown(self) -> None:
        """Рассчитать максимальную просадку."""
        if not self.equity_curve:
            return
        
        peak = self.initial_capital
        max_dd = 0
        
        for _, equity in self.equity_curve:
            if equity > peak:
                peak = equity
            drawdown = peak - equity
            if drawdown > max_dd:
                max_dd = drawdown
        
        self.max_drawdown = max_dd
        self.max_drawdown_percent = (max_dd / peak * 100) if peak > 0 else 0
    
    def _calculate_risk_metrics(self) -> None:
        """Рассчитать Sharpe и Sortino ratios."""
        if len(self.equity_curve) < 2:
            return
        
        # Дневные доходности
        returns = []
        for i in range(1, len(self.equity_curve)):
            prev_equity = self.equity_curve[i-1][1]
            curr_equity = self.equity_curve[i][1]
            ret = (curr_equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            returns.append(ret)
        
        if not returns:
            return
        
        import statistics
        
        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns) if len(returns) > 1 else 0
        
        # Sharpe ratio (предполагаем безрисковую ставку = 0)
        self.sharpe_ratio = (mean_return / std_return * (252 ** 0.5)) if std_return > 0 else 0
        
        # Sortino ratio (только отрицательные отклонения)
        negative_returns = [r for r in returns if r < 0]
        if negative_returns:
            downside_std = statistics.stdev(negative_returns)
            self.sortino_ratio = (mean_return / downside_std * (252 ** 0.5)) if downside_std > 0 else 0
    
    def print_summary(self) -> None:
        """Вывести краткую сводку результатов."""
        print("=" * 70)
        print("РЕЗУЛЬТАТЫ БЭКТЕСТИНГА")
        print("=" * 70)
        print(f"Начальный капитал:    {self.initial_capital:,.2f}")
        print(f"Конечный капитал:     {self.final_capital:,.2f}")
        print(f"Общая доходность:     {self.total_return:,.2f} ({self.total_return_percent:.2f}%)")
        print(f"Макс. просадка:       {self.max_drawdown:,.2f} ({self.max_drawdown_percent:.2f}%)")
        print()
        print(f"Всего сделок:         {self.total_trades}")
        print(f"Прибыльных:           {self.winning_trades} ({self.win_rate*100:.1f}%)")
        print(f"Убыточных:            {self.losing_trades}")
        print()
        print(f"Средняя прибыль:      {self.avg_win:,.2f}")
        print(f"Средний убыток:       {self.avg_loss:,.2f}")
        print(f"Profit Factor:        {self.profit_factor:.2f}")
        print()
        print(f"Sharpe Ratio:         {self.sharpe_ratio:.2f}")
        print(f"Sortino Ratio:        {self.sortino_ratio:.2f}")
        print(f"Комиссии:             {self.total_commission:,.2f}")
        print("=" * 70)


class Backtest:
    """
    Движок бэктестинга для симуляции торговли.
    
    Attributes:
        data: Исторические данные
        strategy_class: Класс стратегии
        initial_capital: Начальный капитал
        commission: Комиссия за сделку (доля от объёма)
        slippage: Проскальзывание (доля от цены)
    """
    
    def __init__(
        self,
        data: pl.DataFrame,
        strategy_class: Type[Strategy],
        strategy_params: Dict[str, Any],
        initial_capital: float = 100000.0,
        commission: float = 0.001,  # 0.1%
        slippage: float = 0.0005,  # 0.05%
        position_size: float = 1.0,  # Доля капитала на сделку
    ):
        """
        Инициализация бэктестинга.
        
        Args:
            data: DataFrame с историческими данными
            strategy_class: Класс стратегии (не экземпляр!)
            strategy_params: Параметры стратегии
            initial_capital: Начальный капитал
            commission: Комиссия (0.001 = 0.1%)
            slippage: Проскальзывание (0.0005 = 0.05%)
            position_size: Доля капитала на сделку (1.0 = 100%)
        """
        self.data = data.sort('date')
        self.strategy_class = strategy_class
        self.strategy_params = strategy_params
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size = position_size
        
        # Состояние бэктестинга
        self.cash = initial_capital
        self.equity = initial_capital
        self.position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.equity_curve: List[tuple[datetime, float]] = []
        
        # Стратегия
        self.strategy: Optional[Strategy] = None
    
    def run(self) -> BacktestResult:
        """
        Запустить бэктестинг.
        
        Returns:
            BacktestResult с результатами
        """
        logger.info(f"Запуск бэктестинга: {self.strategy_class.__name__}")
        logger.info(f"Параметры: {self.strategy_params}")
        logger.info(f"Данных: {len(self.data)} баров")
        
        # Инициализируем стратегию
        self.strategy = self.strategy_class(self.data, self.strategy_params)
        
        # Проходим по каждому бару
        for i in range(len(self.data)):
            self.strategy.current_bar = i
            
            # Пропускаем первые бары (нужны для индикаторов)
            if i < 50:  # Минимальный прогрев для индикаторов
                continue
            
            current_price = self.data['close'][i]
            current_date = self.data['date'][i]
            
            # Обновляем позицию
            if self.position:
                self.position.update_price(current_price)
                
                # Проверяем стоп-лосс и тейк-профит
                should_close, reason = self.position.should_close(current_price)
                if should_close:
                    self._close_position(i, reason)
            
            # Получаем сигнал от стратегии
            signal = self.strategy.next()
            
            # Исполняем сигнал
            if signal.action == 'BUY' and not self.position:
                self._open_position(i, signal)
            elif signal.action == 'SELL' and self.position:
                self._close_position(i, 'signal')
            
            # Обновляем equity
            self._update_equity(i)
        
        # Закрываем открытую позицию в конце
        if self.position:
            self._close_position(len(self.data) - 1, 'end')
        
        # Создаём результат
        result = self._create_result()
        
        logger.info(f"Бэктестинг завершён. Сделок: {len(self.trades)}")
        
        return result
    
    def _open_position(self, bar_index: int, signal: Signal) -> None:
        """Открыть позицию."""
        current_date = self.data['date'][bar_index]
        
        # Цена с учётом проскальзывания
        entry_price = signal.price * (1 + self.slippage)
        
        # Размер позиции
        position_value = self.cash * self.position_size * signal.size
        size = position_value / entry_price
        
        # Комиссия
        commission_cost = position_value * self.commission
        
        # Проверяем достаточность средств
        total_cost = position_value + commission_cost
        if total_cost > self.cash:
            logger.debug(f"Недостаточно средств для открытия позиции. Нужно: {total_cost:.2f}, Доступно: {self.cash:.2f}")
            return
        
        # Открываем позицию
        self.position = Position(
            ticker="",  # Заполнится из метаданных
            entry_price=entry_price,
            entry_time=current_date,
            size=size,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            current_price=entry_price
        )
        
        self.cash -= total_cost
        self.strategy.set_position(self.position)
        
        logger.debug(f"Открыта позиция: цена={entry_price:.2f}, размер={size:.2f}, комиссия={commission_cost:.2f}")
    
    def _close_position(self, bar_index: int, reason: str) -> None:
        """Закрыть позицию."""
        if not self.position:
            return
        
        current_date = self.data['date'][bar_index]
        current_price = self.data['close'][bar_index]
        
        # Цена с учётом проскальзывания
        exit_price = current_price * (1 - self.slippage)
        
        # Расчёт P&L
        position_value = self.position.size * exit_price
        pnl = (exit_price - self.position.entry_price) * self.position.size
        pnl_percent = ((exit_price - self.position.entry_price) / self.position.entry_price) * 100
        
        # Комиссия
        commission_cost = position_value * self.commission
        
        # Обновляем cash
        self.cash += position_value - commission_cost
        
        # Сохраняем сделку
        trade = Trade(
            ticker=self.position.ticker,
            entry_time=self.position.entry_time,
            exit_time=current_date,
            entry_price=self.position.entry_price,
            exit_price=exit_price,
            size=self.position.size,
            pnl=pnl - commission_cost,
            pnl_percent=pnl_percent,
            commission=commission_cost,
            exit_reason=reason
        )
        self.trades.append(trade)
        
        logger.debug(f"Закрыта позиция: причина={reason}, P&L={pnl:.2f} ({pnl_percent:.2f}%)")
        
        # Очищаем позицию
        self.position = None
        self.strategy.set_position(None)
    
    def _update_equity(self, bar_index: int) -> None:
        """Обновить equity (капитал)."""
        current_price = self.data['close'][bar_index]
        current_date = self.data['date'][bar_index]
        
        # Equity = cash + стоимость позиции
        position_value = 0
        if self.position:
            position_value = self.position.size * current_price
        
        self.equity = self.cash + position_value
        self.equity_curve.append((current_date, self.equity))
    
    def _create_result(self) -> BacktestResult:
        """Создать объект с результатами."""
        result = BacktestResult(
            initial_capital=self.initial_capital,
            final_capital=self.equity,
            total_return=self.equity - self.initial_capital,
            total_return_percent=((self.equity - self.initial_capital) / self.initial_capital) * 100,
            trades=self.trades,
            equity_curve=self.equity_curve
        )
        
        result.calculate_metrics()
        
        return result


def load_data(ticker: str, timeframe: str, data_dir: Path = None) -> pl.DataFrame:
    """
    Загрузить исторические данные для бэктестинга.
    
    Args:
        ticker: Тикер инструмента
        timeframe: Таймфрейм (5M, 15M, 1H, 1D и т.д.)
        data_dir: Директория с данными (по умолчанию data/tickers)
        
    Returns:
        DataFrame с данными
    """
    if data_dir is None:
        data_dir = Path.cwd() / "data" / "tickers"
    
    file_path = data_dir / ticker / f"{timeframe}.parquet"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    
    df = pl.read_parquet(file_path)
    
    logger.info(f"Загружено {len(df)} баров для {ticker} ({timeframe})")
    
    return df
