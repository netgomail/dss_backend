"""
Пример использования системы бэктестинга.

Демонстрирует:
- Простой бэктестинг стратегии
- Оптимизацию параметров (Grid Search)
- Walk-Forward Analysis
- Визуализацию результатов
"""

import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))

from backtest_engine import Backtest, load_data
from optimizer import Optimizer
from strategies import SMACrossoverStrategy, RSIStrategy, BollingerBandsStrategy, MACDStrategy


def example_simple_backtest():
    """Пример 1: Простой бэктестинг."""
    print("\n" + "="*70)
    print("ПРИМЕР 1: ПРОСТОЙ БЭКТЕСТИНГ")
    print("="*70 + "\n")
    
    # Загрузка данных
    data = load_data('AFKS', '1H')
    print(f"Загружено {len(data)} баров данных")
    
    # Параметры стратегии
    strategy_params = {
        'fast_period': 10,
        'slow_period': 30,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.05
    }
    
    # Создание и запуск бэктестинга
    backtest = Backtest(
        data=data,
        strategy_class=SMACrossoverStrategy,
        strategy_params=strategy_params,
        initial_capital=100000.0,
        commission=0.001,  # 0.1%
    )
    
    result = backtest.run()
    
    # Вывод результатов
    result.print_summary()
    
    # Детализация сделок
    print("\nПоследние 5 сделок:")
    for trade in result.trades[-5:]:
        print(f"  {trade.entry_time.date()} → {trade.exit_time.date()}: "
              f"P&L={trade.pnl:+.2f} ({trade.pnl_percent:+.1f}%) | {trade.exit_reason}")


def example_grid_search():
    """Пример 2: Оптимизация с Grid Search."""
    print("\n" + "="*70)
    print("ПРИМЕР 2: ОПТИМИЗАЦИЯ (GRID SEARCH)")
    print("="*70 + "\n")
    
    # Загрузка данных
    data = load_data('AFKS', '1H')
    
    # Пространство параметров для оптимизации
    param_space = {
        'fast_period': [5, 10, 15],
        'slow_period': [20, 30, 50],
        'stop_loss_pct': [0.02, 0.03],
        'take_profit_pct': [0.05, 0.10]
    }
    
    # Создание оптимизатора
    optimizer = Optimizer(
        data=data,
        strategy_class=SMACrossoverStrategy,
        param_space=param_space,
        optimization_metric='sharpe_ratio',  # Оптимизируем по Sharpe Ratio
        initial_capital=100000.0,
        commission=0.001,
        n_jobs=1  # Используйте >1 для параллелизма
    )
    
    # Запуск оптимизации
    best_result = optimizer.grid_search(verbose=True)
    
    # Вывод результатов
    print("\nДетали лучшего результата:")
    best_result.backtest_result.print_summary()
    
    # Топ-5 комбинаций
    print("\nТоп-5 комбинаций параметров:")
    for i, result in enumerate(optimizer.get_top_results(5), 1):
        print(f"{i}. Params: {result.params}")
        print(f"   Sharpe: {result.metric_value:.2f}, "
              f"Return: {result.backtest_result.total_return_percent:.1f}%, "
              f"Trades: {result.backtest_result.total_trades}")
    
    # Сохранение результатов
    optimizer.save_results(Path('optimization_results.json'))
    print("\nРезультаты сохранены в optimization_results.json")


def example_random_search():
    """Пример 3: Random Search оптимизация."""
    print("\n" + "="*70)
    print("ПРИМЕР 3: RANDOM SEARCH ОПТИМИЗАЦИЯ")
    print("="*70 + "\n")
    
    # Загрузка данных
    data = load_data('AFKS', '1D')  # Дневные данные
    
    # Пространство параметров
    param_space = {
        'rsi_period': [7, 10, 14, 21, 28],
        'oversold_level': [20, 25, 30, 35, 40],
        'overbought_level': [60, 65, 70, 75, 80],
        'stop_loss_pct': [0.02, 0.03, 0.05, 0.07, 0.10]
    }
    
    # Создание оптимизатора
    optimizer = Optimizer(
        data=data,
        strategy_class=RSIStrategy,
        param_space=param_space,
        optimization_metric='risk_adjusted_return',  # Return / Drawdown
        initial_capital=100000.0,
        commission=0.001
    )
    
    # Запуск Random Search (100 случайных комбинаций)
    best_result = optimizer.random_search(n_iterations=100, verbose=True)
    
    # Вывод результатов
    print("\nДетали лучшего результата:")
    best_result.backtest_result.print_summary()


def example_walk_forward():
    """Пример 4: Walk-Forward Analysis."""
    print("\n" + "="*70)
    print("ПРИМЕР 4: WALK-FORWARD ANALYSIS")
    print("="*70 + "\n")
    
    # Загрузка данных
    data = load_data('AFKS', '1H')
    
    # Пространство параметров (упрощённое для скорости)
    param_space = {
        'bb_period': [15, 20, 25],
        'bb_std': [1.5, 2.0, 2.5],
        'stop_loss_pct': [0.02, 0.03]
    }
    
    # Создание оптимизатора
    optimizer = Optimizer(
        data=data,
        strategy_class=BollingerBandsStrategy,
        param_space=param_space,
        optimization_metric='sharpe_ratio',
        initial_capital=100000.0,
        commission=0.001
    )
    
    # Запуск Walk-Forward Analysis
    wfa_results = optimizer.walk_forward_analysis(
        train_size=500,   # 500 баров на обучение
        test_size=100,    # 100 баров на тест
        step_size=50,     # Сдвиг на 50 баров
        optimization_method='grid',
        verbose=True
    )
    
    print("\nСводка по фолдам:")
    for fold in wfa_results['folds']:
        print(f"\nФолд {fold['fold']}:")
        print(f"  Параметры: {fold['best_params']}")
        print(f"  Train metric: {fold['train_metric']:.2f}")
        print(f"  Test metric: {fold['test_metric']:.2f}")
        print(f"  Test return: {fold['test_result'].total_return_percent:.2f}%")


def example_multiple_strategies():
    """Пример 5: Сравнение нескольких стратегий."""
    print("\n" + "="*70)
    print("ПРИМЕР 5: СРАВНЕНИЕ СТРАТЕГИЙ")
    print("="*70 + "\n")
    
    # Загрузка данных
    data = load_data('AFKS', '1H')
    
    # Список стратегий для сравнения
    strategies = [
        (SMACrossoverStrategy, {'fast_period': 10, 'slow_period': 30, 'stop_loss_pct': 0.02, 'take_profit_pct': 0.05}),
        (RSIStrategy, {'rsi_period': 14, 'oversold_level': 30, 'overbought_level': 70, 'stop_loss_pct': 0.03}),
        (BollingerBandsStrategy, {'bb_period': 20, 'bb_std': 2.0, 'stop_loss_pct': 0.03}),
        (MACDStrategy, {'fast_period': 12, 'slow_period': 26, 'signal_period': 9, 'stop_loss_pct': 0.02}),
    ]
    
    results = []
    
    for strategy_class, params in strategies:
        print(f"\nТестирование: {strategy_class.__name__}")
        
        backtest = Backtest(
            data=data,
            strategy_class=strategy_class,
            strategy_params=params,
            initial_capital=100000.0,
            commission=0.001
        )
        
        result = backtest.run()
        results.append((strategy_class.__name__, result))
        
        print(f"  Return: {result.total_return_percent:+.2f}%")
        print(f"  Trades: {result.total_trades}")
        print(f"  Win Rate: {result.win_rate*100:.1f}%")
        print(f"  Sharpe: {result.sharpe_ratio:.2f}")
        print(f"  Max DD: {result.max_drawdown_percent:.2f}%")
    
    # Сводная таблица
    print("\n" + "="*70)
    print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
    print("="*70)
    print(f"{'Стратегия':<30} {'Return %':>10} {'Sharpe':>8} {'Max DD %':>10} {'Trades':>8}")
    print("-"*70)
    
    for name, result in results:
        print(f"{name:<30} {result.total_return_percent:>9.2f}% {result.sharpe_ratio:>8.2f} "
              f"{result.max_drawdown_percent:>9.2f}% {result.total_trades:>8}")


def main():
    """Главная функция."""
    print("\n" + "="*70)
    print(" СИСТЕМА ГЛУБОКОГО БЭКТЕСТИНГА С ОПТИМИЗАЦИЕЙ ПАРАМЕТРОВ")
    print("="*70)
    
    # Запуск примеров
    try:
        # Пример 1: Простой бэктестинг
        example_simple_backtest()
        
        # Пример 2: Grid Search
        # example_grid_search()  # Раскомментируйте для запуска
        
        # Пример 3: Random Search
        # example_random_search()  # Раскомментируйте для запуска
        
        # Пример 4: Walk-Forward Analysis
        # example_walk_forward()  # Раскомментируйте для запуска
        
        # Пример 5: Сравнение стратегий
        example_multiple_strategies()
        
    except FileNotFoundError as e:
        print(f"\nОшибка: {e}")
        print("\nУбедитесь, что:")
        print("1. Данные находятся в data/tickers/AFKS/")
        print("2. Запустили download_archives.py и process_archives.py")
    except Exception as e:
        print(f"\nНеожиданная ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
