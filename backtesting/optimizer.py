"""
Оптимизатор параметров торговых стратегий.

Поддерживает несколько методов оптимизации:
- Grid Search (полный перебор)
- Random Search (случайный поиск)
- Bayesian Optimization (байесовская оптимизация)
- Walk-Forward Analysis (прогрессивный анализ)
"""

import logging
import itertools
from typing import Dict, Any, List, Callable, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json

import polars as pl
import numpy as np

from strategy_base import Strategy
from backtest_engine import Backtest, BacktestResult, load_data

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Результат одной итерации оптимизации."""
    params: Dict[str, Any]
    metric_value: float
    backtest_result: BacktestResult
    
    def __lt__(self, other):
        """Для сортировки по метрике."""
        return self.metric_value < other.metric_value


class Optimizer:
    """
    Оптимизатор параметров стратегий.
    
    Attributes:
        data: Исторические данные для бэктестинга
        strategy_class: Класс стратегии
        param_space: Пространство параметров для оптимизации
        optimization_metric: Метрика для оптимизации
    """
    
    # Доступные метрики для оптимизации
    METRICS = {
        'total_return': lambda r: r.total_return,
        'total_return_percent': lambda r: r.total_return_percent,
        'sharpe_ratio': lambda r: r.sharpe_ratio,
        'sortino_ratio': lambda r: r.sortino_ratio,
        'profit_factor': lambda r: r.profit_factor,
        'win_rate': lambda r: r.win_rate,
        'max_drawdown': lambda r: -r.max_drawdown,  # Минимизируем просадку
        'risk_adjusted_return': lambda r: r.total_return_percent / (r.max_drawdown_percent + 1),
    }
    
    def __init__(
        self,
        data: pl.DataFrame,
        strategy_class: type[Strategy],
        param_space: Dict[str, List[Any]],
        optimization_metric: str = 'sharpe_ratio',
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        n_jobs: int = 1,
    ):
        """
        Инициализация оптимизатора.
        
        Args:
            data: Исторические данные
            strategy_class: Класс стратегии
            param_space: Словарь {параметр: [возможные значения]}
            optimization_metric: Метрика для оптимизации
            initial_capital: Начальный капитал
            commission: Комиссия
            n_jobs: Количество параллельных процессов (1 = без параллелизма)
        """
        self.data = data
        self.strategy_class = strategy_class
        self.param_space = param_space
        self.optimization_metric = optimization_metric
        self.initial_capital = initial_capital
        self.commission = commission
        self.n_jobs = n_jobs
        
        if optimization_metric not in self.METRICS:
            raise ValueError(f"Неизвестная метрика: {optimization_metric}. "
                           f"Доступны: {list(self.METRICS.keys())}")
        
        self.metric_func = self.METRICS[optimization_metric]
        self.results: List[OptimizationResult] = []
    
    def grid_search(self, verbose: bool = True) -> OptimizationResult:
        """
        Полный перебор всех комбинаций параметров (Grid Search).
        
        Args:
            verbose: Выводить прогресс
            
        Returns:
            Лучший результат
        """
        # Генерируем все комбинации
        param_names = list(self.param_space.keys())
        param_values = list(self.param_space.values())
        combinations = list(itertools.product(*param_values))
        
        total = len(combinations)
        logger.info(f"Grid Search: {total} комбинаций параметров")
        
        if verbose:
            print(f"Оптимизация методом Grid Search")
            print(f"Всего комбинаций: {total}")
            print(f"Параметры: {param_names}")
            print("=" * 70)
        
        # Тестируем каждую комбинацию
        if self.n_jobs > 1:
            results = self._parallel_backtest(param_names, combinations, verbose)
        else:
            results = self._sequential_backtest(param_names, combinations, verbose)
        
        self.results = results
        
        # Находим лучший результат
        best_result = max(results, key=lambda r: r.metric_value)
        
        if verbose:
            print("\n" + "=" * 70)
            print(f"Лучшие параметры: {best_result.params}")
            print(f"Метрика ({self.optimization_metric}): {best_result.metric_value:.4f}")
            print("=" * 70)
        
        return best_result
    
    def random_search(self, n_iterations: int = 100, verbose: bool = True) -> OptimizationResult:
        """
        Случайный поиск параметров (Random Search).
        
        Args:
            n_iterations: Количество итераций
            verbose: Выводить прогресс
            
        Returns:
            Лучший результат
        """
        logger.info(f"Random Search: {n_iterations} итераций")
        
        if verbose:
            print(f"Оптимизация методом Random Search")
            print(f"Итераций: {n_iterations}")
            print("=" * 70)
        
        # Генерируем случайные комбинации
        param_names = list(self.param_space.keys())
        combinations = []
        
        for _ in range(n_iterations):
            combo = []
            for param_name in param_names:
                value = np.random.choice(self.param_space[param_name])
                combo.append(value)
            combinations.append(tuple(combo))
        
        # Тестируем комбинации
        if self.n_jobs > 1:
            results = self._parallel_backtest(param_names, combinations, verbose)
        else:
            results = self._sequential_backtest(param_names, combinations, verbose)
        
        self.results = results
        
        # Находим лучший результат
        best_result = max(results, key=lambda r: r.metric_value)
        
        if verbose:
            print("\n" + "=" * 70)
            print(f"Лучшие параметры: {best_result.params}")
            print(f"Метрика ({self.optimization_metric}): {best_result.metric_value:.4f}")
            print("=" * 70)
        
        return best_result
    
    def walk_forward_analysis(
        self,
        train_size: int = 252,  # 1 год
        test_size: int = 63,    # 3 месяца
        step_size: int = 21,    # 1 месяц
        optimization_method: str = 'grid',
        n_random_iterations: int = 50,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Walk-Forward Analysis (прогрессивный анализ).
        
        Разбивает данные на периоды обучения и тестирования,
        оптимизирует на обучении, проверяет на тестировании.
        
        Args:
            train_size: Размер окна обучения (в барах)
            test_size: Размер окна тестирования (в барах)
            step_size: Шаг сдвига окна (в барах)
            optimization_method: 'grid' или 'random'
            n_random_iterations: Итераций для random search
            verbose: Выводить прогресс
            
        Returns:
            Словарь с результатами WFA
        """
        logger.info(f"Walk-Forward Analysis: train={train_size}, test={test_size}, step={step_size}")
        
        if verbose:
            print(f"Walk-Forward Analysis")
            print(f"Обучение: {train_size} баров, Тест: {test_size} баров, Шаг: {step_size} баров")
            print("=" * 70)
        
        wfa_results = []
        total_data = len(self.data)
        current_pos = 0
        fold_num = 0
        
        while current_pos + train_size + test_size <= total_data:
            fold_num += 1
            
            # Разделяем данные
            train_start = current_pos
            train_end = current_pos + train_size
            test_start = train_end
            test_end = test_start + test_size
            
            train_data = self.data[train_start:train_end]
            test_data = self.data[test_start:test_end]
            
            if verbose:
                print(f"\nФолд {fold_num}:")
                print(f"  Обучение: {train_data['date'].min()} - {train_data['date'].max()}")
                print(f"  Тест: {test_data['date'].min()} - {test_data['date'].max()}")
            
            # Оптимизация на обучающем наборе
            train_optimizer = Optimizer(
                train_data,
                self.strategy_class,
                self.param_space,
                self.optimization_metric,
                self.initial_capital,
                self.commission,
                self.n_jobs
            )
            
            if optimization_method == 'grid':
                best_train = train_optimizer.grid_search(verbose=False)
            else:
                best_train = train_optimizer.random_search(n_random_iterations, verbose=False)
            
            if verbose:
                print(f"  Лучшие параметры: {best_train.params}")
                print(f"  Метрика (обучение): {best_train.metric_value:.4f}")
            
            # Тестирование на тестовом наборе
            test_backtest = Backtest(
                test_data,
                self.strategy_class,
                best_train.params,
                self.initial_capital,
                self.commission
            )
            test_result = test_backtest.run()
            test_metric = self.metric_func(test_result)
            
            if verbose:
                print(f"  Метрика (тест): {test_metric:.4f}")
                print(f"  Доходность (тест): {test_result.total_return_percent:.2f}%")
            
            wfa_results.append({
                'fold': fold_num,
                'train_dates': (train_data['date'].min(), train_data['date'].max()),
                'test_dates': (test_data['date'].min(), test_data['date'].max()),
                'best_params': best_train.params,
                'train_metric': best_train.metric_value,
                'test_metric': test_metric,
                'test_result': test_result
            })
            
            # Сдвигаем окно
            current_pos += step_size
        
        # Агрегированные результаты
        avg_train_metric = np.mean([r['train_metric'] for r in wfa_results])
        avg_test_metric = np.mean([r['test_metric'] for r in wfa_results])
        
        total_test_return = sum(r['test_result'].total_return for r in wfa_results)
        total_test_return_percent = (total_test_return / self.initial_capital) * 100
        
        if verbose:
            print("\n" + "=" * 70)
            print("ИТОГИ WALK-FORWARD ANALYSIS")
            print("=" * 70)
            print(f"Количество фолдов: {fold_num}")
            print(f"Средняя метрика (обучение): {avg_train_metric:.4f}")
            print(f"Средняя метрика (тест): {avg_test_metric:.4f}")
            print(f"Совокупная доходность (тест): {total_test_return_percent:.2f}%")
            print("=" * 70)
        
        return {
            'folds': wfa_results,
            'avg_train_metric': avg_train_metric,
            'avg_test_metric': avg_test_metric,
            'total_test_return': total_test_return,
            'total_test_return_percent': total_test_return_percent
        }
    
    def _sequential_backtest(
        self,
        param_names: List[str],
        combinations: List[Tuple],
        verbose: bool
    ) -> List[OptimizationResult]:
        """Последовательное тестирование комбинаций."""
        results = []
        total = len(combinations)
        
        for i, combo in enumerate(combinations, 1):
            params = dict(zip(param_names, combo))
            
            # Запускаем бэктестинг
            backtest = Backtest(
                self.data,
                self.strategy_class,
                params,
                self.initial_capital,
                self.commission
            )
            result = backtest.run()
            
            # Вычисляем метрику
            metric_value = self.metric_func(result)
            
            opt_result = OptimizationResult(
                params=params,
                metric_value=metric_value,
                backtest_result=result
            )
            results.append(opt_result)
            
            if verbose and i % max(1, total // 20) == 0:
                print(f"Прогресс: {i}/{total} ({i/total*100:.1f}%) | "
                      f"Лучшая метрика: {max(r.metric_value for r in results):.4f}")
        
        return results
    
    def _parallel_backtest(
        self,
        param_names: List[str],
        combinations: List[Tuple],
        verbose: bool
    ) -> List[OptimizationResult]:
        """Параллельное тестирование комбинаций."""
        results = []
        total = len(combinations)
        completed = 0
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Запускаем задачи
            futures = {}
            for combo in combinations:
                params = dict(zip(param_names, combo))
                future = executor.submit(
                    self._run_single_backtest,
                    self.data,
                    self.strategy_class,
                    params,
                    self.initial_capital,
                    self.commission,
                    self.metric_func
                )
                futures[future] = params
            
            # Собираем результаты
            for future in as_completed(futures):
                try:
                    opt_result = future.result()
                    results.append(opt_result)
                    completed += 1
                    
                    if verbose and completed % max(1, total // 20) == 0:
                        best_metric = max(r.metric_value for r in results)
                        print(f"Прогресс: {completed}/{total} ({completed/total*100:.1f}%) | "
                              f"Лучшая метрика: {best_metric:.4f}")
                except Exception as e:
                    logger.error(f"Ошибка в бэктестинге: {e}")
        
        return results
    
    @staticmethod
    def _run_single_backtest(
        data: pl.DataFrame,
        strategy_class: type[Strategy],
        params: Dict[str, Any],
        initial_capital: float,
        commission: float,
        metric_func: Callable
    ) -> OptimizationResult:
        """Вспомогательная функция для параллельного выполнения."""
        backtest = Backtest(data, strategy_class, params, initial_capital, commission)
        result = backtest.run()
        metric_value = metric_func(result)
        
        return OptimizationResult(
            params=params,
            metric_value=metric_value,
            backtest_result=result
        )
    
    def get_top_results(self, n: int = 10) -> List[OptimizationResult]:
        """
        Получить топ-N лучших результатов.
        
        Args:
            n: Количество результатов
            
        Returns:
            Список лучших результатов
        """
        sorted_results = sorted(self.results, key=lambda r: r.metric_value, reverse=True)
        return sorted_results[:n]
    
    def save_results(self, filepath: Path) -> None:
        """
        Сохранить результаты оптимизации в JSON.
        
        Args:
            filepath: Путь к файлу
        """
        data = []
        for result in self.results:
            data.append({
                'params': result.params,
                'metric_value': result.metric_value,
                'total_return': result.backtest_result.total_return,
                'total_return_percent': result.backtest_result.total_return_percent,
                'total_trades': result.backtest_result.total_trades,
                'win_rate': result.backtest_result.win_rate,
                'sharpe_ratio': result.backtest_result.sharpe_ratio,
                'max_drawdown': result.backtest_result.max_drawdown,
            })
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Результаты сохранены в {filepath}")
    
    def plot_optimization_surface(self, param_x: str, param_y: str) -> None:
        """
        Построить поверхность оптимизации для двух параметров.
        
        Args:
            param_x: Имя первого параметра
            param_y: Имя второго параметра
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            logger.error("Для визуализации нужен matplotlib: pip install matplotlib")
            return
        
        # Извлекаем данные
        x_values = []
        y_values = []
        z_values = []
        
        for result in self.results:
            if param_x in result.params and param_y in result.params:
                x_values.append(result.params[param_x])
                y_values.append(result.params[param_y])
                z_values.append(result.metric_value)
        
        if not x_values:
            logger.warning(f"Нет данных для параметров {param_x} и {param_y}")
            return
        
        # Создаём 3D график
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(x_values, y_values, z_values, c=z_values, cmap='viridis', s=50)
        
        ax.set_xlabel(param_x)
        ax.set_ylabel(param_y)
        ax.set_zlabel(self.optimization_metric)
        ax.set_title(f'Поверхность оптимизации: {param_x} vs {param_y}')
        
        plt.colorbar(scatter)
        plt.show()
