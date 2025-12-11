"""
Модуль для отображения финансовых графиков с GUI интерфейсом.

Предоставляет графическое окно с возможностью:
    - Выбора тикера из списка доступных инструментов
    - Выбора таймфрейма (M5, M15, M30, H1, H2, H4, D1, Week)
    - Отображения свечного графика с объёмом
    - Добавления технических индикаторов (SMA, EMA, Bollinger Bands)
    - Настройки количества отображаемых свечей

Пример использования:
    >>> from services.chart_viewer import ChartViewer
    >>> app = ChartViewer()
    >>> app.run()

Или через командную строку:
    $ python services/chart_viewer.py
"""

import sys
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# === НАСТРОЙКА ПУТИ ПРОЕКТА ===
# Добавляем корень проекта в sys.path для импорта локальных модулей
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import settings


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================


@dataclass
class ChartConfig:
    """
    Конфигурация для отображения графика.
    
    Attributes:
        data_dir: Директория с данными тикеров
        default_ticker: Тикер по умолчанию при запуске
        default_timeframe: Таймфрейм по умолчанию
        default_candles: Количество свечей по умолчанию
        available_timeframes: Список доступных таймфреймов
        candle_options: Варианты количества свечей для выбора
        window_size: Размер окна (ширина, высота)
        chart_style: Стиль графика mplfinance
    """
    # Пути к данным
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "tickers")
    
    # Настройки по умолчанию
    default_ticker: str = "SBER"
    default_timeframe: str = "H1"
    default_candles: int = 100
    
    # Доступные таймфреймы с человекочитаемыми названиями
    available_timeframes: Dict[str, str] = field(default_factory=lambda: {
        "M5": "5 минут",
        "M15": "15 минут",
        "M30": "30 минут",
        "H1": "1 час",
        "H2": "2 часа",
        "H4": "4 часа",
        "D1": "1 день",
        "Week": "Неделя",
    })
    
    # Варианты количества свечей
    candle_options: List[int] = field(default_factory=lambda: [50, 100, 200, 500, 1000])
    
    # Размер окна приложения
    window_size: Tuple[int, int] = (1400, 900)
    
    # Стиль графика (доступные: 'charles', 'mike', 'nightclouds', 'yahoo', 'binance')
    chart_style: str = "charles"


# ============================================================================
# ЗАГРУЗЧИК ДАННЫХ
# ============================================================================


class DataLoader:
    """
    Загрузчик данных из Parquet файлов.
    
    Обеспечивает загрузку и подготовку данных для отображения на графике.
    
    Attributes:
        data_dir: Директория с данными тикеров
    """
    
    def __init__(self, data_dir: Path) -> None:
        """
        Инициализирует загрузчик данных.
        
        Args:
            data_dir: Путь к директории с данными тикеров
        """
        self.data_dir = data_dir
    
    def get_available_tickers(self) -> List[str]:
        """
        Возвращает список доступных тикеров.
        
        Сканирует директорию данных и возвращает список папок (тикеров),
        отсортированный по алфавиту.
        
        Returns:
            Список тикеров
        """
        if not self.data_dir.exists():
            return []
        
        tickers = [
            d.name for d in self.data_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        return sorted(tickers)
    
    def load_data(
        self,
        ticker: str,
        timeframe: str,
        num_candles: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Загружает данные для указанного тикера и таймфрейма.
        
        Args:
            ticker: Тикер инструмента
            timeframe: Название таймфрейма (M5, H1, D1 и т.д.)
            num_candles: Количество последних свечей (None = все данные)
        
        Returns:
            DataFrame с OHLCV данными или None при ошибке
        """
        file_path = self.data_dir / ticker / f"{timeframe}.parquet"
        
        if not file_path.exists():
            return None
        
        try:
            df = pd.read_parquet(file_path)
            
            # Убеждаемся, что индекс — datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                if "time" in df.columns:
                    df = df.set_index("time")
            
            # Сортируем по времени
            df = df.sort_index()
            
            # Ограничиваем количество свечей
            if num_candles is not None and len(df) > num_candles:
                df = df.tail(num_candles)
            
            return df
            
        except Exception as e:
            print(f"Ошибка загрузки данных: {e}")
            return None
    
    def get_ticker_info(self, ticker: str) -> Optional[Dict]:
        """
        Возвращает информацию о тикере из settings.
        
        Args:
            ticker: Тикер инструмента
        
        Returns:
            Словарь с информацией о тикере или None
        """
        for instrument in settings.INSTRUMENTS:
            if instrument["name"] == ticker:
                return instrument
        return None


# ============================================================================
# ПОСТРОИТЕЛЬ ГРАФИКОВ
# ============================================================================


class ChartBuilder:
    """
    Построитель финансовых графиков на основе mplfinance.
    
    Создаёт свечные графики с объёмом и техническими индикаторами.
    
    Attributes:
        style: Стиль графика mplfinance
    """
    
    def __init__(self, style: str = "charles") -> None:
        """
        Инициализирует построитель графиков.
        
        Args:
            style: Название стиля mplfinance
        """
        self.style = style
        
        # Кастомный стиль для более приятного отображения
        self.market_colors = mpf.make_marketcolors(
            up="#26a69a",      # Зелёный для роста
            down="#ef5350",    # Красный для падения
            edge="inherit",
            wick="inherit",
            volume="inherit",
        )
        
        self.custom_style = mpf.make_mpf_style(
            marketcolors=self.market_colors,
            gridstyle="-",
            gridcolor="#e0e0e0",
            facecolor="white",
            figcolor="white",
        )
    
    def create_figure(
        self,
        df: pd.DataFrame,
        title: str,
        show_volume: bool = True,
        show_sma: bool = False,
        show_ema: bool = False,
        show_bb: bool = False,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Создаёт фигуру графика.
        
        Args:
            df: DataFrame с OHLCV данными
            title: Заголовок графика
            show_volume: Отображать ли объём
            show_sma: Отображать ли SMA (20, 50)
            show_ema: Отображать ли EMA (10, 20)
            show_bb: Отображать ли Bollinger Bands
            figsize: Размер фигуры в дюймах
        
        Returns:
            Объект Figure matplotlib
        """
        # Подготавливаем данные — оставляем только OHLCV
        plot_df = df[["open", "high", "low", "close", "volume"]].copy()
        
        # Переименовываем столбцы для mplfinance (требует заглавные буквы)
        plot_df.columns = ["Open", "High", "Low", "Close", "Volume"]
        
        # Список дополнительных графиков (addplot)
        addplots = []
        
        # Добавляем SMA если запрошено и есть в данных
        if show_sma:
            if "sma20" in df.columns:
                addplots.append(mpf.make_addplot(
                    df["sma20"].tail(len(plot_df)),
                    color="#2196f3",
                    width=1.0,
                    label="SMA 20"
                ))
            if "sma50" in df.columns:
                addplots.append(mpf.make_addplot(
                    df["sma50"].tail(len(plot_df)),
                    color="#ff9800",
                    width=1.0,
                    label="SMA 50"
                ))
        
        # Добавляем EMA если запрошено и есть в данных
        if show_ema:
            if "ema10" in df.columns:
                addplots.append(mpf.make_addplot(
                    df["ema10"].tail(len(plot_df)),
                    color="#9c27b0",
                    width=1.0,
                    linestyle="--",
                    label="EMA 10"
                ))
            if "ema20" in df.columns:
                addplots.append(mpf.make_addplot(
                    df["ema20"].tail(len(plot_df)),
                    color="#e91e63",
                    width=1.0,
                    linestyle="--",
                    label="EMA 20"
                ))
        
        # Добавляем Bollinger Bands если запрошено и есть в данных
        if show_bb:
            if all(col in df.columns for col in ["bb_upper", "bb_middle", "bb_lower"]):
                addplots.append(mpf.make_addplot(
                    df["bb_upper"].tail(len(plot_df)),
                    color="#607d8b",
                    width=0.8,
                    linestyle=":",
                ))
                addplots.append(mpf.make_addplot(
                    df["bb_middle"].tail(len(plot_df)),
                    color="#607d8b",
                    width=0.8,
                ))
                addplots.append(mpf.make_addplot(
                    df["bb_lower"].tail(len(plot_df)),
                    color="#607d8b",
                    width=0.8,
                    linestyle=":",
                ))
        
        # Создаём график
        # Формируем kwargs для mpf.plot (addplot передаём только если есть индикаторы)
        plot_kwargs = {
            "type": "candle",
            "style": self.custom_style,
            "title": title,
            "volume": show_volume,
            "figsize": figsize,
            "returnfig": True,
            "panel_ratios": (4, 1) if show_volume else (1,),
            "tight_layout": True,
        }
        
        # Добавляем addplot только если список не пустой
        if addplots:
            plot_kwargs["addplot"] = addplots
        
        fig, axes = mpf.plot(plot_df, **plot_kwargs)
        
        return fig


# ============================================================================
# ГЛАВНОЕ ОКНО ПРИЛОЖЕНИЯ
# ============================================================================


class ChartViewer:
    """
    Главное окно приложения для просмотра финансовых графиков.
    
    Предоставляет GUI с возможностью выбора тикера, таймфрейма,
    количества свечей и отображения индикаторов.
    
    Attributes:
        config: Конфигурация приложения
        data_loader: Загрузчик данных
        chart_builder: Построитель графиков
    
    Example:
        >>> viewer = ChartViewer()
        >>> viewer.run()
    """
    
    def __init__(self, config: Optional[ChartConfig] = None) -> None:
        """
        Инициализирует приложение.
        
        Args:
            config: Конфигурация (по умолчанию создаётся стандартная)
        """
        self.config = config or ChartConfig()
        self.data_loader = DataLoader(self.config.data_dir)
        self.chart_builder = ChartBuilder(self.config.chart_style)
        
        # Текущая фигура графика
        self._current_fig: Optional[plt.Figure] = None
        self._canvas: Optional[FigureCanvasTkAgg] = None
        
        # Инициализация GUI
        self._init_window()
        self._init_controls()
        self._init_chart_area()
        
        # Загружаем начальный график
        self._update_chart()
    
    def _init_window(self) -> None:
        """Инициализирует главное окно приложения."""
        self.root = tk.Tk()
        self.root.title("📈 DSS Chart Viewer — Финансовые графики")
        
        # Устанавливаем размер окна
        width, height = self.config.window_size
        self.root.geometry(f"{width}x{height}")
        
        # Минимальный размер окна
        self.root.minsize(800, 600)
        
        # Обработка закрытия окна
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _init_controls(self) -> None:
        """Инициализирует панель управления с элементами выбора."""
        # Фрейм для элементов управления
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X, side=tk.TOP)
        
        # === ВЫБОР ТИКЕРА ===
        ttk.Label(control_frame, text="Тикер:").pack(side=tk.LEFT, padx=(0, 5))
        
        # Получаем список тикеров
        tickers = self.data_loader.get_available_tickers()
        if not tickers:
            tickers = [self.config.default_ticker]
        
        self.ticker_var = tk.StringVar(value=self.config.default_ticker)
        ticker_combo = ttk.Combobox(
            control_frame,
            textvariable=self.ticker_var,
            values=tickers,
            state="readonly",
            width=10
        )
        ticker_combo.pack(side=tk.LEFT, padx=(0, 20))
        ticker_combo.bind("<<ComboboxSelected>>", lambda e: self._update_chart())
        
        # === ВЫБОР ТАЙМФРЕЙМА ===
        ttk.Label(control_frame, text="Таймфрейм:").pack(side=tk.LEFT, padx=(0, 5))
        
        # Создаём список для combobox с читаемыми названиями
        tf_display = [
            f"{k} ({v})" for k, v in self.config.available_timeframes.items()
        ]
        tf_keys = list(self.config.available_timeframes.keys())
        
        self.timeframe_var = tk.StringVar(value=self.config.default_timeframe)
        self._tf_display_to_key = dict(zip(tf_display, tf_keys))
        self._tf_key_to_display = dict(zip(tf_keys, tf_display))
        
        timeframe_combo = ttk.Combobox(
            control_frame,
            textvariable=self.timeframe_var,
            values=tf_display,
            state="readonly",
            width=15
        )
        # Устанавливаем отображаемое значение
        timeframe_combo.set(self._tf_key_to_display.get(
            self.config.default_timeframe,
            tf_display[0]
        ))
        timeframe_combo.pack(side=tk.LEFT, padx=(0, 20))
        timeframe_combo.bind("<<ComboboxSelected>>", self._on_timeframe_change)
        
        # === КОЛИЧЕСТВО СВЕЧЕЙ ===
        ttk.Label(control_frame, text="Свечей:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.candles_var = tk.StringVar(value=str(self.config.default_candles))
        candles_combo = ttk.Combobox(
            control_frame,
            textvariable=self.candles_var,
            values=[str(n) for n in self.config.candle_options],
            state="readonly",
            width=8
        )
        candles_combo.pack(side=tk.LEFT, padx=(0, 20))
        candles_combo.bind("<<ComboboxSelected>>", lambda e: self._update_chart())
        
        # === РАЗДЕЛИТЕЛЬ ===
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=10
        )
        
        # === ЧЕКБОКСЫ ИНДИКАТОРОВ ===
        ttk.Label(control_frame, text="Индикаторы:").pack(side=tk.LEFT, padx=(0, 10))
        
        # SMA
        self.show_sma_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            control_frame,
            text="SMA",
            variable=self.show_sma_var,
            command=self._update_chart
        ).pack(side=tk.LEFT, padx=5)
        
        # EMA
        self.show_ema_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            control_frame,
            text="EMA",
            variable=self.show_ema_var,
            command=self._update_chart
        ).pack(side=tk.LEFT, padx=5)
        
        # Bollinger Bands
        self.show_bb_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            control_frame,
            text="BB",
            variable=self.show_bb_var,
            command=self._update_chart
        ).pack(side=tk.LEFT, padx=5)
        
        # Volume (включён по умолчанию)
        self.show_volume_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            control_frame,
            text="Объём",
            variable=self.show_volume_var,
            command=self._update_chart
        ).pack(side=tk.LEFT, padx=5)
        
        # === КНОПКА ОБНОВЛЕНИЯ ===
        ttk.Button(
            control_frame,
            text="🔄 Обновить",
            command=self._update_chart
        ).pack(side=tk.RIGHT, padx=5)
        
        # === ИНФОРМАЦИОННАЯ ПАНЕЛЬ ===
        self.info_label = ttk.Label(
            control_frame,
            text="",
            foreground="gray"
        )
        self.info_label.pack(side=tk.RIGHT, padx=20)
    
    def _init_chart_area(self) -> None:
        """Инициализирует область для отображения графика."""
        # Фрейм для графика
        self.chart_frame = ttk.Frame(self.root)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
    
    def _on_timeframe_change(self, event) -> None:
        """Обработчик изменения таймфрейма."""
        # Получаем выбранное отображаемое значение
        display_value = event.widget.get()
        # Конвертируем в ключ таймфрейма
        tf_key = self._tf_display_to_key.get(display_value, self.config.default_timeframe)
        self.timeframe_var.set(tf_key)
        self._update_chart()
    
    def _update_chart(self) -> None:
        """Обновляет график с текущими настройками."""
        ticker = self.ticker_var.get()
        
        # Получаем таймфрейм (может быть ключ или отображаемое значение)
        tf_value = self.timeframe_var.get()
        timeframe = self._tf_display_to_key.get(tf_value, tf_value)
        
        num_candles = int(self.candles_var.get())
        
        # Загружаем данные
        df = self.data_loader.load_data(ticker, timeframe, num_candles)
        
        if df is None or df.empty:
            messagebox.showwarning(
                "Нет данных",
                f"Не удалось загрузить данные для {ticker} ({timeframe})"
            )
            return
        
        # Получаем информацию о тикере
        ticker_info = self.data_loader.get_ticker_info(ticker)
        ticker_name = ticker_info["alias"] if ticker_info else ticker
        
        # Формируем заголовок
        tf_display = self.config.available_timeframes.get(timeframe, timeframe)
        title = f"{ticker} ({ticker_name}) — {tf_display}"
        
        # Удаляем старый график
        self._clear_chart()
        
        # Создаём новый график
        try:
            fig = self.chart_builder.create_figure(
                df=df,
                title=title,
                show_volume=self.show_volume_var.get(),
                show_sma=self.show_sma_var.get(),
                show_ema=self.show_ema_var.get(),
                show_bb=self.show_bb_var.get(),
                figsize=(14, 8)
            )
            
            self._current_fig = fig
            
            # Встраиваем график в tkinter
            self._canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
            self._canvas.draw()
            
            # Добавляем тулбар навигации
            toolbar_frame = ttk.Frame(self.chart_frame)
            toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
            toolbar = NavigationToolbar2Tk(self._canvas, toolbar_frame)
            toolbar.update()
            
            # Добавляем canvas
            self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Обновляем информационную панель
            self._update_info(df, ticker, timeframe)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось построить график:\n{e}")
    
    def _clear_chart(self) -> None:
        """Очищает текущий график."""
        if self._current_fig is not None:
            plt.close(self._current_fig)
            self._current_fig = None
        
        # Удаляем все виджеты из фрейма графика
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
    
    def _update_info(self, df: pd.DataFrame, ticker: str, timeframe: str) -> None:
        """
        Обновляет информационную панель.
        
        Args:
            df: DataFrame с данными
            ticker: Тикер
            timeframe: Таймфрейм
        """
        if df.empty:
            self.info_label.config(text="")
            return
        
        # Получаем последнюю цену и изменение
        last_close = df["close"].iloc[-1]
        prev_close = df["close"].iloc[-2] if len(df) > 1 else last_close
        change = last_close - prev_close
        change_pct = (change / prev_close * 100) if prev_close != 0 else 0
        
        # Формируем текст
        sign = "+" if change >= 0 else ""
        info_text = (
            f"Цена: {last_close:.2f} | "
            f"Изменение: {sign}{change:.2f} ({sign}{change_pct:.2f}%) | "
            f"Свечей: {len(df)}"
        )
        
        # Меняем цвет в зависимости от изменения
        color = "#26a69a" if change >= 0 else "#ef5350"
        
        self.info_label.config(text=info_text, foreground=color)
    
    def _on_close(self) -> None:
        """Обработчик закрытия окна."""
        self._clear_chart()
        self.root.destroy()
    
    def run(self) -> None:
        """Запускает главный цикл приложения."""
        self.root.mainloop()


# ============================================================================
# ТОЧКА ВХОДА
# ============================================================================


def main() -> None:
    """
    Точка входа для запуска приложения.
    
    Создаёт экземпляр ChartViewer и запускает GUI.
    """
    # Проверяем наличие данных
    data_dir = PROJECT_ROOT / "data" / "tickers"
    if not data_dir.exists():
        print("⚠️ Директория с данными не найдена!")
        print(f"   Ожидаемый путь: {data_dir}")
        print("   Сначала запустите scripts/data_collector.py для загрузки данных.")
        return
    
    # Запускаем приложение
    app = ChartViewer()
    app.run()


if __name__ == "__main__":
    main()
