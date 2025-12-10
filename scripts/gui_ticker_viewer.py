"""Небольшое Tkinter-приложение для просмотра свечных графиков из папки `data/tickers`."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path

import pandas as pd

# Используем TkAgg до импорта pyplot, чтобы корректно встраивать график в Tkinter.
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

# Корень проекта и путь до данных, работает даже при запуске из папки scripts/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = (PROJECT_ROOT / "data" / "tickers").resolve()


def list_tickers() -> list[str]:
    """Возвращает список доступных тикеров (имена папок)."""
    if not DATA_ROOT.exists():
        return []
    return sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())


def list_timeframes(ticker: str) -> list[str]:
    """Возвращает список таймфреймов (имена parquet-файлов без суффикса)."""
    ticker_dir = DATA_ROOT / ticker
    if not ticker_dir.exists():
        return []
    return sorted(f.stem for f in ticker_dir.glob("*.parquet"))


def load_candles(ticker: str, timeframe: str) -> pd.DataFrame:
    """Читает parquet и нормализует столбец времени."""
    file_path = DATA_ROOT / ticker / f"{timeframe}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Не найден файл {file_path}")

    df = pd.read_parquet(file_path)
    # Индекс часто хранится в parquet как time, восстанавливаем явный столбец.
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    else:
        df = df.reset_index()
        df.rename(columns={df.columns[0]: "time"}, inplace=True)
        df["time"] = pd.to_datetime(df["time"])

    # Снимаем tz, чтобы matplotlib корректно форматировал ось времени.
    if pd.api.types.is_datetime64_any_dtype(df["time"]) and hasattr(
        df["time"].dt, "tz"
    ):
        try:
            df["time"] = df["time"].dt.tz_convert(None)
        except TypeError:
            # Если tz уже отсутствует.
            pass
    return df.sort_values("time")


class ChartApp:
    """Tkinter GUI для выбора тикера, таймфрейма и просмотра свечей."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Просмотр графиков из data/tickers")

        # Состояние выбора пользователя.
        self.ticker_var = tk.StringVar()
        self.timeframe_var = tk.StringVar()
        self.limit_var = tk.IntVar(value=300)
        self.show_sma_var = tk.BooleanVar(value=True)
        self.show_ema_var = tk.BooleanVar(value=False)

        # Матплотлиб-объекты.
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.ax_price = self.figure.add_subplot(2, 1, 1)
        self.ax_volume = self.figure.add_subplot(2, 1, 2, sharex=self.ax_price)

        self._build_ui()
        self._load_initial_options()

    def _build_ui(self) -> None:
        """Создает панель управления и поле графика."""
        controls = ttk.Frame(self.root, padding=10)
        controls.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(controls, text="Тикер").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.ticker_combo = ttk.Combobox(
            controls, textvariable=self.ticker_var, state="readonly", width=10
        )
        self.ticker_combo.grid(row=0, column=1, padx=5)
        self.ticker_combo.bind("<<ComboboxSelected>>", self._on_ticker_changed)

        ttk.Label(controls, text="Таймфрейм").grid(
            row=0, column=2, sticky=tk.W, padx=5
        )
        self.tf_combo = ttk.Combobox(
            controls, textvariable=self.timeframe_var, state="readonly", width=8
        )
        self.tf_combo.grid(row=0, column=3, padx=5)

        ttk.Label(controls, text="Свечей").grid(row=0, column=4, sticky=tk.W, padx=5)
        ttk.Spinbox(
            controls, from_=50, to=5000, textvariable=self.limit_var, width=7
        ).grid(row=0, column=5, padx=5)

        ttk.Checkbutton(
            controls, text="SMA20", variable=self.show_sma_var
        ).grid(row=0, column=6, padx=5)
        ttk.Checkbutton(
            controls, text="EMA50", variable=self.show_ema_var
        ).grid(row=0, column=7, padx=5)

        ttk.Button(controls, text="Обновить", command=self._refresh_chart).grid(
            row=0, column=8, padx=5
        )
        ttk.Button(controls, text="Обновить список", command=self._load_initial_options).grid(
            row=0, column=9, padx=5
        )

        self.status_var = tk.StringVar(value="Готово")
        ttk.Label(controls, textvariable=self.status_var, foreground="gray").grid(
            row=1, column=0, columnspan=10, sticky=tk.W, pady=(5, 0)
        )

        canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas = canvas

    def _load_initial_options(self) -> None:
        """Подгружает списки тикеров/таймфреймов и отрисовывает график по умолчанию."""
        tickers = list_tickers()
        self.ticker_combo["values"] = tickers
        if tickers:
            # Оставляем текущий выбор, если он еще существует.
            current = self.ticker_var.get()
            self.ticker_var.set(current if current in tickers else tickers[0])
            self._set_timeframes(self.ticker_var.get())
            self._refresh_chart()
        else:
            self.status_var.set("Не найдены данные в папке data/tickers")

    def _set_timeframes(self, ticker: str) -> None:
        """Обновляет список таймфреймов для выбранного тикера."""
        timeframes = list_timeframes(ticker)
        self.tf_combo["values"] = timeframes
        if timeframes:
            current = self.timeframe_var.get()
            self.timeframe_var.set(current if current in timeframes else timeframes[0])

    def _on_ticker_changed(self, event: tk.Event) -> None:
        """При смене тикера обновляем таймфреймы и график."""
        selected = self.ticker_var.get()
        self._set_timeframes(selected)
        self._refresh_chart()

    def _refresh_chart(self) -> None:
        """Читает parquet и перерисовывает график."""
        ticker = self.ticker_var.get()
        timeframe = self.timeframe_var.get()
        if not ticker or not timeframe:
            self.status_var.set("Выберите тикер и таймфрейм")
            return

        try:
            df = load_candles(ticker, timeframe)
        except Exception as exc:  # pylint: disable=broad-except
            messagebox.showerror("Ошибка загрузки", str(exc))
            self.status_var.set("Ошибка загрузки данных")
            return

        limit = max(1, int(self.limit_var.get()))
        if len(df) > limit:
            df = df.tail(limit)

        self._draw_candles(df, ticker, timeframe)
        self.canvas.draw()
        self.status_var.set(f"Показано {len(df)} свечей {ticker} / {timeframe}")

    def _draw_candles(self, df: pd.DataFrame, ticker: str, timeframe: str) -> None:
        """Рисует свечи и объемы, накладывает индикаторы при наличии столбцов."""
        self.ax_price.clear()
        self.ax_volume.clear()

        # Приводим время к datetime и убираем NaT, чтобы избежать падений при конверсии.
        times = pd.to_datetime(df["time"], errors="coerce")
        df = df.loc[~times.isna()].copy()
        dates = mdates.date2num(times.dropna().dt.to_pydatetime())
        width = 0.6

        for x, (_, row) in zip(dates, df.iterrows()):
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]
            color = "#2ca02c" if c >= o else "#d62728"
            self.ax_price.plot([x, x], [l, h], color=color, linewidth=1)
            body_height = abs(c - o) or 1e-6
            bottom = min(o, c)
            rect = Rectangle(
                (x - width / 2, bottom), width, body_height, color=color, alpha=0.6
            )
            self.ax_price.add_patch(rect)

        if self.show_sma_var.get() and "sma20" in df.columns:
            self.ax_price.plot(dates, df["sma20"], color="#1f77b4", label="SMA20")
        if self.show_ema_var.get() and "ema50" in df.columns:
            self.ax_price.plot(dates, df["ema50"], color="#9467bd", label="EMA50")

        self.ax_price.set_title(f"{ticker} / {timeframe}")
        self.ax_price.set_ylabel("Цена")
        if self.ax_price.get_legend_handles_labels()[0]:
            self.ax_price.legend(loc="upper left")

        self.ax_price.xaxis_date()
        self.ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        self.figure.autofmt_xdate()

        if "volume" in df.columns:
            self.ax_volume.bar(
                dates, df["volume"], width=width * 0.8, color="#4c72b0", alpha=0.8
            )
            self.ax_volume.set_ylabel("Объем")
        else:
            self.ax_volume.text(
                0.5,
                0.5,
                "Нет столбца volume",
                ha="center",
                va="center",
                transform=self.ax_volume.transAxes,
            )
        self.ax_volume.xaxis_date()
        self.ax_volume.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))


def main() -> None:
    root = tk.Tk()
    app = ChartApp(root)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()

