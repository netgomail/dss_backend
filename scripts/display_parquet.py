"""
Скрипт для отображения данных из Parquet файла в человеко читаемом формате.

Показывает OHLCV данные (Open, High, Low, Close, Volume) с датами в московском времени.

Использование:
    python scripts/display_parquet.py <путь_к_parquet_файлу> [--limit N]

Примеры:
    python scripts/display_parquet.py data/tickers/AFKS/1D.parquet
    python scripts/display_parquet.py data/tickers/SBER/1H.parquet --limit 20
    python scripts/display_parquet.py data/tickers/AFKS/5M.parquet --limit 50
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# === НАСТРОЙКА ПУТИ ПРОЕКТА ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

console = Console()


def format_datetime(dt) -> str:
    """
    Форматирует datetime в человеко читаемый формат.

    Args:
        dt: Pandas datetime объект

    Returns:
        Строка с отформатированной датой
    """
    # Конвертируем в московское время
    dt_moscow = dt.tz_convert("Europe/Moscow")
    return dt_moscow.strftime("%Y-%m-%d %H:%M:%S")


def format_number(value: float, decimals: int = 2) -> str:
    """
    Форматирует числовое значение с заданным количеством десятичных знаков.

    Args:
        value: Числовое значение
        decimals: Количество десятичных знаков

    Returns:
        Отформатированная строка
    """
    if decimals == 0:
        return f"{value:,.0f}"
    return f"{value:,.{decimals}f}"


def display_parquet_data(file_path: str, limit: Optional[int] = None) -> None:
    """
    Отображает данные из Parquet файла в красивом табличном формате.

    Args:
        file_path: Путь к Parquet файлу
        limit: Максимальное количество строк для отображения (None - все строки)
    """
    file_path_obj = Path(file_path)

    if not file_path_obj.exists():
        console.print(f"[red]Ошибка:[/red] Файл не найден: {file_path}")
        return

    try:
        # Читаем Parquet файл
        df = pd.read_parquet(file_path_obj)

        if df.empty:
            console.print(f"[yellow]Файл пуст:[/yellow] {file_path}")
            return

        # Показываем информацию о файле
        console.print(Panel.fit(
            f"[bold blue]Файл:[/bold blue] {file_path}\n"
            f"[bold green]Всего строк:[/bold green] {len(df)}\n"
            f"[bold cyan]Колонки:[/bold cyan] {', '.join(df.columns)}",
            title="📊 Информация о данных"
        ))

        # Применяем лимит если указан
        if limit:
            df_display = df.head(limit)
            if len(df) > limit:
                console.print(f"[dim]Показаны первые {limit} строк из {len(df)}[/dim]")
        else:
            df_display = df

        # Создаем таблицу
        table = Table(title=f"📈 Данные из {file_path_obj.name}", show_header=True, header_style="bold magenta")

        # Добавляем колонки
        table.add_column("Дата/Время (МСК)", style="cyan", no_wrap=True)
        table.add_column("Open", style="green", justify="right")
        table.add_column("High", style="red", justify="right")
        table.add_column("Low", style="red", justify="right")
        table.add_column("Close", style="green", justify="right")
        table.add_column("Volume", style="yellow", justify="right")

        # Добавляем строки данных
        for _, row in df_display.iterrows():
            date_str = format_datetime(row['date'])

            table.add_row(
                date_str,
                format_number(row['open'], 2),
                format_number(row['high'], 2),
                format_number(row['low'], 2),
                format_number(row['close'], 2),
                format_number(row['volume'], 0)
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Ошибка при чтении файла:[/red] {e}")


def main():
    """Основная функция скрипта."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Отображение данных из Parquet файла в человеко читаемом формате",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python scripts/display_parquet.py data/tickers/AFKS/1D.parquet
  python scripts/display_parquet.py data/tickers/SBER/1H.parquet --limit 20
  python scripts/display_parquet.py data/tickers/AFKS/5M.parquet --limit 50
        """
    )

    parser.add_argument(
        "file_path",
        help="Путь к Parquet файлу"
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Максимальное количество строк для отображения (по умолчанию все строки)"
    )

    args = parser.parse_args()

    # Показываем данные
    display_parquet_data(args.file_path, args.limit)


if __name__ == "__main__":
    main()
