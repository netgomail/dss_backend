"""Generate INSTRUMENTS entries for selected tickers and update settings.py.

Requires environment variable INVEST_TOKEN (real or sandbox).
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from tinkoff.invest import Client, InstrumentIdType
from tinkoff.invest.schemas import InstrumentStatus

# Tickers to fetch metadata for.
ASSETS = [
    "IMOEXF",
    "AFKS",
    "AFLT",
    "ALRS",
    "ASTR",
    "BSPB",
    "CBOM",
    "CHMF",
    "ENPG",
    "FEES",
    "FLOT",
    "GAZP",
    "GMKN",
    "HEAD",
    "HYDR",
    "IRAO",
    "LEAS",
    "LKOH",
    "MAGN",
    "MGNT",
    "MOEX",
    "MSNG",
    "MTLR",
    "MTSS",
    "NLMK",
    "NVTK",
    "PHOR",
    "PIKK",
    "PLZL",
    "POSI",
    "ROSN",
    "RTKM",
    "RUAL",
    "SBER",
    "SELG",
    "SMLT",
    "SNGS",
    "SVCB",
    "T",
    "TATN",
    "UGLD",
    "UPRO",
    "VKCO",
    "VTBR",
    "YDEX",
]

# Instruments we always keep in settings in addition to generated shares.
STATIC_INSTRUMENTS = []

# Prefer MOEX class codes when several instruments share the same ticker.
# Includes codes for shares, ETFs, bonds, and futures
PREFERRED_CLASS_CODES = [
    # Акции и основные инструменты
    "TQBR",   # Основной режим торгов акциями
    "TQTF",   # Режим торгов ETF
    "TQTD",   # Режим торгов депозитарными расписками
    "TQTE",   # Режим торгов иностранными ETF
    "TQIF",   # Режим торгов ПИФами
    "TQPI",   # Режим торгов паями
    "TQBD",   # Режим торгов облигациями
    "EQNE",   # Внесписочный режим
    "SPBXM",  # Режим SPB Exchange
    "SPBB",   # Режим SPB Биржи

    # Фьючерсы и производные
    "SPBFUT", # Фьючерсы на СПБ
    "SPBDE",  # Опционы на СПБ
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = PROJECT_ROOT / "settings.py"
ENV_PATH = PROJECT_ROOT / ".env.local"


def fetch_instrument_by_ticker(client: Client, ticker: str, class_code: str = "") -> object:
    """
    Получить полный объект инструмента по тикеру и класс-коду.

    Использует метод get_instrument_by для получения полной информации об инструменте.

    Args:
        client: Клиент Tinkoff Invest API
        ticker: Тикер инструмента
        class_code: Код режима торгов (опционально)

    Returns:
        Объект Instrument с полной информацией или None, если не найден
    """
    try:
        response = client.instruments.get_instrument_by(
            id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_TICKER,
            id=ticker,
            class_code=class_code if class_code else None
        )
        return response.instrument
    except Exception as e:
        # Инструмент не найден с данным class_code
        return None


def fetch_instruments(client: Client) -> Dict[str, object]:
    """
    Получить все доступные инструменты по списку тикеров с полной информацией.

    Для каждого тикера из ASSETS пытается найти инструмент, перебирая
    приоритетные class_codes. Использует get_instrument_by для получения
    полной информации о каждом инструменте.

    Returns:
        Словарь {ticker: instrument} с полными объектами Instrument
    """
    instruments: Dict[str, object] = {}

    for ticker in ASSETS:
        instrument = None

        # Сначала пробуем получить без указания class_code
        instrument = fetch_instrument_by_ticker(client, ticker)

        # Если не нашли, пробуем с приоритетными class_codes
        if instrument is None:
            for class_code in PREFERRED_CLASS_CODES:
                instrument = fetch_instrument_by_ticker(client, ticker, class_code)
                if instrument is not None:
                    break

        if instrument is not None:
            instruments[ticker] = instrument

    return instruments


def build_instruments(client: Client) -> Tuple[List[dict], List[str]]:
    """
    Prepare instrument dicts for configured assets and track missing tickers.

    Использует get_instrument_by для получения полной информации о каждом инструменте.
    Определяет тип инструмента по полю instrument_type.

    Returns:
        Tuple[List[dict], List[str]]: (список инструментов, список ненайденных тикеров)
    """
    # Получаем все инструменты с полной информацией
    instruments_by_ticker = fetch_instruments(client)

    instruments: List[dict] = []
    missing: List[str] = []

    for ticker in ASSETS:
        instrument = instruments_by_ticker.get(ticker)

        if instrument is None:
            missing.append(ticker)
            continue

        # Создаем запись на основе полной информации об инструменте
        instrument_dict = {
            "ticker": instrument.ticker,
            "name": getattr(instrument, 'name', ''),
            "figi": instrument.figi,
            "class_code": instrument.class_code,
            "instrument_type": getattr(instrument, 'instrument_type', ''),
        }

        instruments.append(instrument_dict)

    return instruments, missing


def load_env_local(env_path: Path = ENV_PATH) -> None:
    """Load key=value pairs from .env.local into os.environ if present."""
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def format_instrument_entry(entry: dict) -> str:
    """Render a single instrument dict to a python-literal string."""
    ordered_keys = [
        "ticker",
        "name",
        "figi",
        "class_code",
        "instrument_type",
    ]
    parts = []
    for key in ordered_keys:
        if key not in entry or entry[key] is None:
            continue
        parts.append(f'"{key}": {json.dumps(entry[key], ensure_ascii=False)}')
    return "{ " + ", ".join(parts) + " }"


def render_instruments_block(instruments: List[dict]) -> str:
    """Render the full INSTRUMENTS assignment block."""
    lines = ["INSTRUMENTS = ["]
    for instrument in instruments:
        lines.append(f"    {format_instrument_entry(instrument)},")
    lines.append("]")
    return "\n".join(lines)


def update_settings_file(instruments: List[dict]) -> None:
    """Replace INSTRUMENTS block in settings.py with freshly generated data."""
    serialized_block = render_instruments_block(instruments)
    content = SETTINGS_PATH.read_text(encoding="utf-8")
    pattern = re.compile(r"INSTRUMENTS\s*=\s*\[[\s\S]*?\]", re.MULTILINE)
    new_content, count = pattern.subn(serialized_block, content)
    if count == 0:
        new_content = content.rstrip() + "\n\n" + serialized_block + "\n"
    SETTINGS_PATH.write_text(new_content, encoding="utf-8")


def main() -> None:
    load_env_local()

    token = os.environ.get("INVEST_TOKEN")
    if not token:
        raise RuntimeError("Set INVEST_TOKEN environment variable before running.")

    with Client(token) as client:
        generated, missing = build_instruments(client)

    instruments = STATIC_INSTRUMENTS + sorted(generated, key=lambda item: item["ticker"])
    update_settings_file(instruments)

    print(f"Updated {SETTINGS_PATH.name} with {len(instruments)} instruments.")
    if missing:
        print("Tickers not found and skipped:", ", ".join(missing))


if __name__ == "__main__":
    main()

