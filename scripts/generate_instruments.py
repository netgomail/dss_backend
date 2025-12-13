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

from tinkoff.invest import Client
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
PREFERRED_CLASS_CODES = [
    "TQBR",
    "TQTF",
    "TQTD",
    "TQTE",
    "TQIF",
    "TQPI",
    "TQBD",
    "EQNE",
    "SPBXM",
    "SPBB",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = PROJECT_ROOT / "settings.py"
ENV_PATH = PROJECT_ROOT / ".env.local"


def fetch_shares(client: Client) -> Dict[str, object]:
    """
    Fetch all base shares once and index by ticker with class-code priority.
    
    Returns:
        Словарь {ticker: instrument} для акций
    """
    response = client.instruments.shares(
        instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE
    )

    by_ticker: Dict[str, List[object]] = defaultdict(list)
    for share in response.instruments:
        by_ticker[share.ticker].append(share)

    def select_preferred(shares: List[object]) -> object:
        def weight(s: object) -> Tuple[int, str]:
            try:
                idx = PREFERRED_CLASS_CODES.index(s.class_code)
            except ValueError:
                idx = len(PREFERRED_CLASS_CODES)
            return (idx, s.class_code)

        return sorted(shares, key=weight)[0]

    return {ticker: select_preferred(shares) for ticker, shares in by_ticker.items()}


def fetch_futures(client: Client) -> Dict[str, object]:
    """
    Fetch all base futures and index by ticker.
    
    Фьючерсы имеют тикеры вида IMOEXF, SiZ4 и т.д.
    Возвращает только активные (не истёкшие) фьючерсы.
    
    Returns:
        Словарь {ticker: instrument} для фьючерсов
    """
    response = client.instruments.futures(
        instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE
    )

    by_ticker: Dict[str, List[object]] = defaultdict(list)
    for future in response.instruments:
        # Используем basic_asset_short_name или ticker
        by_ticker[future.ticker].append(future)

    # Для фьючерсов с одинаковым тикером выбираем ближайший по дате экспирации
    def select_nearest(futures: List[object]) -> object:
        # Сортируем по дате экспирации (ближайший первым)
        return sorted(futures, key=lambda f: f.expiration_date)[0]

    return {ticker: select_nearest(futures) for ticker, futures in by_ticker.items()}


def build_instruments(client: Client) -> Tuple[List[dict], List[str]]:
    """
    Prepare instrument dicts for configured assets and track missing tickers.
    
    Ищет инструменты сначала среди акций (shares), затем среди фьючерсов (futures).
    Это позволяет добавлять как акции (SBER, GAZP), так и фьючерсы (IMOEXF, SiZ4).
    
    Returns:
        Tuple[List[dict], List[str]]: (список инструментов, список ненайденных тикеров)
    """
    # Загружаем все доступные акции и фьючерсы
    share_by_ticker = fetch_shares(client)
    future_by_ticker = fetch_futures(client)
    
    instruments: List[dict] = []
    missing: List[str] = []

    for ticker in ASSETS:
        # Сначала ищем среди акций
        share = share_by_ticker.get(ticker)
        if share:
            instruments.append(
                {
                    "name": share.ticker,
                    "alias": share.name,
                    "figi": share.figi,
                    "full_name": share.name,
                    "class_code": share.class_code,
                }
            )
            continue
        
        # Если не нашли среди акций, ищем среди фьючерсов
        future = future_by_ticker.get(ticker)
        if future:
            instruments.append(
                {
                    "name": future.ticker,
                    "alias": future.name,
                    "figi": future.figi,
                    "future": True,  # Флаг, что это фьючерс
                    "full_name": future.name,
                    "class_code": future.class_code,
                }
            )
            continue
        
        # Не нашли ни среди акций, ни среди фьючерсов
        missing.append(ticker)

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
        "name",
        "alias",
        "figi",
        "future",
        "full_name",
        "class_code",
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

    instruments = STATIC_INSTRUMENTS + sorted(generated, key=lambda item: item["name"])
    update_settings_file(instruments)

    print(f"Updated {SETTINGS_PATH.name} with {len(instruments)} instruments.")
    if missing:
        print("Tickers not found and skipped:", ", ".join(missing))


if __name__ == "__main__":
    main()

