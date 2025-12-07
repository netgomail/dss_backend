"""Async fetch of OHLCV candles for all instruments in settings.INSTRUMENTS (real contour).

Outputs Parquet files to data/tickers/<TICKER>/<INTERVAL>.parquet
with columns: time, open, high, low, close, volume, is_complete.

Intervals fetched by default:
    M1, M2, M3, M5, M10, M15, M30,
    H1, H2, H4,
    D1, W1, MO1

Date range:
    From START_DATE (2015-01-01 UTC) to now().

Tokens:
    - .env.local is loaded automatically (INVEST_TOKEN).

Run:
    python scripts/fetch_ohlcv.py
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import Iterable, List, Tuple

import pandas as pd
from tinkoff.invest import CandleInterval, AsyncClient
from tinkoff.invest.constants import INVEST_GRPC_API
from tinkoff.invest.exceptions import AioRequestError
from tinkoff.invest.utils import now

# ensure project root on sys.path to import settings when run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import settings

ENV_PATH = PROJECT_ROOT / ".env.local"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "tickers"

START_DATE = datetime(2015, 1, 1, tzinfo=timezone.utc)

INTERVALS: List[Tuple[str, CandleInterval, timedelta]] = [
    ("M1", CandleInterval.CANDLE_INTERVAL_1_MIN, timedelta(days=1)),
    ("M2", CandleInterval.CANDLE_INTERVAL_2_MIN, timedelta(days=1)),
    ("M3", CandleInterval.CANDLE_INTERVAL_3_MIN, timedelta(days=1)),
    ("M5", CandleInterval.CANDLE_INTERVAL_5_MIN, timedelta(days=3)),
    ("M10", CandleInterval.CANDLE_INTERVAL_10_MIN, timedelta(days=7)),
    ("M15", CandleInterval.CANDLE_INTERVAL_15_MIN, timedelta(days=7)),
    ("M30", CandleInterval.CANDLE_INTERVAL_30_MIN, timedelta(days=14)),
    ("H1", CandleInterval.CANDLE_INTERVAL_HOUR, timedelta(days=90)),
    ("H2", CandleInterval.CANDLE_INTERVAL_2_HOUR, timedelta(days=180)),
    ("H4", CandleInterval.CANDLE_INTERVAL_4_HOUR, timedelta(days=365)),
    ("D1", CandleInterval.CANDLE_INTERVAL_DAY, timedelta(days=365 * 3)),
    ("Week", CandleInterval.CANDLE_INTERVAL_WEEK, timedelta(days=365 * 6)),
    ("Month", CandleInterval.CANDLE_INTERVAL_MONTH, timedelta(days=365 * 10)),
]


def load_env_local(env_path: Path = ENV_PATH) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def price_to_float(price) -> float:
    return price.units + price.nano / 1_000_000_000


def daterange_batches(start: datetime, end: datetime, step: timedelta) -> Iterable[Tuple[datetime, datetime]]:
    cursor = start
    while cursor < end:
        nxt = min(cursor + step, end)
        yield cursor, nxt
        cursor = nxt


async def fetch_candles_for_interval(
    client: AsyncClient, figi: str, interval: CandleInterval, batch_step: timedelta, start: datetime, end: datetime
) -> List[dict]:
    rows: List[dict] = []
    for dt_from, dt_to in daterange_batches(start, end, batch_step):
        resp = await safe_get_candles(client, figi, interval, dt_from, dt_to)
        # small delay between batches to avoid hammering API
        await asyncio.sleep(0.2)
        for c in resp.candles:
            rows.append(
                {
                    "time": c.time.astimezone(timezone.utc),
                    "open": price_to_float(c.open),
                    "high": price_to_float(c.high),
                    "low": price_to_float(c.low),
                    "close": price_to_float(c.close),
                    "volume": c.volume,
                    "is_complete": c.is_complete,
                }
            )
    return rows


def ensure_output_path(ticker: str, interval_tag: str) -> Path:
    path = OUTPUT_ROOT / ticker / f"{interval_tag}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def pick_token_and_target() -> tuple[str, str]:
    token = os.environ.get("INVEST_TOKEN") or getattr(settings, "INVEST_TOKEN", None)
    if not token:
        raise RuntimeError("Token not found. Set INVEST_TOKEN in .env.local or settings.py.")
    target = INVEST_GRPC_API
    return token, target


async def safe_get_candles(client: AsyncClient, figi: str, interval: CandleInterval, dt_from: datetime, dt_to: datetime):
    """Get candles with retry/backoff for rate limits."""
    max_retries = 8
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            return await client.market_data.get_candles(
                figi=figi,
                from_=dt_from,
                to=dt_to,
                interval=interval,
            )
        except AioRequestError as e:
            metadata = getattr(e, "metadata", None)
            reset = None
            if metadata and getattr(metadata, "ratelimit_reset", None):
                try:
                    reset = int(str(metadata.ratelimit_reset).split(",")[0])
                except Exception:
                    reset = None
            sleep_for = reset if reset is not None else backoff
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(max(sleep_for, 1))
            backoff = min(backoff * 2, 60)


async def fetch_for_ticker(client: AsyncClient, ticker: str, figi: str) -> None:
    end_dt = now()
    start_dt = START_DATE
    for interval_tag, interval, step in INTERVALS:
        rows = await fetch_candles_for_interval(client, figi, interval, step, start_dt, end_dt)
        if not rows:
            print(f"{ticker} {interval_tag}: no data")
            continue
        df = pd.DataFrame(rows)
        outfile = ensure_output_path(ticker, interval_tag)
        df.to_parquet(outfile, index=False)
        print(f"{ticker} {interval_tag}: saved {len(df)} rows to {outfile.relative_to(PROJECT_ROOT)}")


async def main_async() -> None:
    load_env_local()
    token, target = pick_token_and_target()

    # Limit simultaneous tickers to reduce rate limits
    semaphore = asyncio.Semaphore(1)

    async with AsyncClient(token, target=target) as client:
        async def run_one(instr: dict):
            async with semaphore:
                await fetch_for_ticker(client, instr["name"], instr.get("figi"))

        tasks = [
            run_one(instr)
            for instr in settings.INSTRUMENTS
            if instr.get("figi")
        ]
        if not tasks:
            print("No instruments with figi found.")
            return
        await asyncio.gather(*tasks)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

