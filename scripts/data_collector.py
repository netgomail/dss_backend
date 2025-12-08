import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
from grpc import RpcError, StatusCode
from tinkoff.invest import CandleInterval, Client
from tinkoff.invest.utils import now

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import settings


# === КОНФИГУРАЦИЯ ===

@dataclass(frozen=True)
class TimeframeConfig:
    interval: CandleInterval
    days_back: int


TIMEFRAMES: Dict[str, TimeframeConfig] = {
    "M5": TimeframeConfig(CandleInterval.CANDLE_INTERVAL_5_MIN, 30),
    "M15": TimeframeConfig(CandleInterval.CANDLE_INTERVAL_15_MIN, 60),
    "M30": TimeframeConfig(CandleInterval.CANDLE_INTERVAL_30_MIN, 120),
    "H1": TimeframeConfig(CandleInterval.CANDLE_INTERVAL_HOUR, 365),
    "H2": TimeframeConfig(CandleInterval.CANDLE_INTERVAL_2_HOUR, 730),
    "H4": TimeframeConfig(CandleInterval.CANDLE_INTERVAL_4_HOUR, 730),
    "D1": TimeframeConfig(CandleInterval.CANDLE_INTERVAL_DAY, 3650),
    "Week": TimeframeConfig(CandleInterval.CANDLE_INTERVAL_WEEK, 3650),
}

DATA_DIR = PROJECT_ROOT / "data" / "tickers"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MIN_HISTORY_ROWS = 50
MAX_WORKERS = 1  # ограничиваем параллелизм, чтобы не упираться в лимиты
BASE_COLUMNS = ["open", "high", "low", "close", "volume"]
REQUEST_PAUSE_SECONDS = 1.0  # базовая пауза между запросами таймфреймов
REQUEST_JITTER_SECONDS = 0.3  # джиттер, чтобы разнести запросы по времени

try:
    import pyarrow  # type: ignore  # noqa: F401

    PARQUET_ENGINE: Optional[str] = "pyarrow"
except ImportError:
    try:
        import fastparquet  # type: ignore  # noqa: F401

        PARQUET_ENGINE = "fastparquet"
    except ImportError:
        PARQUET_ENGINE = None

# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===

def cast_money(v) -> float:
    return v.units + v.nano / 1e9


def has_sufficient_history(df: pd.DataFrame, min_length: int = MIN_HISTORY_ROWS) -> bool:
    """Check whether there is enough history to compute long-window indicators."""
    return len(df) >= min_length


def safe_candle_pattern(fn_name: str, df: pd.DataFrame) -> pd.Series:
    """Safely compute candle pattern; return zeros if function is missing or fails."""
    pattern_fn = getattr(ta, fn_name, None)
    if pattern_fn is None:
        return pd.Series(0, index=df.index)

    try:
        series = pattern_fn(df["open"], df["high"], df["low"], df["close"])
        if series is None:
            return pd.Series(0, index=df.index)
        return series.fillna(0) / 100
    except Exception:
        return pd.Series(0, index=df.index)


def add_trend_indicators(df: pd.DataFrame) -> None:
    df["sma5"] = ta.sma(df["close"], length=5)
    df["sma10"] = ta.sma(df["close"], length=10)
    df["sma20"] = ta.sma(df["close"], length=20)
    df["sma50"] = ta.sma(df["close"], length=50)

    df["ema10"] = ta.ema(df["close"], length=10)
    df["ema20"] = ta.ema(df["close"], length=20)
    df["ema50"] = ta.ema(df["close"], length=50)
    if len(df) >= 200:
        df["ema200"] = ta.ema(df["close"], length=200)

    macd = ta.macd(df["close"])
    if macd is not None and not macd.empty:
        df["macd"] = macd.get("MACD_12_26_9", macd.iloc[:, 0])
        df["macd_signal"] = macd.get("MACDs_12_26_9", macd.iloc[:, 1])
        df["macd_hist"] = macd.get("MACDh_12_26_9", macd.iloc[:, 2])


def add_oscillator_indicators(df: pd.DataFrame) -> None:
    df["rsi"] = ta.rsi(df["close"], length=14)

    stoch = ta.stoch(df["high"], df["low"], df["close"])
    if stoch is not None and not stoch.empty:
        df["stoch_k"] = stoch.get("STOCHk_14_3_3", stoch.iloc[:, 0])
        df["stoch_d"] = stoch.get("STOCHd_14_3_3", stoch.iloc[:, 1])

    df["cci"] = ta.cci(df["high"], df["low"], df["close"])
    df["willr"] = ta.willr(df["high"], df["low"], df["close"])


def add_volatility_indicators(df: pd.DataFrame) -> None:
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    bb = ta.bbands(df["close"], length=20)
    if bb is not None and not bb.empty:
        df["bb_lower"] = bb.iloc[:, 0]
        df["bb_middle"] = bb.iloc[:, 1]
        df["bb_upper"] = bb.iloc[:, 2]

        denom = df["bb_upper"] - df["bb_lower"]
        df["bb_pband"] = np.where(denom != 0, (df["close"] - df["bb_lower"]) / denom, 0)


def add_volume_indicators(df: pd.DataFrame) -> None:
    df["vol_roc"] = df["volume"].pct_change()
    df["vol_sma20"] = ta.sma(df["volume"], length=20)
    df["vol_rel"] = np.where(df["vol_sma20"] > 0, df["volume"] / df["vol_sma20"], 1)


def add_pattern_indicators(df: pd.DataFrame) -> None:
    df["pat_doji"] = safe_candle_pattern("cdl_doji", df)
    df["pat_hammer"] = safe_candle_pattern("cdl_hammer", df)
    df["pat_engulfing"] = safe_candle_pattern("cdl_engulfing", df)


def add_regime_features(df: pd.DataFrame) -> None:
    df["atr_sma50"] = ta.sma(df["atr"], length=50)
    df["regime_vol"] = np.where(df["atr"] > df["atr_sma50"], 1, 0)

    adx = ta.adx(df["high"], df["low"], df["close"])
    if adx is not None and not adx.empty:
        df["adx"] = adx.get("ADX_14", adx.iloc[:, 0])
        df["regime_trend"] = np.where(df["adx"] > 25, 1, 0)
    else:
        df["regime_trend"] = 0

    df["regime_liq"] = np.where(df["volume"] > df["vol_sma20"], 1, 0)
    df["market_regime"] = (df["regime_vol"] * 100) + (df["regime_trend"] * 10) + df["regime_liq"]


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Расчет индикаторов.
    ВАЖНО: Если данных мало, расчет пропускается, чтобы избежать KeyError.
    """
    if not has_sufficient_history(df):
        return df

    try:
        add_trend_indicators(df)
        add_oscillator_indicators(df)
        add_volatility_indicators(df)
        add_volume_indicators(df)
        add_pattern_indicators(df)
        add_regime_features(df)
    except Exception as e:
        print(f"⚠️ Ошибка расчета индикаторов: {e}")
        return df

    return df


def parquet_engine_or_warn() -> Optional[str]:
    """Return parquet engine or warn once per process."""
    if PARQUET_ENGINE is None:
        print("❌ Нет доступного движка Parquet (pyarrow/fastparquet). Установите pyarrow.")
        return None
    return PARQUET_ENGINE


def load_existing_base(file_path: Path, engine: str) -> Optional[pd.DataFrame]:
    """Прочитать существующий parquet и оставить только базовые столбцы цен/объема."""
    if not file_path.exists():
        return None
    try:
        df = pd.read_parquet(file_path, engine=engine)
        return df[BASE_COLUMNS]
    except Exception as exc:
        print(f"⚠️ Не удалось прочитать {file_path.name}: {exc}")
        return None


def start_from_timestamp(existing: Optional[pd.DataFrame], tf_config: TimeframeConfig):
    """Определить точку старта выгрузки: хвост +1 сек или исторический диапазон."""
    if existing is None or existing.empty:
        return now() - timedelta(days=tf_config.days_back)

    last_ts = existing.index.max()
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=now().tzinfo)
    return last_ts + timedelta(seconds=1)

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляем дубликаты, некорректные строки и упорядочиваем индекс."""
    df = df[~df.index.duplicated(keep="first")]
    df = df[df["volume"] > 0]
    df = df[df["close"] > 0]
    df = df[~df.index.weekday.isin([5, 6])]
    df = df.dropna(how="all")
    return df.sort_index()


def get_candles_with_retry(
    client: Client,
    figi: str,
    from_,
    interval: CandleInterval,
    max_retries: int = 5,
) -> List[dict]:
    """Синхронная загрузка свечей с повторными попытками при 429."""
    attempt = 0
    base_delay_seconds = 5

    while attempt < max_retries:
        try:
            return [
                {
                    "time": candle.time,
                    "open": cast_money(candle.open),
                    "high": cast_money(candle.high),
                    "low": cast_money(candle.low),
                    "close": cast_money(candle.close),
                    "volume": candle.volume,
                }
                for candle in client.get_all_candles(
                    figi=figi, from_=from_, interval=interval
                )
            ]
        except RpcError as e:
            if e.code() == StatusCode.RESOURCE_EXHAUSTED:
                attempt += 1
                wait_time = base_delay_seconds * attempt + np.random.uniform(0, 1)
                print(
                    f"⏳ Лимит запросов (429/Exhausted) для {figi}. "
                    f"Ждем {wait_time:.1f} сек... (Попытка {attempt}/{max_retries})"
                )
                time.sleep(wait_time)
                continue

            print(f"❌ Критическая ошибка API: {e}")
            return []
        except Exception as e:
            print(f"❌ Неизвестная ошибка: {e}")
            return []

    print(f"❌ Не удалось получить данные после {max_retries} попыток.")
    return []


def process_instrument(client: Client, instrument: Dict[str, str]) -> None:
    ticker = instrument["name"]
    figi = instrument["figi"]
    ticker_dir = DATA_DIR / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    engine = parquet_engine_or_warn()
    if engine is None:
        return

    for tf_name, tf_config in TIMEFRAMES.items():
        file_path = ticker_dir / f"{tf_name}.parquet"
        existing_base = load_existing_base(file_path, engine)

        print(f"⬇️ {ticker} | {tf_name} loading...")
        time.sleep(REQUEST_PAUSE_SECONDS + np.random.uniform(0, REQUEST_JITTER_SECONDS))

        _from = start_from_timestamp(existing_base, tf_config)
        candles = get_candles_with_retry(client, figi, _from, tf_config.interval)

        if not candles:
            print(f"⚠️ {ticker} | {tf_name}: Нет новых данных")
            continue

        df_new = pd.DataFrame(candles).set_index("time")
        df = df_new if existing_base is None else pd.concat([existing_base, df_new])
        df = clean_dataframe(df)

        if df.empty:
            print(f"⚠️ {ticker} | {tf_name}: Пустой датафрейм после очистки")
            continue

        previous_len = len(existing_base) if existing_base is not None else 0
        if len(df) <= previous_len:
            print(f"ℹ️ {ticker} | {tf_name}: Нет новых строк после объединения")
            continue

        df = calculate_indicators(df)
        df = df.dropna()

        if df.empty:
            print(f"⚠️ {ticker} | {tf_name}: Пустой датафрейм после индикаторов")
            continue

        df.to_parquet(file_path, compression="snappy", engine=engine)
        print(f"✅ {ticker} | {tf_name} saved")


def check_missing_files() -> None:
    """Проверка наличия всех выгруженных файлов."""
    print("\n🔍 Проверка целостности данных...")
    missing = []

    for instrument in settings.INSTRUMENTS:
        ticker = instrument["name"]
        ticker_dir = DATA_DIR / ticker

        for tf_name in TIMEFRAMES.keys():
            file_path = ticker_dir / f"{tf_name}.parquet"
            if not file_path.exists():
                missing.append(f"{ticker} - {tf_name}")

    if missing:
        print(f"❌ Отсутствуют файлы ({len(missing)} шт.):")
        for m in missing:
            print(f"  - {m}")
    else:
        print("✅ Все файлы успешно загружены!")


def run_all_instruments(client: Client) -> None:
    total = len(settings.INSTRUMENTS)
    workers = min(MAX_WORKERS, total) or 1
    print(f"🏎️ Параллельная загрузка: {total} тикеров, потоки: {workers}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_instrument, client, instrument): instrument["name"]
            for instrument in settings.INSTRUMENTS
        }

        for future in as_completed(futures):
            ticker = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"❌ Ошибка при обработке {ticker}: {exc}")


def main() -> None:
    token = settings.INVEST_TOKEN

    with Client(token) as client:
        print(f"🚀 Старт загрузки для {len(settings.INSTRUMENTS)} тикеров (Sync Mode)...")

        run_all_instruments(client)

    print("\n🏁 Загрузка и обработка завершены!")
    check_missing_files()


if __name__ == "__main__":
    main()
