import os
import time
import sys
from pathlib import Path
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import timedelta
from grpc import StatusCode, RpcError

from tinkoff.invest import Client, CandleInterval
from tinkoff.invest.utils import now

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import settings


# === КОНФИГУРАЦИЯ ===

TIMEFRAMES = {
    "M5":  {"interval": CandleInterval.CANDLE_INTERVAL_5_MIN,  "days_back": 30},
    "M15": {"interval": CandleInterval.CANDLE_INTERVAL_15_MIN, "days_back": 60},
    "M30": {"interval": CandleInterval.CANDLE_INTERVAL_30_MIN, "days_back": 120},
    "H1":  {"interval": CandleInterval.CANDLE_INTERVAL_HOUR,   "days_back": 365},
    "H2":  {"interval": CandleInterval.CANDLE_INTERVAL_2_HOUR, "days_back": 730},
    "H4":  {"interval": CandleInterval.CANDLE_INTERVAL_4_HOUR, "days_back": 730},
    "D1":  {"interval": CandleInterval.CANDLE_INTERVAL_DAY,    "days_back": 3650},
    "Week": {"interval": CandleInterval.CANDLE_INTERVAL_WEEK,   "days_back": 3650},
}

DATA_DIR = "data/tickers"
os.makedirs(DATA_DIR, exist_ok=True)

# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===

def cast_money(v):
    return v.units + v.nano / 1e9

def calculate_indicators(df):
    """
    Расчет индикаторов.
    ВАЖНО: Если данных мало, расчет пропускается, чтобы избежать KeyError.
    """
    # Если строк меньше 50, мы не можем посчитать SMA50 и многие другие индикаторы корректно.
    if len(df) < 50:
        return df

    try:
        # 1. Трендовые
        df["sma5"] = ta.sma(df["close"], length=5)
        df["sma10"] = ta.sma(df["close"], length=10)
        df["sma20"] = ta.sma(df["close"], length=20)
        df["sma50"] = ta.sma(df["close"], length=50)
        
        df["ema10"] = ta.ema(df["close"], length=10)
        df["ema20"] = ta.ema(df["close"], length=20)
        df["ema50"] = ta.ema(df["close"], length=50)
        if len(df) >= 200:
            df["ema200"] = ta.ema(df["close"], length=200)

        # MACD
        macd = ta.macd(df["close"])
        if macd is not None:
            df["macd"] = macd["MACD_12_26_9"]
            df["macd_signal"] = macd["MACDs_12_26_9"]
            df["macd_hist"] = macd["MACDh_12_26_9"]

        # 2. Осцилляторы
        df["rsi"] = ta.rsi(df["close"], length=14)
        
        stoch = ta.stoch(df["high"], df["low"], df["close"])
        if stoch is not None:
            df["stoch_k"] = stoch["STOCHk_14_3_3"]
            df["stoch_d"] = stoch["STOCHd_14_3_3"]
        
        df["cci"] = ta.cci(df["high"], df["low"], df["close"])
        df["willr"] = ta.willr(df["high"], df["low"], df["close"])

        # 3. Волатильность
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        
        bb = ta.bbands(df["close"], length=20)
        if bb is not None:
            # Используем .get() или проверяем колонки, чтобы не падать
            # pandas_ta обычно возвращает BBU_20_2.0, BBL_20_2.0, BBM_20_2.0
            # Мы берем по индексам, так надежнее, если версия либы изменит имена
            df["bb_lower"] = bb.iloc[:, 0]  
            df["bb_middle"] = bb.iloc[:, 1] 
            df["bb_upper"] = bb.iloc[:, 2] 
            
            # Позиция цены в канале BB (защита от деления на ноль)
            denom = (df["bb_upper"] - df["bb_lower"])
            df["bb_pband"] = np.where(denom != 0, (df["close"] - df["bb_lower"]) / denom, 0)

        # 4. Объемы
        df["vol_roc"] = df["volume"].pct_change()
        df["vol_sma20"] = ta.sma(df["volume"], length=20)
        
        # Защита от деления на ноль для vol_rel
        df["vol_rel"] = np.where(df["vol_sma20"] > 0, df["volume"] / df["vol_sma20"], 1)

        # 5. Свечные паттерны
        try:
            # Doji
            if hasattr(ta, "cdl_doji"):
                df["pat_doji"] = ta.cdl_doji(df["open"], df["high"], df["low"], df["close"]) / 100
            else:
                 df["pat_doji"] = 0

            # Hammer
            if hasattr(ta, "cdl_hammer"):
                df["pat_hammer"] = ta.cdl_hammer(df["open"], df["high"], df["low"], df["close"]) / 100
            else:
                 df["pat_hammer"] = 0
            
            # Engulfing
            if hasattr(ta, "cdl_engulfing"):
                df["pat_engulfing"] = ta.cdl_engulfing(df["open"], df["high"], df["low"], df["close"]) / 100
            else:
                 df["pat_engulfing"] = 0
                 
            df["pat_doji"] = df["pat_doji"].fillna(0)
            df["pat_hammer"] = df["pat_hammer"].fillna(0)
            df["pat_engulfing"] = df["pat_engulfing"].fillna(0)

        except Exception as e:
            # print(f"Warning: Candle patterns calculation failed: {e}")
            df["pat_doji"] = 0
            df["pat_hammer"] = 0
            df["pat_engulfing"] = 0

        # --- РЕЖИМЫ ---
        df["atr_sma50"] = ta.sma(df["atr"], length=50)
        
        # np.where безопаснее прямых сравнений с NaN
        df["regime_vol"] = np.where(df["atr"] > df["atr_sma50"], 1, 0)

        adx = ta.adx(df["high"], df["low"], df["close"])
        if adx is not None:
            df["adx"] = adx["ADX_14"]
            df["regime_trend"] = np.where(df["adx"] > 25, 1, 0)
        else:
            df["regime_trend"] = 0

        df["regime_liq"] = np.where(df["volume"] > df["vol_sma20"], 1, 0)
        df["market_regime"] = (df["regime_vol"] * 100) + (df["regime_trend"] * 10) + df["regime_liq"]
        
    except Exception as e:
        print(f"⚠️ Ошибка расчета индикаторов: {e}")
        # Возвращаем DF без новых колонок, но не крашим программу
        return df

    return df

def clean_dataframe(df):
    df = df[~df.index.duplicated(keep='first')]
    df = df[df["volume"] > 0]
    df = df[df["close"] > 0]
    df = df[~df.index.weekday.isin([5, 6])]
    # Удаляем строки, где всё NaN (например, если индикаторы не посчитались в начале)
    df = df.dropna(how='all') 
    df = df.sort_index()
    return df

def get_candles_with_retry(client, figi, from_, interval, max_retries=5):
    """
    Обертка с повторными попытками при ошибке RESOURCE_EXHAUSTED (Синхронная версия)
    """
    attempt = 0
    base_delay = 5 # секунд
    
    while attempt < max_retries:
        try:
            candles = []
            # Используем синхронный генератор
            for candle in client.get_all_candles(figi=figi, from_=from_, interval=interval):
                candles.append({
                    "time": candle.time,
                    "open": cast_money(candle.open),
                    "high": cast_money(candle.high),
                    "low": cast_money(candle.low),
                    "close": cast_money(candle.close),
                    "volume": candle.volume
                })
            return candles
        except RpcError as e:
            if e.code() == StatusCode.RESOURCE_EXHAUSTED:
                attempt += 1
                wait_time = base_delay * attempt + np.random.uniform(0, 1) # Экспоненциальная задержка + джиттер
                print(f"⏳ Лимит запросов (429/Exhausted) для {figi}. Ждем {wait_time:.1f} сек... (Попытка {attempt}/{max_retries})")
                time.sleep(wait_time)
            else:
                # Если ошибка другая (не лимиты), пробрасываем её
                print(f"❌ Критическая ошибка API: {e}")
                return []
        except Exception as e:
             print(f"❌ Неизвестная ошибка: {e}")
             return []
    
    print(f"❌ Не удалось получить данные после {max_retries} попыток.")
    return []

def process_instrument(client, instrument):
    ticker = instrument["name"]
    figi = instrument["figi"]
    ticker_dir = os.path.join(DATA_DIR, ticker)
    os.makedirs(ticker_dir, exist_ok=True)

    for tf_name, tf_params in TIMEFRAMES.items():
        # == RATE LIMITER == 
        # Небольшая пауза перед каждым запросом, чтобы сгладить пики
        print(f"⬇️ {ticker} | {tf_name} loading...")
        time.sleep(0.5) # Увеличили паузу для надежности
        
        _from = now() - timedelta(days=tf_params["days_back"])
        
        # Используем безопасную функцию загрузки
        candles = get_candles_with_retry(client, figi, _from, tf_params["interval"])

        if not candles:
            print(f"⚠️ {ticker} | {tf_name}: Нет данных")
            continue

        df = pd.DataFrame(candles)
        df = df.set_index("time")
        df = clean_dataframe(df)

        if df.empty:
            print(f"⚠️ {ticker} | {tf_name}: Пустой датафрейм после очистки")
            continue
        
        # Расчет индикаторов с защитой от ошибок
        df = calculate_indicators(df)
        
        # Удаляем NaN в начале (появившиеся из-за window functions типа SMA50)
        df = df.dropna()

        if not df.empty:
            file_path = os.path.join(ticker_dir, f"{tf_name}.parquet")
            df.to_parquet(file_path, compression='snappy')
            print(f"✅ {ticker} | {tf_name} saved")
        else:
            print(f"⚠️ {ticker} | {tf_name}: Пустой датафрейм после индикаторов")

def check_missing_files():
    """Проверка загруженных файлов"""
    print("\n🔍 Проверка целостности данных...")
    missing = []
    
    for instrument in settings.INSTRUMENTS:
        ticker = instrument["name"]
        ticker_dir = os.path.join(DATA_DIR, ticker)
        
        for tf_name in TIMEFRAMES.keys():
            file_path = os.path.join(ticker_dir, f"{tf_name}.parquet")
            if not os.path.exists(file_path):
                missing.append(f"{ticker} - {tf_name}")
    
    if missing:
        print(f"❌ Отсутствуют файлы ({len(missing)} шт.):")
        for m in missing:
            print(f"  - {m}")
    else:
        print("✅ Все файлы успешно загружены!")

def main():
    token = settings.INVEST_TOKEN
    
    # Синхронный клиент
    with Client(token) as client:
        print(f"🚀 Старт загрузки для {len(settings.INSTRUMENTS)} тикеров (Sync Mode)...")
        
        for instrument in settings.INSTRUMENTS:
            process_instrument(client, instrument)
        
    print("\n🏁 Загрузка и обработка завершены!")
    check_missing_files()


if __name__ == "__main__":
    main()
