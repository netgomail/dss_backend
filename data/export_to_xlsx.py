from pathlib import Path

import pandas as pd


def main() -> None:
    # Читаем исходный parquet и берем только первые 50 строк
    data_dir = Path(__file__).resolve().parent
    src_path = data_dir / "tickers" / "SBER" / "M30.parquet"
    df = pd.read_parquet(src_path).head(50)
    # Индекс parquet называется time, поэтому сбрасываем его в колонку
    df = df.reset_index()
    # Excel не принимает таймзоны — убираем tz в колонке time
    if "time" in df.columns and pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = df["time"].dt.tz_localize(None)

    # Сохраняем выборку в Excel в ту же папку
    out_path = data_dir / "SBER_M30.xlsx"
    df.to_excel(out_path, index=False)


if __name__ == "__main__":
    main()
