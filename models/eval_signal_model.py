"""
Оценка сохранённой модели сигналов buy/sell (бинарно) на тестовом срезе.
Берёт параметры обучения из meta, загружает модель/скейлер и считает метрики.
"""
import argparse
import json
from pathlib import Path
import sys
from typing import List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
import settings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "tickers"
DEFAULT_META = PROJECT_ROOT / "models" / "signal_model_meta.json"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "signal_model.keras"
DEFAULT_SCALER = PROJECT_ROOT / "models" / "signal_model_scaler.pkl"
DEFAULT_METRICS_OUT = PROJECT_ROOT / "models" / "eval_signal_metrics.json"
DEFAULT_CM_OUT = PROJECT_ROOT / "models" / "eval_signal_confusion.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Тестирование сохранённой модели сигналов buy/sell (бинарно)"
    )
    parser.add_argument("--meta-path", type=Path, default=DEFAULT_META, help="Путь к meta JSON")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL, help="Путь к модели")
    parser.add_argument("--scaler-path", type=Path, default=DEFAULT_SCALER, help="Путь к scaler.pkl")
    parser.add_argument(
        "--metrics-path", type=Path, default=DEFAULT_METRICS_OUT, help="Куда сохранить метрики JSON"
    )
    parser.add_argument(
        "--cm-path", type=Path, default=DEFAULT_CM_OUT, help="Куда сохранить confusion matrix PNG"
    )
    parser.add_argument(
        "--timeframe",
        default=None,
        help="Переопределить таймфрейм (если не задан — берётся из meta)",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Список тикеров для теста (если не задан — берутся из meta)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Ограничить строки для быстрого прогона (хвост)",
    )
    return parser.parse_args()


def load_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        raise SystemExit(f"Не найден meta-файл {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_ticker_df(ticker: str, timeframe: str, max_rows: int | None) -> pd.DataFrame:
    """Читаем parquet выбранного таймфрейма для тикера."""
    file_path = DATA_DIR / ticker / f"{timeframe}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Нет файла {file_path}")

    df = pd.read_parquet(file_path)
    df = df.sort_index()
    if max_rows:
        df = df.tail(max_rows)

    df["ticker"] = ticker
    df.index = pd.to_datetime(df.index)
    return df


def concat_panel(tickers: Sequence[str], timeframe: str, max_rows: int | None) -> pd.DataFrame:
    """Объединяем данные всех тикеров в один датафрейм."""
    frames = []
    for ticker in tickers:
        try:
            frames.append(load_ticker_df(ticker, timeframe, max_rows))
        except FileNotFoundError as exc:
            print(f"⚠️ Пропускаем {ticker}: {exc}")

    if not frames:
        raise SystemExit("Не удалось загрузить ни один тикер для оценки.")

    df = pd.concat(frames).sort_index()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    ticker_to_id = {name: idx for idx, name in enumerate(sorted(tickers))}
    df["ticker_id"] = df["ticker"].map(ticker_to_id)
    return df


def add_labels(df: pd.DataFrame, horizon: int, threshold: float) -> pd.DataFrame:
    """Бинарная разметка: 0 = sell, 1 = buy; нейтральные выбрасываем."""
    df = df.copy()
    df["future_close"] = df["close"].shift(-horizon)
    df["future_ret"] = (df["future_close"] - df["close"]) / df["close"]
    mask_buy = df["future_ret"] > threshold
    mask_sell = df["future_ret"] < -threshold
    df = df[mask_buy | mask_sell]
    df["signal"] = np.where(df["future_ret"] > threshold, 1, 0)
    df = df.dropna(subset=["future_ret", "signal"])
    if df.empty:
        raise SystemExit("После фильтрации по threshold данные пусты. Уменьшите threshold или возьмите больше строк.")
    return df


def build_sequences(
    feature_values: np.ndarray, labels: np.ndarray, lookback: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Строим окна длины lookback, метка относится к последней свече окна."""
    sequences: list[np.ndarray] = []
    seq_labels: list[int] = []

    for end_idx in range(lookback - 1, len(feature_values)):
        label = labels[end_idx]
        if np.isnan(label):
            continue
        window = feature_values[end_idx - lookback + 1 : end_idx + 1]
        if np.any(np.isnan(window)):
            continue
        sequences.append(window)
        seq_labels.append(int(label))

    if not sequences:
        raise SystemExit("После построения окон нет данных для оценки.")

    X = np.asarray(sequences, dtype=np.float32)
    y = np.asarray(seq_labels, dtype=np.int64)
    return X, y


def chronological_split(
    X: np.ndarray, y: np.ndarray, val_size: float, test_size: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Тот же сплит, что и в обучении: train -> val -> test."""
    if val_size + test_size >= 1:
        raise SystemExit("Сумма val_size и test_size должна быть < 1.")

    n = len(X)
    train_end = int(n * (1 - val_size - test_size))
    val_end = int(n * (1 - test_size))
    train_end = max(train_end, 1)
    val_end = max(val_end, train_end + 1)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    if len(X_test) == 0:
        raise SystemExit("Пустой тестовый набор после разбиения.")

    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_scaler(X: np.ndarray, scaler) -> np.ndarray:
    """Применяем сохранённый StandardScaler к тензору X."""
    flat = X.reshape(len(X), -1)
    scaled = scaler.transform(flat)
    return scaled.reshape(X.shape)


def plot_confusion(cm: np.ndarray, class_names: list[str], out_path: Path) -> None:
    """Рисуем confusion matrix и сохраняем в PNG."""
    plt.figure(figsize=(5, 4))
    im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = "d"
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("Истина")
    plt.xlabel("Предсказание")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def collect_feature_columns(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Берём только нужные признаки и сигнал."""
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"В данных отсутствуют признаки: {missing}")
    features = df[feature_cols].to_numpy(dtype=np.float32)
    labels = df["signal"].to_numpy(dtype=np.float32)
    return features, labels


def main() -> None:
    args = parse_args()

    meta = load_meta(args.meta_path)
    feature_cols = meta["feature_columns"]
    class_mapping = meta["class_mapping"]
    timeframe = args.timeframe or meta["timeframe"]
    tickers = args.tickers or meta["tickers"]
    lookback = int(meta["lookback"])
    horizon = int(meta["horizon"])
    threshold = float(meta["threshold"])
    val_size = float(meta["val_size"])
    test_size = float(meta.get("test_size", 0.1))

    print(f"📥 Загружаем данные для оценки: {len(tickers)} тикеров, tf={timeframe}")
    df = concat_panel(tickers, timeframe, args.max_rows)
    df = add_labels(df, horizon=horizon, threshold=threshold)

    features, labels = collect_feature_columns(df, feature_cols)
    X, y = build_sequences(features, labels, lookback=lookback)
    _, _, X_test, _, _, y_test = chronological_split(X, y, val_size=val_size, test_size=test_size)

    scaler = joblib.load(args.scaler_path)
    X_test = apply_scaler(X_test, scaler)

    from keras import models  # импорт здесь, чтобы не грузить заранее, если не нужно

    model = models.load_model(args.model_path)
    probs = model.predict(X_test, verbose=0)
    preds = probs.argmax(axis=1)

    class_names = [class_mapping[str(i)] if isinstance(class_mapping, dict) else class_mapping[i] for i in range(len(class_mapping))]
    report = classification_report(
        y_test,
        preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, preds, labels=list(range(len(class_names))))

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"test_report": report, "test_size": len(y_test)}, f, ensure_ascii=False, indent=2)

    plot_confusion(cm, class_names, args.cm_path)

    print(f"✅ Тестирование завершено. Тестовых сэмплов: {len(y_test)}")
    buy_key = "buy" if "buy" in report else class_names[-1]
    sell_key = "sell" if "sell" in report else class_names[0]
    print(
        f"🎯 Test accuracy: {report['accuracy']:.3f} | "
        f"buy F1: {report[buy_key]['f1-score']:.3f}, "
        f"sell F1: {report[sell_key]['f1-score']:.3f}"
    )
    print(f"📂 Метрики сохранены в {args.metrics_path}")
    print(f"🖼️  Confusion matrix сохранена в {args.cm_path}")


if __name__ == "__main__":
    main()

