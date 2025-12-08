"""
Тренировка простой LSTM-модели, которая выдаёт сигналы buy/sell/hold
по историческим OHLCV данным и сгенерированным индикаторам из data/tickers.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from typing import Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Указываем backend для Keras (ожидается tensorflow или torch)
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

try:
    from keras import callbacks, layers, models, optimizers
except ImportError as exc:  # pragma: no cover - явное сообщение об ошибке окружения
    raise SystemExit(
        "Не найден backend для Keras. Установите, например: pip install 'tensorflow>=2.16'"
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
import settings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "tickers"
MODELS_DIR = PROJECT_ROOT / "models"
# Бинарная схема классов: 0 = sell, 1 = buy
CLASS_NAMES = {0: "sell", 1: "buy"}
DEFAULT_TICKERS = [item["name"] for item in settings.INSTRUMENTS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обучение LSTM модели для сигналов по ценовым данным"
    )
    parser.add_argument(
        "--timeframe",
        default="M30",
        help="Имя parquet-файла таймфрейма (например H1, H4, D1)",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Список тикеров. Если не задан, берём все из settings.INSTRUMENTS",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=64,
        help="Длина окна истории (кол-во баров в последовательности)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=4,
        help="Горизонт прогноза в барах (через сколько баров оцениваем результат)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.003,
        help="Минимальная доходность для сигнала buy/sell (например 0.003 = 0.3%)",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Доля валидации (берётся с хвоста выборки для честности по времени)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Доля теста (самый свежий хвост для финальной оценки)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Кол-во эпох обучения",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Размер батча",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Скорость обучения Adam",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Ограничить число строк из каждого тикера (для быстрых тестов)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODELS_DIR / "signal_model.keras",
        help="Куда сохранить модель",
    )
    parser.add_argument(
        "--scaler-path",
        type=Path,
        default=MODELS_DIR / "signal_model_scaler.pkl",
        help="Куда сохранить StandardScaler",
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=MODELS_DIR / "signal_model_meta.json",
        help="Куда сохранить метаданные (фичи, мэппинг классов)",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=MODELS_DIR / "signal_model_learning_curve.png",
        help="Куда сохранить график кривых обучения (loss/accuracy)",
    )
    return parser.parse_args()


def load_ticker_df(ticker: str, timeframe: str, max_rows: int | None) -> pd.DataFrame:
    """Читаем parquet выбранного таймфрейма для тикера (сортируем и отмечаем тикер)."""
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
    """Объединяем данные всех тикеров в один датафрейм и чистим бесконечности/NaN."""
    frames = []
    for ticker in tickers:
        try:
            frames.append(load_ticker_df(ticker, timeframe, max_rows))
        except FileNotFoundError as exc:
            print(f"⚠️ Пропускаем {ticker}: {exc}")

    if not frames:
        raise SystemExit("Не удалось загрузить ни один тикер.")

    df = pd.concat(frames).sort_index()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    ticker_to_id = {name: idx for idx, name in enumerate(sorted(tickers))}
    df["ticker_id"] = df["ticker"].map(ticker_to_id)
    return df


def add_labels(df: pd.DataFrame, horizon: int, threshold: float) -> pd.DataFrame:
    """
    Добавляем столбец signal (бинарно):
    0 = sell, 1 = buy.
    Нейтральные значения (|ret| <= threshold) удаляются, чтобы не плодить hold.
    """
    df = df.copy()
    df["future_close"] = df["close"].shift(-horizon)
    df["future_ret"] = (df["future_close"] - df["close"]) / df["close"]
    # Маски для бай/селл; нейтральные выбрасываем
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
    sequences: List[np.ndarray] = []
    seq_labels: List[int] = []

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
        raise SystemExit("После построения окон нет данных для обучения.")

    X = np.asarray(sequences, dtype=np.float32)
    y = np.asarray(seq_labels, dtype=np.int64)
    return X, y


def chronological_split(
    X: np.ndarray, y: np.ndarray, val_size: float, test_size: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Честное разбиение по времени: train -> val -> test (самый свежий).
    """
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

    if len(X_val) == 0 or len(X_test) == 0:
        raise SystemExit("Слишком мало данных для val/test после разбиения.")

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_sequences(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, StandardScaler]:
    """Масштабируем признаки на основе train и возвращаем scaler (val/test через тот же масштаб)."""
    scaler = StandardScaler()
    flat_train = X_train.reshape(len(X_train), -1)
    scaler.fit(flat_train)

    def _transform(arr: np.ndarray) -> np.ndarray:
        flat = arr.reshape(len(arr), -1)
        return scaler.transform(flat).reshape(arr.shape)

    X_train_scaled = _transform(X_train)
    X_val_scaled = _transform(X_val)
    X_test_scaled = _transform(X_test) if X_test is not None else None
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def build_model(lookback: int, n_features: int, learning_rate: float) -> models.Model:
    """Простая двухслойная LSTM с dropout под бинарную классификацию."""
    model = models.Sequential(
        [
            layers.Input(shape=(lookback, n_features)),
            layers.Masking(mask_value=0.0),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(len(CLASS_NAMES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def fit_model(
    model: models.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
) -> callbacks.History:
    """
    Обучаем модель с early stopping и снижением lr, возвращаем history.
    Веса классов считаются автоматически, чтобы компенсировать дисбаланс.
    """
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights_dict = {cls: float(w) for cls, w in zip(np.unique(y_train), class_weights)}

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights_dict,
        callbacks=[early_stop, reduce_lr],
        verbose=2,
    )
    return history


def plot_history(history: callbacks.History, out_path: Path) -> None:
    """Строим кривые loss/accuracy для train/val и сохраняем PNG."""
    hist = history.history
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(hist.get("loss", []), label="train_loss")
    plt.plot(hist.get("val_loss", []), label="val_loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hist.get("accuracy", []), label="train_acc")
    plt.plot(hist.get("val_accuracy", []), label="val_acc")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def collect_feature_columns(df: pd.DataFrame) -> List[str]:
    """Список числовых признаков, исключая служебные поля."""
    exclude = {"signal", "future_close", "future_ret", "ticker"}
    numeric_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    return sorted(numeric_cols)


def main() -> None:
    args = parse_args()
    tickers = args.tickers or DEFAULT_TICKERS

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Загрузка и подготовка датафрейма
    print(f"📥 Загружаем данные: {len(tickers)} тикеров, tf={args.timeframe}")
    df = concat_panel(tickers, args.timeframe, args.max_rows)
    # 2) Разметка: бинарно buy/sell, нейтральные выбрасываем
    df = add_labels(df, horizon=args.horizon, threshold=args.threshold)

    feature_cols = collect_feature_columns(df)
    if not feature_cols:
        raise SystemExit("Не найдено числовых признаков для обучения.")

    features = df[feature_cols].to_numpy(dtype=np.float32)
    labels = df["signal"].to_numpy(dtype=np.float32)

    # 3) Построение последовательностей и временной сплит train/val/test
    X, y = build_sequences(features, labels, lookback=args.lookback)
    X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(
        X, y, val_size=args.val_size, test_size=args.test_size
    )
    # 4) Масштабирование по train
    X_train, X_val, X_test, scaler = scale_sequences(X_train, X_val, X_test)

    print(
        f"🧾 Обучение на {len(X_train)} сэмплах, валидация {len(X_val)}, "
        f"фич: {len(feature_cols)}"
    )
    # 5) Сборка и обучение модели
    model = build_model(args.lookback, n_features=len(feature_cols), learning_rate=args.learning_rate)
    history = fit_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # 6) Оценка на val/test
    val_pred = model.predict(X_val, verbose=0).argmax(axis=1)
    test_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    report = classification_report(
        y_val, val_pred, target_names=[CLASS_NAMES[i] for i in sorted(CLASS_NAMES)], output_dict=True
    )
    test_report = classification_report(
        y_test, test_pred, target_names=[CLASS_NAMES[i] for i in sorted(CLASS_NAMES)], output_dict=True
    )

    # Сохраняем артефакты
    model.save(args.model_path)
    joblib.dump(scaler, args.scaler_path)
    with args.meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_columns": feature_cols,
                "class_mapping": CLASS_NAMES,
                "timeframe": args.timeframe,
                "lookback": args.lookback,
                "horizon": args.horizon,
                "threshold": args.threshold,
                "val_size": args.val_size,
                "test_size": args.test_size,
                "tickers": tickers,
                "val_report": report,
                "test_report": test_report,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 8) Визуализация и короткий отчёт
    plot_history(history, args.plot_path)

    print(f"✅ Модель сохранена в {args.model_path}")
    print(f"✅ Scaler сохранён в {args.scaler_path}")
    print(f"🖼️  Кривые обучения сохранены в {args.plot_path}")
    print(
        f"📊 Val accuracy: {report['accuracy']:.3f} | "
        f"buy F1: {report['buy']['f1-score']:.3f}, "
        f"sell F1: {report['sell']['f1-score']:.3f}"
    )
    print(
        f"🧪 Test accuracy: {test_report['accuracy']:.3f} | "
        f"buy F1: {test_report['buy']['f1-score']:.3f}, "
        f"sell F1: {test_report['sell']['f1-score']:.3f}"
    )


if __name__ == "__main__":
    main()

