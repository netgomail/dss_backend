import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Шаг 1: Загрузка данных из файла Apache Parquet
# Предполагаем, что ваш файл называется 'SBER_M30.parquet' и имеет ту же структуру колонок, как в примере (time, open, high, low, close, volume, sma5, ..., market_regime).
# Если имя файла другое, замените на правильное. Pandas может читать Parquet напрямую, если установлен pyarrow или fastparquet (в вашем окружении pandas должен поддерживать это).
file_path = 'SBER_M30.parquet'  # Укажите путь к вашему Parquet файлу
df = pd.read_parquet(file_path)

# Удаляем колонку 'time', так как она не нужна для модели (временной ряд определяется порядком строк).
# Но если нужно, можно сохранить её отдельно для анализа.
if 'time' in df.columns:
    df.drop('time', axis=1, inplace=True)

# Шаг 2: Подробное объяснение разметки (лейблинга) для buy/sell
# Разметка — это создание целевой переменной (target), которая определяет, что модель должна предсказывать.
# Поскольку нужно только покупать (buy) или продавать (sell), без "держать" (hold), мы делаем бинарную классификацию:
# - target = 1 (buy): Если цена закрытия (close) через 'horizon' свечей вперёд выше текущей close плюс небольшой порог (threshold).
#   Почему порог? Чтобы учесть комиссии, спред и шум рынка. Например, threshold=0.001 значит +0.1% — если рост меньше, это не выгодно для buy.
# - target = 0 (sell): Во всех других случаях (падение или рост меньше threshold).
# 
# Как определяется:
# - Мы используем shift(-horizon) в pandas, чтобы "сдвинуть" будущие значения close назад.
# - Сравниваем: если future_close > current_close * (1 + threshold), то buy (1), иначе sell (0).
# - Horizon: Количество свечей вперёд для предсказания. Для M30 (30-минутный таймфрейм) horizon=1 значит предсказываем на следующие 30 минут.
#   Можно увеличить до 3-5 для более стабильных сигналов, но это уменьшит количество данных (удалятся последние horizon строк).
# - Threshold: Подстройте под ваш брокер (комиссия ~0.05-0.2%). Если 0, то просто > current_close.
# - Важно: Это "look-ahead" — модель учится на исторических данных, предсказывая "будущее" относительно прошлого.
# - Баланс: После создания проверьте np.mean(target) — если ~0.5, хорошо; если сильно смещено, модель может偏向 к большинству.

threshold = 0.001  # 0.1% порог для buy (учёт комиссий/спреда)
horizon = 1  # Предсказываем на 1 свечу вперёд (можно изменить на 3-5)
df['target'] = np.where(df['close'].shift(-horizon) > df['close'] * (1 + threshold), 1, 0)
df.dropna(inplace=True)  # Удаляем строки без target (последние horizon строк)

# Печатаем баланс классов для проверки
print(f"Баланс классов в target: {df['target'].mean():.2f} (доля buy=1)")

# Шаг 3: Разделение данных на train, validation и test
# Поскольку это временной ряд, разделяем хронологически: первые 70% — train, следующие 15% — val, последние 15% — test.
# Нет shuffle! Train — прошлое, test — "будущее".
total_rows = len(df)
train_size = int(total_rows * 0.7)
val_size = int(total_rows * 0.15)
test_size = total_rows - train_size - val_size

train_df = df.iloc[:train_size]
val_df = df.iloc[train_size:train_size + val_size]
test_df = df.iloc[train_size + val_size:]

# Извлекаем фичи (все колонки кроме target) и target
features = df.columns.drop('target')
X_train = train_df[features].values
y_train = train_df['target'].values
X_val = val_df[features].values
y_val = val_df['target'].values
X_test = test_df[features].values
y_test = test_df['target'].values

# Шаг 4: Нормализация данных
# Используем MinMaxScaler для приведения всех фич к [0,1]. Fit только на train, чтобы избежать утечки данных.
# Бинарные фичи (как pat_doji) не пострадают, так как они уже 0/1.
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Шаг 5: Формирование последовательностей для LSTM (windowing)
# LSTM требует 3D-данных: [samples, timesteps, features].
# timesteps — длина окна (сколько прошлых свечей видит модель для предсказания одной).
# Например, timesteps=30 значит модель смотрит 30 предыдущих строк для прогноза target следующей.
# Подберите: 30-60 для M30 таймфрейма.
def create_sequences(X, y, timesteps=30):
    Xs, ys = [], []
    for i in range(len(X) - timesteps):
        Xs.append(X[i:i + timesteps])
        ys.append(y[i + timesteps])  # Target для конца окна
    return np.array(Xs), np.array(ys)

timesteps = 30  # Можно изменить
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, timesteps)
X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, timesteps)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, timesteps)

# Преобразуем y в categorical для softmax (2 класса: 0 и 1)
num_classes = 2
y_train_seq = to_categorical(y_train_seq, num_classes=num_classes)
y_val_seq = to_categorical(y_val_seq, num_classes=num_classes)
y_test_seq = to_categorical(y_test_seq, num_classes=num_classes)

# Шаг 6: Построение модели LSTM
# Архитектура: 2 слоя LSTM с dropout для предотвращения переобучения.
# Input: (timesteps, num_features)
# Output: 2 нейрона с softmax для вероятностей buy/sell.
num_features = X_train_seq.shape[2]
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(timesteps, num_features)))
model.add(Dropout(0.2))  # 20% dropout
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# Компиляция: Adam optimizer, categorical_crossentropy для multi-class (даже для 2 классов).
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Шаг 7: Обучение модели
# EarlyStopping: Останавливаем, если val_loss не улучшается 10 эпох.
# class_weight: Если классы несбалансированы (например, больше sell), взвешиваем (здесь пример; подстройте по балансу).
class_weight = {0: 1.0, 1: 1.2} if df['target'].mean() < 0.5 else {0: 1.2, 1: 1.0}  # Больше веса меньшему классу
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(X_train_seq, y_train_seq, epochs=100, batch_size=32, validation_data=(X_val_seq, y_val_seq),
          callbacks=[early_stop], class_weight=class_weight, verbose=1)

# Шаг 8: Оценка модели на test set
# Предсказываем вероятности, затем классы (argmax).
y_pred_prob = model.predict(X_test_seq)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test_seq, axis=1)

# Метрики: Accuracy — общая точность, Precision — точность сигналов (важно для торговли, чтобы минимизировать ложные buy/sell).
# Recall — полнота, F1 — баланс.
print(f"Test Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(f"Test Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Test Recall: {recall_score(y_true, y_pred):.4f}")
print(f"Test F1 Score: {f1_score(y_true, y_pred):.4f}")

# Шаг 9: Сохранение модели (опционально)
model.save('stock_lstm_model.h5')  # Для дальнейшего использования

# Как использовать модель для предсказаний:
# Для новых данных: нормализуйте, создайте sequence, model.predict() -> если prob[1] > 0.5 -> buy, else sell.