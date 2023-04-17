import requests
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import tensorflow as tf

# Определите URL-адрес API и параметры запроса
url = 'https://api.binance.com/api/v3/klines'
params = {
    'symbol': 'BTCUSDT', # торговая пара
    'interval': '1h', # временной интервал
    'limit': 1000 # количество записей
}

# Сделайте GET-запрос и получите ответ в формате JSON
response = requests.get(url, params=params)
data = response.json()

# Преобразование данных в Pandas DataFrame
df = pd.DataFrame(data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignored'])

# Преобразование времени в Pandas datetime
df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

# Сохранение данных в CSV-файл
df.to_csv('btcusdt.csv', index=False)

# Чтение CSV-файла в Pandas DataFrame
df = pd.read_csv('btcusdt.csv')

# Удаление пропущенных значений
df.dropna(inplace=True)

# Задание параметров модели
n_neurons = 128 # количество нейронов в скрытом слое
n_steps = 60 # количество временных шагов
batch_size = 32 # размер батча

# Масштабирование данных
scaler = MinMaxScaler()
df[['open', 'high', 'low', 'close', 'volume']] = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume']])

# Выборка признаков
X = df[['open', 'high', 'low', 'close', 'volume']].values

# Выборка целевой переменной
y = df['close'].values

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Разделение данных на последовательности длиной n_steps
def split_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        end_idx = i + n_steps
        if end_idx >= len(sequence):
            break
        seq_x, seq_y = sequence[i:end_idx, :], sequence[end_idx, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

X_train, y_train = split_sequence(X_train, n_steps)
X_test, y_test = split_sequence(X_test, n_steps)

# Создание модели
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(n_steps, X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

# Обучение модели
model.fit(X_train,y_train,
    epochs=10,
    batch_size=batch_size,
    validation_data=(X_test, y_test)
    )

scores = model.evaluate(X_test, y_test, verbose=0)
print(f'Mean squared error: {scores[1]}')
# Получение предсказаний на тестовой выборке
y_pred = model.predict(X_test)


# Обратное масштабирование предсказанных значений
y_pred = y_pred.reshape(-1, 5)
y_pred = scaler.inverse_transform(y_pred)

# Обратное масштабирование целевых переменных
y_test = y_test.reshape(-1, 5)
y_test = scaler.inverse_transform(y_test)

# Расчет ошибки модели на тестовой выборке
mse = tf.keras.metrics.mean_squared_error(y_test, y_pred)
mse_np = mse.numpy()
for i in range(len(mse_np)):
    print('Group %d MSE: %.6f' % (i+1, mse_np[i]))

# Построение графика
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()
