import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import FinanceDataReader as fdr
import warnings
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Lambda
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam

URL = "C:\\perdictShareprice\\share csv\\01-삼성전자-주가.csv"

# %matplotlib inline
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'NanumGothic'

# 삼성전자(005930) 전체 (1996-11-05//2017 ~ 현재)
samsung = fdr.DataReader('005930')
# Open: 시가
# High: 고가
# Low: 저가
# Close: 종가
# Volume: 거래량
# Change: 대비

# print(samsung.tail())
# print(samsung.head())
# print(samsung.index)

samsung['Year'] = samsung.index.year
samsung['Month'] = samsung.index.month
samsung['Day'] = samsung.index.day

plt.figure(figsize=(16, 9))
sns.lineplot(y=samsung['Close'], x=samsung.index)
plt.xlabel('time')
plt.ylabel('price')
time_steps = [['1990', '2000'],
              ['2000', '2010'],
              ['2010', '2015'],
              ['2015', '2020']]

fig, axes = plt.subplots(2, 2)
fig.set_size_inches(16, 9)

for i in range(4):
    ax = axes[i//2, i % 2]
    df = samsung.loc[(samsung.index > time_steps[i][0]) &
                     (samsung.index < time_steps[i][1])]
    sns.lineplot(y=df['Close'], x=df.index, ax=ax)
    ax.set_title(f'{time_steps[i][0]}~{time_steps[i][1]}')
    ax.set_xlabel('time')
    ax.set_ylabel('price')
plt.tight_layout()
# plt.show()

# 정규화작업
scaler = MinMaxScaler()
scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
scaled = scaler.fit_transform(samsung[scale_cols])
# print(scaled)

df = pd.DataFrame(scaled, columns=scale_cols)


# train /test 분할
x_train, x_test, y_train, y_test = train_test_split(
    df.drop('Close', 1), df['Close'], test_size=0.2, random_state=0, shuffle=False)
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)
# print(x_train)

# 데이터 시퀀스 데이터셋 구성


def dataset(series, window_size, batch_size, shuffle):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensors(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    return ds.batch(batch_size).prefetch(1)


WINDOW_SIZE = 20
BATCH_SIZE = 32

# trian_data는 학습용 데이터셋, test_data는 검증용 데이터셋
train_data = dataset(y_train, WINDOW_SIZE, BATCH_SIZE, True)
test_data = dataset(y_test, WINDOW_SIZE, BATCH_SIZE, False)
# for data in train_data.take(1):
#     print(f'데이터셋(X) 구성(batch_size, window_size, feature갯수): {data[0].shape}')
#     print(f'데이터셋(Y) 구성(batch_size, window_size, feature갯수): {data[1].shape}')

model = Sequential()
model.add(LSTM(16, activation='tanh'),
          )
model.add(Conv1D(filters=32, kernel_size=5,
                 padding="causal",
                 activation="relu",
                 input_shape=[WINDOW_SIZE, 1]
                 ),)
model.add(Dense(1))
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv1D(filters=32, kernel_size=5,
#                            padding="causal",
#                            activation="relu",
#                            input_shape=[WINDOW_SIZE, 1]
#                            ),
#     # LSTM
#     tf.keras.layers.LSTM(16, activation='tanh'),
#     tf.keras.layers.Dense(16, activation="relu"),
#     tf.keras.layers.Dense(1),
# ])


# 스퀀스 학습 = Huber
# loss=tf.keras.losses.Huber()
optimizer = Adam(0.0005)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer, metrics=['mse'])

# earlystopping= 10번 epoching  val_loss 개선이 없다면 학습을 멈춤
# earlystopping = EarlyStopping(
#     monitor='val_loss', patience=10)
# # val_loss 기준 체크 포인터
# # filename = os.path.join('tmp', 'tmp_checkpoint.h5')
# filename = os.path.join('tmp', 'checkpointer.ckpt')
# checkpoint = ModelCheckpoint(
#     filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
# # checkpoint = ModelCheckpoint(filename,
# #                              save_weights_only=True,
# #                              save_best_only=True,
# #                              monitor='val_loss',
# #                              verbose=1,
# #                              mode='auto')

# early_stop = EarlyStopping(monitor='val_loss', patience=5)
# # history = model.fit(x_train, y_train,
# #                     epochs=50,
# #                     batch_size=16,
# #                     validation_data=(x_valid, y_valid),
# #                     callbacks=[early_stop, checkpoint],
# #                     verbose=1)

# history = model.fit(train_data,
#                     epochs=50,
#                     batch_size=16,
#                     validation_data=(test_data),
#                     callbacks=[early_stop, checkpoint],
#                     )

# model.load_weights(filename)

pred = model.predict(test_data)
print(pred.shape)


# 예측 데이터 시각화 20일치를 이용하여 21일을 예측
plt.figure(figsize=(12, 9))
plt.plot(np.asarray(y_test)[20:], label='actual')
plt.plot(pred, label='prediction')
plt.legend()
plt.show()


# Shareprice = pd.read_csv(
#     "C:\\perdict Share price\\share csv\\01-삼성전자-주가.csv")
# Shareprice.head()
# # print(Shareprice.shape)
# # print(Shareprice.columns)
# 종속 = Shareprice[['Volume']]
# 독립 = Shareprice[['Open', 'High', 'Low', 'Close', 'Adj Close']]
# x = tf.keras.layers.Input(shape=[5])
# h = tf.keras.layers.Dense(8, activation='swish')(x)
# h = tf.keras.layers.Dense(8, activation='swish')(h)
# h = tf.keras.layers.Dense(8, activation='swish')(h)
# h = tf.keras.layers.Dense(8, activation='swish')(h)
# h = tf.keras.layers.Dense(8, activation='swish')(h)
# h = tf.keras.layers.Dense(8, activation='swish')(h)
# h = tf.keras.layers.Dense(8, activation='swish')(h)
# h = tf.keras.layers.Dense(8, activation='swish')(h)
# h = tf.keras.layers.Dense(8, activation='swish')(h)
# y = tf.keras.layers.Dense(1)(h)
# model = tf.keras.models.Model(x, y)
# model.compile(loss='mse')
# #model.fit(독립, 종속, epochs=1000, verbose=0)
# model.fit(독립, 종속, epochs=100)
