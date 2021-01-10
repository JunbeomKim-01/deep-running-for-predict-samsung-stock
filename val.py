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

data_path = "C:\\perdictShareprice\\share csv\\"
df_price = pd.read_csv(os.path.join(
    data_path, '01-삼성전자-주가.csv'), encoding='utf8')
print(df_price.describe())

pd.to_datetime(df_price['일자'], format='%Y%m%d')
# 0      2020-01-07
# 1      2020-01-06
# 2      2020-01-03
# 3      2020-01-02
# 4      2019-12-30

df_price['일자'] = pd.to_datetime(df_price['일자'], format='%Y%m%d')
df_price['연도'] = df_price['일자'].dt.year
df_price['월'] = df_price['일자'].dt.month
df_price['일'] = df_price['일자'].dt.day


plt.figure(figsize=(16, 9))
sns.lineplot(y=df_price['종가'], x=df_price['일자'])
plt.xlabel('time')
plt.ylabel('price')
# plt.show()

scaler = MinMaxScaler()
scale_cols = ['시가', '고가', '저가', '종가', '거래량']
df_scaled = scaler.fit_transform(df_price[scale_cols])

df_scaled = pd.DataFrame(df_scaled)
df_scaled.columns = scale_cols

print(df_scaled)
TEST_SIZE = 200  # 200일 이전의 데이터로 학습
train = df_scaled[:-TEST_SIZE]
test = df_scaled[-TEST_SIZE:]


def make_dataset(data, label, window_size=20):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)


feature_cols = ['시가', '고가', '저가', '거래량']
label_cols = ['종가']

train_feature = train[feature_cols]
train_label = train[label_cols]
test_feature = train[feature_cols]
test_label = train[label_cols]
# train dataset
train_feature, train_label = make_dataset(train_feature, train_label, 20)

# train, validation set 생성
x_train, x_valid, y_train, y_valid = train_test_split(
    train_feature, train_label, test_size=0.2)

x_train.shape, x_valid.shape
# ((6086, 20, 4), (1522, 20, 4))

# test dataset (실제 예측 해볼 데이터)
test_feature, test_label = make_dataset(test_feature, test_label, 20)
test_feature.shape, test_label.shape
# ((180, 20, 4), (180, 1))

model = Sequential()
model.add(LSTM(16,
               input_shape=(train_feature.shape[1], train_feature.shape[2]),
               activation='relu',
               return_sequences=False)
          )
model.add(Dense(1))
optimizer = tf.keras.optimizers.Adam(0.0005)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer, metrics=['mse'])

early_stop = EarlyStopping(monitor='val_loss', patience=5)
filename = os.path.join('tmp', 'tmp_checkpoint.h5')
checkpoint = ModelCheckpoint(
    filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = model.fit(x_train, y_train,
                    epochs=50,
                    batch_size=16,
                    validation_data=(x_valid, y_valid),
                    callbacks=[early_stop, checkpoint],
                    verbose=1)

# weight 로딩
model.load_weights(filename)

# 예측
pred = model.predict(test_feature)
print(pred)
plt.figure(figsize=(12, 9))
plt.plot(test_label, label='actual')
plt.plot(pred, label='prediction')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
plt.show()
