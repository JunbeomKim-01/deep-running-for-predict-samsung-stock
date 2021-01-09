import tensorflow as tf
import pandas as pd
import val
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import FinanceDataReader as fdr
import warnings
import os

# URL = "C:\\perdictShareprice\\share csv\\01-삼성전자-주가.csv"
# val.pltlib(URL)
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
