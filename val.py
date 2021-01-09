import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import os


def pltlib(URL):
    se = pd.read_csv(URL)
    # pt.figure(figsize=(12, 8))
    # pt.plot(x=se['일자'], y=se['종가'])
    # pt.xticks(rotation=75)

    plt.figure(figsize=(16, 9))
    se['일자'] = pd.to_datetime(se['일자'], format='%Y%m%d')
    sns.lineplot(y=se['종가'], x=se['일자'])
    plt.xlabel('time')
    plt.ylabel('price')
    plt.show()
