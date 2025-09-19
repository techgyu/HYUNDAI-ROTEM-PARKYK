# 다중선형회귀 - 자동차 연비 예측 - Network 구성 함수작성, 조기종료(early stopping)

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns


dataset = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/auto-mpg.csv', na_values = '?')
# print(dataset.head(2))
print(dataset.columns)
# print(dataset.describe())
del dataset['car name']
dataset.drop(['cylinders', 'acceleration','model year', 'origin'], axis = 1, inplace = True)
print(dataset.corr())
print(dataset.info())
# 실린더, 액셀레이션
dataset = dataset.dropna()
print(dataset.isna().sum())   # 결측치 찾아봐
sns.pairplot(dataset[['mpg', 'displacement', 'horsepower', 'weight']], diag_kind = 'kde')
# plt.show()
plt.close()
# train / test split
train_dataset = dataset.sample(frac = 0.7, random_state = 123)
test_dataset = dataset.drop(train_dataset.index)
print(train_dataset[:2], train_dataset.shape)     # (274, 4)
print(test_dataset[:2], test_dataset.shape)       # (118, 4)

# 표준화 - 수식 (관찰값 - 평균) / 표준편차
train_stat = train_dataset.describe()
print(train_stat)
train_stat.pop('mpg')
print(train_stat.transpose())
train_stat = train_stat.transpose()

def std_func(x):
  return (x - train_stat['mean']) / train_stat['std']

# print(std_func(train_dataset[:3]))
st_train_data = std_func(train_dataset)
st_train_data = st_train_data.drop(['mpg'], axis = 'columns')
print(st_train_data[:2])

st_test_data = std_func(test_dataset)
st_test_data = st_test_data.drop(['mpg'], axis = 'columns')
print(st_test_data[:2])

train_label = train_dataset.pop('mpg')
print(train_label[:2])
test_label = test_dataset.pop('mpg')
print(test_label[:2])


def build_model():
    network = Sequential([
        Input(shape = (3,)),
        Dense(units = 32, activation = 'relu'),
        Dense(units = 32, activation = 'relu'),
        Dense(units = 1, activation = 'linear'),
    ])

    opti = tf.keras.optimizers.Adam(learning_rate = 0.01)
    network.compile(optimizer = opti, loss = 'mean_squared_error', metrics =['mean_squared_error', 'mean_absolute_error'])
    # 메트릭스에서 mse가 먼저 와야돼 mae가 먼저오면 안된대
    return network

model = build_model()
print(model.summary())

epochs = 5000
# 조기종료에 대한 얘기
early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', baseline = 0.03, patience = 5)
# patience의 횟수만큼 학습결과가 반복되면 얼리 스탑 한다 얘기야

history = model.fit(st_train_data, train_label, 
                    batch_size=1024, 
                    epochs = epochs, 
                    validation_split= 0.2, 
                    verbose = 2, 
                    callbacks = [early_stop])
df = pd.DataFrame(history.history)
print(df.head(3))

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure(figsize=(8, 12))
    plt.subplot(2,2,1)
    plt.xlabel('Epoch')
    plt.ylabel('mean squared error[mpg]')
    plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label = 'Val Error')
    plt.subplot(2,1,2)
    plt.xlabel('Epoch')
    plt.ylabel('mean absolute error[mpg]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label = 'Val Error')
    plt.legend()
    plt.savefig('tf15_mlp.png')
plot_history(history)

loss, mse, mae = model.evaluate(st_test_data, test_label)
print('test dataset으로 평가 loss: {:5.3f}'.format(loss))
print('test dataset으로 평가 mse: {:5.3f}'.format(mse))
print('test dataset으로 평가 mae: {:5.3f}'.format(mae))

print()
# 새로운 값으로 예측
# 'displacement', 'horsepower', 'weight'
new_data = pd.DataFrame({'displacement': [300, 400], 'horsepower':[120, 140], 'weight':[2000, 4000]})
new_st_data = std_func(new_data)
print(new_st_data)
new_pred = model.predict(new_st_data).ravel()
print(new_pred)
print('예측 결과: ', new_pred())



