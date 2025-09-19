# wine dataset으로 레드/화이트 와인 분류 모델

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


wdf = pd.read_csv('deeplearning/python-master/python-master/testdata_utf8/wine.csv', header=None)
wdf = pd.DataFrame(wdf)
print(wdf)
print(wdf.info())
print(wdf.iloc[:, 12].unique()) # [1, 0] 레드와인 1, 화이트와인 0
print(len(wdf[wdf.iloc[:, 12] == 0])) # 4898
print(len(wdf[wdf.iloc[:, 12] == 1])) # 1599

dataset = wdf.values
x = dataset[:, 0:12]
y = dataset[:, -1]

np.set_printoptions(suppress=True)
print(x[:3])
print(y[:3])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (4597, 12) (2000, 12) (4597,) (2000,)
print(x_train[:3])
print(y_train[:3])

# 모델 생성
model = Sequential()
model.add(Input(shape=(12,)))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(BatchNormalization()) # 배치 정규화, 역전파시 기울기 소실 또는 폭주 방지, cnn 등에서 특히 효과적.
model.add(Dense(units=16, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(BatchNormalization()) # 배치 정규화, 역전파시 기울기 소실 또는 폭주 방지, cnn 등에서 특히 효과적.
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='sigmoid')) # 이진 분류
model.summary()

model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])
# fit() 전에 model score 확인
loss, acc = model.evaluate(x_train, y_train, verbose=0)
print('훈련 전 모델 정확도 : {:5.2f}'.format(100 * acc))