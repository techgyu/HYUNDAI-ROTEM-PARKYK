# Keras를 사용한 논리회로 분류 모델 작성

from matplotlib.pyplot import hist
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD, RMSprop, Adam

# 1. 데이터 세트 생성
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])  # XOR 문제

# 2. 모델 구성
# model = Sequential([
#     Input(shape=(2,)),  # 입력층: 2개의 입력 뉴런
#     Dense(units = 1),  # 은닉층: 1개의 뉴런
#     Activation('sigmoid')  # 활성화 함수: 시그모이드
# ])

model = Sequential()
model.add(Input(shape=(2,)))  # 입력층: 2개의 입력 뉴런
model.add(Dense(units = 50, activation='relu'))  # 은닉층: 50개의 뉴런
model.add(Dense(units = 50, activation='relu'))  # 은닉층: 50개의 뉴런
model.add(Dense(units = 50, activation='relu'))  # 은닉층: 50개의 뉴런
model.add(Dense(units = 1, activation='sigmoid'))  # 출력층: 1개의 뉴런

print(model.summary())

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

history = model.fit(x, y, epochs=1000, batch_size=1, verbose=0)  # verbose=0: 진행상황 출력 안함

pred = (model.predict(x) > 0.5).astype('int32')  # 예측값을 0 또는 1로 변환

print(model.weights)
print(history.history['loss'][:10])  # 마지막 손실값 출력
print(history.history['accuracy'][:10])  # 마지막 정확도 출력

import matplotlib.pyplot as plt
plt.plot(history.history['loss'], c='red', label='train loss')
plt.plot(history.history['accuracy'], c='blue', label='train acc')
plt.xlabel('Epochs')
plt.legend()
# plt.show()