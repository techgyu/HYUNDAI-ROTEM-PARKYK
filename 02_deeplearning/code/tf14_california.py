# 캘리포니아 주택 가격 데이터로 유연한 함수형 모델 생성
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

housing = fetch_california_housing()
print(housing.keys())
print(housing.data[:3], type(housing.data))  # 설명변수
print(housing.feature_names)  # 설명변수명
print(housing.target_names) # 반응변수명
print(housing.data.shape) # (20640, 8)

# train / test
x_train_all, x_test, y_train_all, y_test = train_test_split(housing.data, housing.target, test_size=0.3, random_state=12)
print(x_train_all.shape, y_train_all.shape, x_test.shape, y_test.shape)

# train / validation
x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=12)
print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)

# 표준화
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.transform(x_valid)
x_test = scaler.transform(x_test)
print(x_train[:3])

print('Sequential api --- 단순한 방법으로 MLP ---')
model = Sequential()
model.add(Input(shape=x_train.shape[1:]))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))
print(model.summary())

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
history = model.fit(x_train, y_train, epochs=20, validation_data=(x_valid, y_valid), batch_size=512, verbose=2)
print('evaluate : ', model.evaluate(x_test, y_test, verbose=0))

# test 일부 자료로 예측
x_new = x_test[:3]
y_pred = model.predict(x_new)
print('예측값 : ', y_pred.ravel())
print('실제값 : ', y_test[:3])

plt.plot(range(1, 21), history.history['mse'], c='blue', label='mse')
plt.plot(range(1, 21), history.history['val_mse'], c='orange', label='val_mse')
plt.xlabel('epoch')
plt.ylabel('mse')
plt.legend()
# plt.show()
plt.savefig('tf14_mse.png')

print('\nFunctional api --- 유연한 방법으로 MLP ---')
input_ = Input(shape=x_train.shape[1:])
net1 = Dense(units=32, activation='relu')(input_)
net2 = Dense(units=32, activation='relu')(net1)
concat = Concatenate()([input_, net2])
output = Dense(units=1)(concat)

model2 = Model(inputs=input_, outputs=output)
model2.compile(optimizer='adam', loss='mse', metrics=['mse'])
history2 = model2.fit(x_train, y_train, epochs=20, validation_data=(x_valid, y_valid), batch_size=512, verbose=2)

plt.plot(range(1, 21), history2.history['mse'], c='blue', label='mse')
plt.plot(range(1, 21), history2.history['val_mse'], c='orange', label='val_mse')
plt.xlabel('epoch')
plt.ylabel('mse')
plt.legend()
plt.savefig('tf14_mse2.png')

print('functional api 2 -- 일부 특성은 짧은 경로로 전달, 다른 특성은 깊은 경로로 전달 MLP---')
input_a = Input(shape=[5], name='wide_input')  # MedInc ~ AveOccup (첫 5개 특성)
input_b = Input(shape=[3], name='deep_input')  # HouseAge ~ Population (나머지 3개 특성)
net1 = Dense(units=32, activation='relu')(input_b)
net2 = Dense(units=32, activation='relu')(net1)
concat = Concatenate()([input_a, net2])
output = Dense(units=1, name='output')(concat)

model3 = Model(inputs=[input_a, input_b], outputs=output)
model3.compile(optimizer='adam', loss='mse', metrics=['mse'])

# fit()을 호출할 때 하나의 입력행렬 x_train을 전달하는 것이 아니라
# 입력마다 하나씩 행렬의 튜플(x_train_a, x_train_b)을 전달
x_train_a, x_train_b = x_train[:, :5], x_train[:, 5:]
x_valid_a, x_valid_b = x_valid[:, :5], x_valid[:, 5:]
x_test_a, x_test_b = x_test[:, :5], x_test[:, 5:] # evaluate 용
x_new_a, x_new_b = x_test_a[:3], x_test_b[:3] # predict 용

history3 = model3.fit([x_train_a, x_train_b], y_train, epochs=20, validation_data=([x_valid_a, x_valid_b], y_valid), batch_size=512, verbose=2)
