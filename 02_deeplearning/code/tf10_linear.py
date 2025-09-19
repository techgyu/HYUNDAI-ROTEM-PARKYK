import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Input   # 케라스의 가장 끝 레이어는 덴스다
from tensorflow.keras import optimizers 
import numpy as np 

x_data = np.array([1.,2.,3.,4.,5.]).reshape(-1,1)
y_data = np.array([1.2,2.0,3.0,3.5,5.3]).reshape(-1,1)

print('상관계수 :' , np.corrcoef(x_data.ravel(), y_data.ravel()))

model = Sequential()
model.add(Input(shape=(1,)))
model.add(Dense(units=32, activation = 'relu'))  
model.add(Dense(units=32, activation = 'relu'))  
model.add(Dense(units=1, activation = 'linear'))
print(model.summary())

model.compile(optimizer='sgd', loss='mse', metrics=['mse'])
model.fit(x_data, y_data, batch_size=1, epochs=10, verbose=1, shuffle=True)
print(model.evaluate(x_data, y_data))

pred = model.predict(x_data)
print('예측값 :' , pred.ravel())
print('실제값 :' , y_data.ravel())

# 결정계수

from sklearn.metrics import r2_score
print('설명력 : ', r2_score(y_data, pred))

import matplotlib.pyplot as plt
plt.scatter(x_data, y_data, color='r', marker='o', label='real')
plt.plot(x_data, pred, 'b--', label='pred')
plt.legend()
plt.show()
