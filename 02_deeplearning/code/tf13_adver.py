import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Input   # 케라스의 가장 끝 레이어는 덴스다
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Advertising.csv")
print(data.head(2))

del data['no']
print(data.head(2))

fdata = data[['tv', 'radio', 'newspaper']]
ldata = data[['sales']]
print(fdata[:2])
print(ldata[:2])

# 정규화
# scaler = MinMaxScaler(feature_range=(0, 1))
# fedata = scaler.fit_transform(fdata)
# print(fedata[:3])

fedata = minmax_scale(fdata, axis=0, copy=True) # 원본이 보존
print(fedata[:3])

# train / test
x_train, x_test, y_train, y_test = train_test_split(fedata, ldata, test_size=0.3, random_state=1234)

model  = Sequential()
model.add(Input(shape=(3,)))
model.add(Dense(units = 16, activation='relu'))
model.add(Dense(units = 8, activation='relu'))
model.add(Dense(units = 1, activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
# print(model.summary())

# tf.keras.utils.plot_model(model, 'tf13.png', show_shapes=True)
history = model.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=2,) # validation_data=(x_vali, y_vali)

# 모델 평가 점수
loss = model.evaluate(x_test, y_test, verbose=0)
print('loss : ', loss[0])

# History 값 확인
print('history: ', history.history)
print('loss : ', history.history['loss'])  # 학습용
print('mse : ', history.history['mse'])  # 검증용
print('val_loss : ', history.history['val_loss'])  # 검증용
print('val_mse : ', history.history['val_mse'])  # 검증용

# loss 시각화
plt.plot(history.history['loss'], color="blue", label='loss')
plt.plot(history.history['val_loss'], color="orange", label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
# plt.show()
plt.savefig('tf13_loss.png')
plt.close()

# r2 score
print('r2 score : ', r2_score(y_test, model.predict(x_test)))

