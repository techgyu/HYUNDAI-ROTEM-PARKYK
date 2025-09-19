# MNIST database(숫자 이미지)
# 숫자 손글씨 이미지에 대한 데이터와 라벨이 포함되어 있으며 60000개의 트레이닝 데이터와
# 10000개의 테스트 데이터로 구성되어 있음

import tensorflow as tf
import sys
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import keras
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)  # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
print(x_train[0])  # 0 번째 feature
print(y_train[0])  # 5 번째 feature

for i in x_train[0]:
    for j in i:
        sys.stdout.write("%d\t" % j)
    sys.stdout.write("\n")

plt.imshow(x_train[0], cmap='Greys')
plt.show()

x_train = x_train.reshape(60000, 784).astype('float32')
x_test = x_test.reshape(10000, 784).astype('float32')
# print(x_train[0])

x_train /= 255.0 # 정규화 : 필수
print(x_train[0])

# label ; One-Hot Encoding : 출력층 활성화 함수를 softmax를 사용하기 때문에
print(set(y_train))  # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
print(y_train[0]) 

# validation data
x_val = x_train[50000:60000]
y_val = y_train[50000:60000]
x_train = x_train[0:50000]
y_train = y_train[0:50000]
print(x_val.shape, y_val.shape, x_train.shape, y_train.shape)  # (10000, 784) (10000, 10) (50000, 784) (50000, 10)

# model
model = Sequential()
model.add(Input(shape=(784,)))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_val, y_val), verbose=2)

print('loss', history.history['loss'])
print('val_loss', history.history['val_loss'])
print('accuracy', history.history['accuracy'])
print('val_accuracy', history.history['val_accuracy'])

epochs = range(1, len(history.history['loss']) + 1)
plt.plot(epochs, history.history['loss'], label='Training loss')
plt.plot(epochs, history.history['val_loss'], label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.save('tf27model.keras')

del model

mymodel = tf.keras.models.load_model('tf27model.keras')

plt.imshow(x_test[:1].reshape(28, 28), cmap='Greys')
plt.show()

pred = mymodel.predict(x_test[:1])
print('pred: ', pred)
print('예측값: ', np.argmax(pred, axis=1))
print('실제값: ', y_test[:1])
print('실제값: ', np.argmax(y_test[:1], axis=1))