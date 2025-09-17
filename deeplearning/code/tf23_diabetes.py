# sigmoid는 softmax로 처리 가능
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

dataset = np.loadtxt('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/diabetes.csv', delimiter=',')
print(dataset.shape) # (759, 9)
print(dataset[:3])
print(set(dataset[:, -1])) # 타겟 클래스: 0, 1

# 이항분류
x_train, x_test, y_train, y_test = train_test_split(dataset[:, 0:8], dataset[:, -1], test_size=0.3, shuffle=True, random_state=123)
print(x_train.shape, x_test.shape) # (531, 8) (228, 8)

model = Sequential()
model.add(Input(shape=(8,)))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='sigmoid')) # 이진 분류에서는 출력층의 활성화 함수로 sigmoid 사용

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # 이진 분류에서는 손실함수로 binary_crossentropy 사용
print(model.summary())

model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
scores = model.evaluate(x_test, y_test)
print('%s : %.2f%%' % (model.metrics_names[1], scores[1]*100))



# 다항분류
x_train, x_test, y_train, y_test = train_test_split(dataset[:, 0:8], dataset[:, -1], test_size=0.3, shuffle=True, random_state=123)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model2 = Sequential()
model2.add(Input(shape=(8,)))
model2.add(Dense(units=64, activation='relu'))
model2.add(Dense(units=32, activation='relu'))
model2.add(Dense(units=2, activation='softmax')) # 다중 분류에서는 출력층의 활성화 함수로 softmax 사용

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # 다중 분류에서는 손실함수로 categorical_crossentropy 사용
print(model2.summary())

model2.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
scores2 = model2.evaluate(x_test, y_test)
print('%s : %.2f%%' % (model2.metrics_names[1], scores2[1]*100))

