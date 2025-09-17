# 다항분류 : 출력값이 softmax 함수로 인해 확률값으로 출력. 이 때 확률 값이 가장 높은 클래스로 분류

# softmax 함수 작성
import numpy as np

# def softmaxFunc(a):
#     c = np.max(a)
#     exp_a = np.exp(a) 
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a
#     return y

# a = np.array([1.0, 1.0, 1.5])
# result = softmaxFunc(a)
# print(result)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical # one-hot encoding 지원
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

np.random.seed()
# data 준비
xdata = np.random.random((1000, 12)) # 시험 점수라고 가정
ydata = np.random.randint(5, size=(1000, 1)) # 0~4 사이의 정수형 레이블 1000개 생성
print(xdata[:5]) # feature
print(ydata[:2]) # label 정수를 다섯가지 형태로 출력될 수 있도록 모양 변경 - 원핫 처리

ydata = to_categorical(ydata, num_classes=5) # one-hot encoding
print(ydata[:2])
# print([np.argmax(i) for i in ydata[:2]]) # 원핫 인코딩된 것을 다시 정수형으로 변환

# 모델 작성
model = Sequential()
model.add(Input(shape=(12,)))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=5, activation='softmax')) # 다중 분류에서는 출력층의 활성화 함수로 softmax 사용
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy']) # 다중 분류에서는 손실함수로 categorical_crossentropy 사용
print('learning rate : ', model.optimizer.learning_rate.numpy())

history = model.fit(xdata, ydata, epochs=1000, batch_size=32, verbose=0)

model_eval = model.evaluate(xdata, ydata, verbose=0)

print('모델 평가 결과 : ', model_eval)

# 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['loss'])
ax1.set_title('Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')

ax2.plot(history.history['accuracy'])
ax2.set_title('Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')

# plt.show()
plt.close()

# 분류 예측 결과 보기
np.set_printoptions(suppress=True, precision=5)
print('예측값 : ', model.predict(xdata[:5]))
print('예측값 : ', np.argmax(model.predict(xdata[:5]), axis=1))
print('실제값 : ', ydata[:5])
print('실제값 : ', [int(i) for i in np.argmax(ydata[:5], axis=1)])

# 새로운 값으로 예측
x_new = np.random.random([1, 12])
print(x_new)
new_pred = model.predict(x_new)
print('분류 결과 : ', new_pred, ', 모두 더하면:', np.sum(new_pred))
print('분류 결과 : ', np.argmax(new_pred))

# 가정 : 레이블에 해당하는 과목명 출력
classes = np.array(['국어', '영어', '수학', '과학', '체육'])
print('예측값 : ', classes[np.argmax(model.predict(xdata[:5]), axis=1)])

