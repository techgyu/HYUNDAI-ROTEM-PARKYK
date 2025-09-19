# Keras를 사용한 논리회로 분류 모델 작성

import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD, RMSprop, Adam

# 1. 데이터 세트 생성
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]])  # XOR 문제

# 2. 모델 구성
# model = Sequential([
#     Input(shape=(2,)),  # 입력층: 2개의 입력 뉴런
#     Dense(units = 1),  # 은닉층: 1개의 뉴런
#     Activation('sigmoid')  # 활성화 함수: 시그모이드
# ])

model = Sequential()
model.add(Input(shape=(2,)))  # 입력층: 2개의 입력 뉴런
model.add(Dense(units = 1))  # 은닉층: 1개의 뉴런
model.add(Activation('sigmoid'))  # 활성화 함수: 시그모이드

# 3. 모델 학습 과정 설정
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# model.compile(optimizer=SGD(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=RMSprop(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

# 4. 모델 학습 시키기
model.fit(x=x, y=y, epochs=10, batch_size=1, verbose=1)

# 5. 모델 평가
loss_metrics = model.evaluate(x, y, batch_size=1)
print('loss_metrics : ', loss_metrics)

# 6. 모델 사용하기 - 예측값 확인
proba = model.predict(x, verbose=0)
print('예측값 : \n', proba)
pred = (proba > 0.5).astype(np.int32)  # 확률을 0과 1로 변환
print(pred.ravel())  # 1차원 배열로 변환하여 출력

# 7. 모델 저장
model.save('model_tf5_keras.h5')

# 8. 모델 읽기
from tensorflow.keras.models import load_model
model2 = load_model('model_tf5_keras.h5')
proba = model2.predict(x, verbose=0)
pred = (proba > 0.5).astype(np.int32)  # 확률을 0과 1로 변환
print("처리결과: ", pred.ravel())  # 1차원 배열로 변환하여 출력