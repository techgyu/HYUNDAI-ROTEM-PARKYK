# mnist dataset으로 CNN 모델 작성
import numpy as np
import matplotlib.pyplot as plt
import keras

# 데이터 로드
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 스케일링 및 차원 확장
x_train = (x_train.astype("float32") / 255.0).reshape((-1, 28, 28, 1))
x_test = (x_test.astype("float32") / 255.0).reshape((-1, 28, 28, 1))

# 모델 정의
model = keras.models.Sequential([
    keras.layers.Input(shape=(28, 28, 1)), # - 입력 데이터는 (28, 28, 1) 크기의 흑백 이미지이다.
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'), # - 첫 번째 합성곱 층(Conv2D)은 필터 수 32개, 커널 크기 (3,3), 활성화 함수 relu를 사용한다.
    keras.layers.MaxPool2D(pool_size=(2, 2)), # - 합성곱 층 뒤에는 (2,2) 크기의 MaxPooling2D를 적용한다.

    keras.layers.Flatten(), # - Flatten 층을 사용하여 Dense 층과 연결한다.

    keras.layers.Dense(units=10, activation='softmax'), # - 출력층은 클래스 개수(10개)에 맞춰 Dense(10, activation="softmax")로 구성한다.
])

print(model.summary())

model.compile(optimizer='adam', # - Optimizer는 'adam'을 사용한다.
              loss='sparse_categorical_crossentropy',  # - label은 원핫 처리 하지 않음
              metrics=['accuracy']) 

history = model.fit(x_train, y_train, epochs=3, batch_size=512, validation_split=0.1, verbose="auto") # - 학습 횟수는 3으로 수행한다 


# 모델 평가
train_loss, train_acc = model.evaluate(x_train, y_train, verbose="0")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose="0")
print("train_acc:", train_acc)
print("test_acc:", test_acc)