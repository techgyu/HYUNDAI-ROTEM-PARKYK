# mnist dataset으로 CNN 모델 작성
import numpy as np
import matplotlib.pyplot as plt
import keras as keras  # standalone keras 사용

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(x_train.shape)
x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
print(x_train.shape)

# model
# 사용자 정의 클래스(모델, 레이어, 함수: 손실, 활성화)를 모델 저장 시 자동으로 직렬화 시스템에 등록해 주는 역할
@keras.utils.register_keras_serializable(package='custom') # 'losses', 'activation'
class MyMnistCnn(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Conv block1
        self.conv1 = keras.layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.pool1 = keras.layers.MaxPool2D(pool_size=(2, 2))

        # Conv block2
        self.conv2 = keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')
        self.pool2 = keras.layers.MaxPool2D(pool_size=(2, 2))

        # Conv block3
        self.conv3 = keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')
        self.pool3 = keras.layers.MaxPool2D(pool_size=(2, 2))

        self.flat = keras.layers.Flatten() # Fully connected Layer에 연결하기 위한 1차원 변환

        self.d1 = keras.layers.Dense(units=64, activation='relu')
        self.do1 = keras.layers.Dropout(0.3)
        self.d1 = keras.layers.Dense(units=32, activation='relu')
        self.do1 = keras.layers.Dropout(0.2)
        self.d1 = keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flat(x)
        x = self.d1(x)
        x = self.do1(x, training=training)
        x = self.d2(x)
        x = self.do2(x, training=training)
        return self.out(x)

model = MyMnistCnn()
model.build(input_shape=(None, 28, 28, 1))
model.summary()


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # 라벨이 정수이므로 sparse 사용
              metrics=['accuracy'])

# 콜백 정의 (조기 종료)
es = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    x_train, y_train, epochs=100, batch_size=512, validation_split=0.1, 
    callbacks=[es], verbose="auto"
)

# 모델 평가
train_loss, train_acc = model.evaluate(x_train, y_train, verbose="0")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose="0")
print("train_acc:", train_acc)
print("test_acc:", test_acc)

# 모델 저장
save_path = './data/user_data/models/mnist_cnn_model.h5'
model.save(save_path)

# 모델 읽기
loaded_model = keras.models.load_model('./data/user_data/models/mnist_cnn_model.h5')
loss2, acc2 = loaded_model.evaluate(x_test, y_test, verbose=0)
print("로드한 모델의 정확도: ", acc2)
print('loss2: ', loss2)
print('acc2: ', acc2)

# 기존 자료 1개로 예측
idx = 0
x_one = x_test[idx:idx + 1] 
y_true = int(y_test[idx])
probs = loaded_model.predict(x_one, verbose=0)
y_pred = int(np.argmax(probs))
print(f'실제값: {y_true}, 예측값: {y_pred}')