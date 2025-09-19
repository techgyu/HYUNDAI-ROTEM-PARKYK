# Subclassing 모델 사용 - 데이터 섞기, Grad

# 1. 라이브러리 임포트
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


#--------------------------------------------------------------------------------------------
# 2. MNIST 데이터셋 로드
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 3. 데이터 구조 확인 및 전처리
print(x_train.shape) # 원본 훈련 데이터 shape 출력
x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0 # 4차원으로 reshape 및 정규화
x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0   # 테스트 데이터도 동일하게 처리
print(x_train.shape) # 전처리 후 shape 출력

# 4. 임의의 NumPy 배열 생성 및 출력
x = np.random.sample((5,2)) # 5x2 크기의 랜덤 배열 생성
print(x) # 생성된 배열 출력

# 5. tf.data.Dataset 객체 생성 및 셔플
# 넘파이 배열을 Dataset 객체로 변환하고 셔플
dset = tf.data.Dataset.from_tensor_slices(x).shuffle(10000) # 데이터셋을 섞어서 생성
print(dset) # Dataset 객체 정보 출력

# # 6. Dataset 객체의 각 요소 출력
# for a in dset:
#     print(a) # 셔플된 데이터셋의 각 요소 출력

# 7. 훈련/테스트 데이터셋 객체 생성 및 셔플, 배치화
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
print(train_ds)
print(test_ds)
#--------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------
# 8. Subclassing 방식으로 모델 정의
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3,3), padding='valid', use_bias=False)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))

        self.conv2 = tf.keras.layers.Conv2D(32, kernel_size=(3,3), padding='valid', use_bias=False)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2))

        self.flatten = tf.keras.layers.Flatten(dtype='float32')

        self.d1 = tf.keras.layers.Dense(32,activation='relu')
        self.drop1 = tf.keras.layers.Dropout(0.3)
        self.d2 = tf.keras.layers.Dense(10,activation='softmax')

    # 9. 모델의 순전파(call) 정의
    def call(self, inputs, training=None):
        net = self.conv1(inputs)
        net = self.pool1(net)
        net = self.conv2(net)
        net = self.pool2(net)
        net = self.flatten(net)
        net = self.d1(net)
        net = self.drop1(net, training=training)
        net = self.d2(net)
        return net
#--------------------------------------------------------------------------------------------
    


#--------------------------------------------------------------------------------------------
# 10. 모델 인스턴스 생성 및 빌드
model=MyModel()
temp_inputs = tf.keras.Input(shape=(28,28,1))
temp_outputs = model(temp_inputs)
model(temp_inputs)
#--------------------------------------------------------------------------------------------



# 11. 손실 함수, 옵티마이저 정의 및 모델 컴파일
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss=loss_object, metrics=['acc'])
#--------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------
# 12. 모델 학습
model.fit(x_train, y_train, batch_size=512, epochs=5, validation_split=0.2, verbose=2,
          max_queue_size=10, workers=10, use_multiprocessing=True) # process 기반 스레딩 처리

score = model.evaluate(x_test, y_test, verbose=0)
print('test loss : ', score[0])
print('test acc : ', score[1])
print('예측값 : ', np.argmax(model.predict(x_test[:2],1)))
print('실제값 : ', y_test[:2])
#--------------------------------------------------------------------------------------------


## gradient tape 운영 = 모델 서브 프로세싱 학습방법
# 모델 손실과 성능을 측저할 지표 선택.
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')  
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels): # 얘를 반복하면 loss를 최소화
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)



@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)

EPOCHS = 5
for epoch in range(EPOCHS):
    for train_images, train_labels in train_ds:
        train_step(train_images, train_labels)