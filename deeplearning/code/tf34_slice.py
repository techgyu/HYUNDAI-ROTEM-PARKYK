# tf.data.Dataset.from_tensor_slices()
import numpy as np
import tensorflow as tf
import keras

x = np.random.sample((5,2))
print(x)

# dset = tf.data.Dataset.from_tensor_slices(x)
# print(dset)
dset = tf.data.Dataset.from_tensor_slices(x).shuffle(10000).batch(2)
print(dset)
for a in dset:
    print(a)
# ------------------------------

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(x_train.shape)
x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
print(x_train.shape)

# subclassing 모델 사용 
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
print(train_ds)
print(test_ds)

# model
class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

        self.conv1 = keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')
        self.pool1 = keras.layers.MaxPool2D(pool_size=(2, 2))

        self.conv2 = keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')
        self.pool2 = keras.layers.MaxPool2D(pool_size=(2, 2))

        self.flatten = keras.layers.Flatten()

        self.d1 = keras.layers.Dense(units=32, activation='relu')
        self.drop1 = keras.layers.Dropout(0.3)
        self.d2 = keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs): # init에서 선언한 층 호출해 네트워크 구성
        net = self.conv1(inputs)
        net = self.pool1(net)
        net = self.conv2(net)
        net = self.pool2(net)
        net = self.flatten(net)
        net = self.d1(net)
        net = self.drop1(net)
        net = self.d2(net)
        return net

model = MyModel()
temp_inputs = keras.Input(shape=(28, 28, 1))
model(temp_inputs) # 모델 빌드

loss_object = keras.losses.SparseCategoricalCrossentropy
optimizer = keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy'])
model.fit(x_train, y_train, batch_size = 2048, epochs=5, verbose=2)

score = model.evaluate(x_test, y_test)
print('test loss: ', score[0])
print('test acc :', score[1])
print('실제값: ', y_test[:2])
print('예측값: ', np.argmax(model.predict(x_test[:2]), axis=1))

# GradientType을 사용, model subprocessing 학습 방법
# 모델 손실과 성능을 측정할 지표 선택, 수집된 측정 지표를 바탕으로 최종 결과 출력을 위한 객체 생성
train_loss = keras.metrics.Mean() # 주어진 값의 (가중) 평균을 계산
train_accuracy = keras.metrics.SparseCategoricalAccuracy
test_loss = keras.metrics.Mean() # 주어진 값의 (가중) 평균을 계산
test_accuracy = keras.metrics.SparseCategoricalAccuracy

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

    for train_images, train_labels in train_ds:
        train_step(train_images, train_labels)

    template = '에포크 {}, 손실: {}, 정확도: {}, 테스트 손실: {}, 테스트 정확도: {}'
    print(template.format(epoch + 1, train_loss.result(), train_accuracy.result(), test_loss.result(), test_accuracy.result() * 100))