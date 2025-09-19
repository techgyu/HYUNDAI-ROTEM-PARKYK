import keras
from keras.layers import BatchNormalization

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 구조 변경(차원)
print(x_train.shape, y_train.shape)  # (60000, 28, 28) (60000, 1)
x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
print(x_train.shape, y_train.shape)  # (60000, 28,

inputs = keras.layers.Input(shape=(28, 28, 1))
# Method 1
# # 모델 정의
# x = keras.layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs)
# x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)
# x = keras.layers.Dropout(0.2)(x)

# x = keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(x)
# x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)

# x = keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(x)
# x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)

# # Fully Connected Layers
# x = keras.layers.Flatten()(x) # 1차원 변환
# x = keras.layers.Dense(units=64, activation='relu')(x)
# x = keras.layers.Dropout(0.3)(x)
# x = keras.layers.Dense(units=32, activation='relu')(x)
# x = keras.layers.Dropout(0.2)(x)
# outputs = keras.layers.Dense(units=10, activation='softmax')(x)

# Method 2 - BatchNormalization : Conv/Dense 층 뒤에 추가
# use_bias = False
x = keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(inputs)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU()(x)
x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)
x = keras.layers.Dropout(0.2)(x)

x = keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(inputs)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU()(x)
x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)
x = keras.layers.Dropout(0.2)(x)

# Fully Connected Layers
x = keras.layers.Flatten()(x) # 1차원 변환
x = keras.layers.Dense(units=64, use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU()(x)
x = keras.layers.Dropout(0.3)(x)

outputs = keras.layers.Dense(10, activation='softmax')(x)

# 모델 객체 생성
model = keras.models.Model(inputs=inputs, outputs=outputs, name='mnist_cnn_func')

print(model.summary())