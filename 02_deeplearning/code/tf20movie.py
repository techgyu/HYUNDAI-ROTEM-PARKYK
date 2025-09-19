# 영화 리뷰 이진 분류

# imdb datasset 불러오기
from tensorflow.keras.datasets import imdb
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import models, layers, regularizers
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("TensorFlow 버전:", tf.__version__)
print("Keras 데이터셋 경로 확인 중...")

# Windows에서 Keras 데이터셋 경로 확인
keras_cache_dir = os.path.expanduser('~/.keras')
datasets_dir = os.path.join(keras_cache_dir, 'datasets')

print(f"Keras 캐시 디렉토리: {keras_cache_dir}")
print(f"데이터셋 디렉토리: {datasets_dir}")

# 디렉토리 존재 여부 확인
if os.path.exists(datasets_dir):
    print(f"\n데이터셋 디렉토리 내용:")
    for item in os.listdir(datasets_dir):
        item_path = os.path.join(datasets_dir, item)
        if os.path.isfile(item_path):
            size = os.path.getsize(item_path) / (1024*1024)  # MB 단위
            print(f"  📄 {item} ({size:.2f} MB)")
        else:
            print(f"  📁 {item}/")
else:
    print("데이터셋 디렉토리가 아직 생성되지 않았습니다.")

print("\nIMDB 데이터셋 로딩 중... (처음 실행시 다운로드됩니다)")
(train_data, train_label), (test_data, test_label) = imdb.load_data(num_words=10000)

print(f"\n✅ 데이터 로딩 완료!")
print(train_data[0])  # 첫 번째 리뷰의 단어 인덱스 시퀀스
print(train_label[0])
print(f"훈련 데이터: {len(train_data[0])}개")

# 참고: 리뷰 데이터 하나를 원래 영어 단어로 보기
word_index = imdb.get_word_index()
print(word_index)
print(word_index.items())
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
print(reverse_word_index)

for i, (k, v) in enumerate(sorted(reverse_word_index.items(), key=lambda x: x[0])):
    if i >= 10:
        break
    print(k, ":", v)

decoded_review = ' '.join([reverse_word_index.get(i) for i in train_data[1]])
print(decoded_review)

# reverse_word_index = dict([(value, key) for (key, value) in imdb.get_word_index().items()])
# print("첫 번째 리뷰의 원래 영어 단어:")
# print(" ".join([reverse_word_index.get(i - 3, "?") for i in train_data[0]]))

# 데이터 준비
def vector_seq(sequences, dim = 10000):
    results = np.zeros((len(sequences), dim))
    for i, seq in enumerate(sequences): # 크기가 (len(sequences), dim)인 2차원 배열
        results[i, seq] = 1.  # 해당 단어 인덱스 위치를 1로 설정
    return results

x_train = vector_seq(train_data)
x_test = vector_seq(test_data)
print(x_train, ' ', x_train.shape)
y_train = train_label.astype('float32')
y_test = test_label.astype('float32')
print(y_train, ' ', y_train.shape)

# 모델 작성

model = models.Sequential()
model.add(layers.Input(shape=(10000, )))
model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())

# validation data 준비
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
print(len(x_val), ' ', len(partial_x_train))
history = model.fit(partial_x_train, partial_y_train, epochs=30, batch_size=512, validation_data=(x_val, y_val), verbose=2)

# 훈련과 검증 손실 / 정확도에 대한 시각화
history_dict = history.history
print(history_dict.keys())
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf() # 그래프 초기화
acc = history_dict['acc']
val_acc = history_dict['val_acc']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.clf() # 그래프 초기화
plt.close() # 그래프 창 닫기

pred = model.predict(x_test[:5])
print('예측값 : ', np.where(pred >= 0.5, 1, 0).ravel())
print('실제값 : ', y_test[:5])