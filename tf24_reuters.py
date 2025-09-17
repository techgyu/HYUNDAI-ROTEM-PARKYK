from keras.datasets import reuters
# print()

(train_data, train_label), (test_data, test_label) = reuters.load_data(num_words=10000)
print(len(train_data), len(test_data)) # 8982 2246
print(train_data[0])  # 첫 번째 뉴스의 단어 인덱스 시퀀스
print(train_label[0])

# 실제 데이터 읽기
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
print(reverse_word_index)

decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print(decoded_review)

import numpy as np

def vector_seq(sequences, dim=10000):
    results = np.zeros((len(sequences), dim))
    for i, seq in enumerate(sequences):
        results[i, seq] = 1.
    return results

x_train = vector_seq(train_data)
x_test = vector_seq(test_data)
import sys
np.set_printoptions(threshold=sys.maxsize)  # 배열 전체 출력 설정
# print(x_test)

# one-hot encoding
# def to_onehot(labels, dim=46):
#     results = np.zeros((len(labels), dim))
#     for i, label in enumerate(labels):
#         results[i, label] = 1.
#     return results

# one_hot_train_labels = to_onehot(train_label)
# one_hot_test_labels = to_onehot(test_label)
# print(one_hot_train_labels[0])

from tensorflow.keras.utils import to_categorical
one_hot_train_labels = to_categorical(train_label)
one_hot_test_labels = to_categorical(test_label)
print(one_hot_train_labels[0])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import models
model = models.Sequential()
model.add(Input(shape=(10000, )))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# validation data
x_val = x_train[:1000]
partial_xtrain = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_ytrain = one_hot_train_labels[1000:]

history = model.fit(partial_xtrain, partial_ytrain, epochs=50, batch_size=128, validation_data=(x_val, y_val), verbose=2)

results = model.evaluate(x_test, one_hot_test_labels)
print(results)  # [loss, accuracy]

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
plt.clf()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, acc, 'bo', label='Acc')
plt.plot(epochs, val_acc, 'r', label='Validation Acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.clf()

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
