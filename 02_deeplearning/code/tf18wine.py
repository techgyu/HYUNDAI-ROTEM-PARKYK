import os
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import plt
import pandas as pd
import numpy as np


# 모델 저장 폴더 설정
MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = MODEL_DIR + 'wine-{epoch:02d}-{val_loss:.4f}.hdf5'
# modelpath = 'abc.keras'

# 모델 학습 과정에서 특정 기준에 따라 자동으로 모델을 저장하는 callback 함수 설정
chkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', mode='max', save_best_only=True)

early_stop = EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(x_train, y_train, validation_split=0.2, epochs=1000, batch_size=32,
                    callbacks=[early_stop, chkpoint], verbose=2)
loss, acc = model.evaluate(x_test, y_test, batch_size=64, verbose=0)
print('훈련 후 모델 정확도: {:5.2f}%', format(100 * acc))

# 시각화
epoch_len = np.arange(len(history.epoch))

plt.plot(epoch_len, history.history['val_loss'], label='val_loss')
plt.plot(epoch_len, history.history['loss'], label='loss', c='red')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()

plt.plot(epoch_len, history.history['val_accuracy'], label='val_accuracy')
plt.plot(epoch_len, history.history['accuracy'], label='accuracy', c='red')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.show()

# best 모델로 예측
from tensorflow.keras.models import load_model
model = load_model(MODEL_DIR + modelpath)

new_data = x_test[:5, :]
print(new_data)
pred = model.predict(new_data)
print('예측 결과: \n', np.where(pred >= 0.5, 1, 0).ravel())