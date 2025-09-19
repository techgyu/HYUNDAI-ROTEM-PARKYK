# zoo dataset으로 다항 분류 모델 작성

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.utils import to_categorical # one-hot encoding 지원
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/zoo.csv')
print(data)

x_data = data.iloc[:, :-1].astype('float32')
y_data = data.iloc[:, -1].astype('float32')
print(x_data[:2], x_data.shape)
print(y_data[:2], y_data.shape)

nb_classes = len(set(y_data))
print('classes 범주: ', nb_classes)

# train / test split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42, stratify=y_data)

# Model
model = Sequential([
    Input(shape=(x_data.shape[1])),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(nb_classes, activation='softmax')  # 다중 분류에서는 출력층의 활성화 함수로 softmax 사용
])
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # 다중 분류에서는 손실함수로 categorical_crossentropy 사용

# callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# restore_best_weights=True: 학습 종료 후 가장 좋은 'val_loss' 상태로 모델 가중치 복원

# checkpoint
checkpoint = ModelCheckpoint('best_zoom_model.keras', monitor='val_loss', save_best_only=True)

history = model.fit(x_train, y_train, epochs=1000, validation_split=0.2, callbacks=[early_stop, checkpoint], verbose=1)

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f'최종 평가 :Loss:{loss:.4f}, Acc:{acc:.4f}')

# 학습 곡선 시각화
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
# plt.show()

plt.clf()  # 그래프 초기화

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
# plt.show()
plt.close()  # 그래프 창 닫기

# Confusion Matrix & Report
y_pred = np.argmax(model.predict(x_test), axis=-1)
print(y_pred)
print('Report:', classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix : \n', cm)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
# plt.show()
plt.close()  # 그래프 창 닫기

# best model 읽어 예측하기
from tensorflow.keras.models import load_model
best_model = load_model('best_zoom_model.keras')

loss, acc = best_model.evaluate(x_test, y_test, verbose=0)
print(f'Best Model 최종 평가 :Loss:{loss:.4f}, Acc:{acc:.4f}')

# 새로운 데이터 분류
new_data = np.array([1., 0., 0., 1., 0., 0., 1., 1., 1., 1., 1., 0., 0., 5., 0., 0., 1.])
new_data = np.array(new_data).reshape(1, -1)  # (1, feature_dim) 형태로 변환

probs = best_model.predict(new_data)
print('분류 확률 : ', probs)
pred_class = np.argmax(probs)
print('분류 결과 : ', pred_class)