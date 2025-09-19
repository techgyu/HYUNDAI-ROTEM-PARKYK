# mnist dataset으로 CNN 모델 작성
import numpy as np
import matplotlib.pyplot as plt
import keras as keras  # standalone keras 사용

# 데이터 로드
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 스케일링 및 차원 확장
x_train = (x_train.astype("float32") / 255.0).reshape((-1, 28, 28, 1))
x_test = (x_test.astype("float32") / 255.0).reshape((-1, 28, 28, 1))

# 모델 정의
model = keras.models.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    keras.layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.2),

    keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),

    keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),

    keras.layers.Flatten(), # Fully connected Layer에 연결하기 위한 1차원 변환

    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(units=32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(units=10, activation='softmax'),
])

print(model.summary())

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

# 시각화
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha = 0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha = 0.3)
plt.tight_layout()
# plt.show()
plt.savefig('./deeplearning/data/user_data/images/31_cnn_1.png')

# 단일 이미지 + 예측 확률 막대 그래프

classes = [str(i) for i in range(10)]
print(classes)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(x_one[0].squeeze(), cmap='gray')
plt.axis('off')
plt.title(f'True: {y_true} | Pred: {y_pred}')

plt.subplot(1, 2, 2)
plt.bar(classes, probs.ravel())
plt.title('Prediction Probabilities')
plt.xlabel('Classes')
plt.ylabel('Probability')
plt.ylim(0, 1.0)
for i, v in enumerate(probs.ravel()):
    plt.text(i, float(v) + 0.02, f"{float(v):.2f}", ha='center', fontsize=9)

plt.legend()
plt.grid(True, alpha = 0.3)
plt.tight_layout()
plt.savefig('./deeplearning/data/user_data/images/31_cnn_2.png')

# Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_pred_all = np.argmax(loaded_model.predict(x_test, verbose=0), axis=1)
print(y_pred_all)
cm = confusion_matrix(y_test, y_pred_all, labels=list(range(10)))
print(cm)
disp =  ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('./deeplearning/data/user_data/images/31_cnn_3.png')
