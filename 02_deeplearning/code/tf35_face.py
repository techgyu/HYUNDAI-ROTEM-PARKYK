# CNN: 성인 남녀 얼굴 이미지 분류 - 이항 분류
import cv2, os, numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

image_dir = '/home/user/Desktop/github_personal/HYUNDAI-ROTEM-PARKYK/deeplearning/data/testdata_utf8/person_img'
xdata, ydata = [], []

# 남녀, 구분 라벨 구하기 - 파일명에서 성별 추출

for file in os.listdir(image_dir):
    if not (file.lower().endswith('.jpg') or file.lower().endswith('.png') or file.lower().endswith('.jpeg')):
        continue  # 이미지 파일이 아니면 건너뜀
    try:
        gender = file.split('_')[1]
        img_path = os.path.join(image_dir, file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"이미지 읽기 실패: {img_path}")
            continue
        img = cv2.resize(img, (64, 64))  # 모델 입력 shape와 맞춤
        xdata.append(img)
        ydata.append(gender)
    except Exception as e:
        print(f"파일 처리 실패: {file}, 에러: {e}")
        continue

# print(xdata[0])
# print(ydata[0])


xdata = np.array(xdata) / 255.0
ydata = np.array(ydata).astype('int32')

if len(xdata) == 0 or len(ydata) == 0:
    print("이미지 데이터가 없습니다. 폴더 경로, 파일명, 확장자를 확인하세요.")
    exit()

# train / test
x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.2, random_state=42)
print(x_train.shape, y_train.shape) # (80, 32, 32, 3) (80,)
print(x_test.shape, y_test.shape)   # (20, 32, 32, 3) (20,)z

# model
from keras.models import Sequential, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Input(shape=(64, 64, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # 이진 분류
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, batch_size=512, validation_split=0.1, verbose=2)

loss, acc = model.evaluate(x_test, y_test)
print(f'test acc: {acc:.4f}, loss: {loss:.4f}')

# 예측
pred = model.predict(x_test[:5])
print('예측값 : ', (pred > 0.5).astype('int32').flatten())  # 0.5 기준 이진 분류
print('실제값 : ', y_test[:5])

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.savefig('deeplearning/data/user_data/images/face_cnn_accuracy.png')
plt.close()

pred_classes = (pred >= 0.5).astype('int32').reshape(-1)
true_classes = y_test

plt.figure(figsize=(20, 4))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_test[i])
    
    # 예측 값과 실제값 표시
    is_correct = pred_classes[i] == true_classes[i]
    label = 'Female' if true_classes[i] == 0 else 'Male'
    prediction = 'Female' if pred_classes[i] == 1 else 'Male'
    title_color = 'black' if is_correct else 'red'
    plt.title(f'Pred: {prediction}\nTrue:{label}', color=title_color)
    plt.axis('off')

plt.tight_layout()
plt.savefig('deeplearning/data/user_data/images/face_cnn_predictions.png')