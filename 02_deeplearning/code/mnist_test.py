import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 1. 모델 로드
model = keras.models.load_model('deeplearning/data/user_data/models/mnist_cnn_model.h5')

# 2. 이미지 로드 및 전처리
img = Image.open('deeplearning/data/user_data/seyun_nist_edited.png').convert('L')  # 흑백 변환
img = img.resize((28, 28))  # MNIST 크기로 리사이즈
img_arr = np.array(img).astype('float32') / 255.0  # 정규화
img_arr = img_arr.reshape(1, 28, 28, 1)  # 배치, 채널 차원 추가

# 3. 예측
probs = model.predict(img_arr)
pred = np.argmax(probs)

print(f'\n최첨단 AI MNIST 판정 모델 결과: {pred}')

# 4. 출력
if(pred != 5):
    print("\n이 글씨는 5로 볼 수 없습니다. 이 글씨를 쓴 친구분, 혹시 5 대신 외계어를 쓰신 건 아닌가요?\n")
    print("최첨단 AI도 포기한 글씨체... 친구분은 글씨체 교정이 시급합니다! 😂✍️")
    