# 이미지 보강
# CNN 모델의 성능을 높이고 오버피팅을 극복할 수 있는 가장 좋은 방법은 다양한 유형의
# 학습 이미지 데이터 양을 늘리는 것입니다.
# 하지만 이미지 데이터의 경우 학습 데이터 량을 늘리는 것은 쉽지가 않습니다.
# 데이터 증강(data augmentation)은 학습 이미지의 개수를 늘리는 것이 아니고 학습 시 마다
import cv2
import matplotlib.pyplot as plt

# cv2.imread는 이미지를 RGB가 아닌 BGR로 받아오기 때문에 바꿔 주어야 함.
image = cv2.cvtColor(cv2.imread('deeplearning/data/user_data/반명함.jpg'), cv2.COLOR_BGR2RGB)
plt.imshow(image)