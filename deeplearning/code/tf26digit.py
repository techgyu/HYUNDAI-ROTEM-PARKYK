# 손글씨(숫자 이미지) 읽기
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open("deeplearning/data/user_data/seyun_nist.png")
# 원래 이미지 크기 28 * 28 크기로 리사이즈(MNIST 기준)
img = img.resize((28, 28))

# 흑백(0 ~ 255)으로 변환 후 numpy 배열로 변환
img = img.convert("L")

img = np.array(img.resize((28, 28), Image.Resampling.LANCZOS))
print(img.shape)  # (28, 28)
plt.imshow(img, cmap="gray")
# plt.show()
plt.close()

# (28 * 28) 이미지를 (1, 784) 벡터로 변환(Dense )
data = img.reshape([1, 784]).astype(np.float32)
print(data.shape)  # (1, 784)
print(data)

data = data / 255.0  # 0 ~ 1 정규화
print(data)

# 다시 시각화 (1, 784) -> (28 / 28)
plt.imshow(data.reshape(28, 28), cmap="Greys")
plt.show()
