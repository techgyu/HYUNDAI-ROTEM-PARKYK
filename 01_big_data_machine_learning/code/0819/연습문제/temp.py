import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
import seaborn as sns

# 평균=중앙값이면서 왜도(꼬리) 형태를 띄는 데이터 예시
data = [1, 2, 2, 3, 5, 5, 8, 8, 9, 10]  # 10이 꼬리 역할
mean_val = np.mean(data)
median_val = np.median(data)
skew_val = skew(data)

print(f"평균: {mean_val}, 중앙값: {median_val}, 왜도: {skew_val:.2f}")

sns.displot(data, bins=5, kde=True) # 정규성을 만족하지 않아서 왜도로 나옴
plt.show()
plt.close()