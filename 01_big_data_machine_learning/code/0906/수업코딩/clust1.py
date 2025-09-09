# 클러스터링 기법 중 계층적 군집화 이해

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')

np.random.seed(123)
var = ['X', 'Y']
labels = ['점0', '점1', '점2', '점3', '점4']
X = np.random.random_sample([5, 2]) * 10
df = pd.DataFrame(X, columns=var, index=labels)
print(df)

# plt.scatter(X[:, 0], X[:, 1], c = 'blue', marker = 'o', s=50)
# plt.grid(True)
# plt.show()
# plt.close()

from scipy.spatial.distance import pdist, squareform
# pdist : 배열에 있는 값을 이용해 각 요소들의 거리를 계산
# squareform : 거리 행렬을 정방 행렬 형태로 변환
dist_vec = pdist(df, metric = 'euclidean')
print('dist_vec : \n', dist_vec)

row_dist = pd.DataFrame(squareform(dist_vec), columns=labels, index=labels)
print('row_dist : \n', row_dist)

# 응집형: 자료 하나하나를 군집으로 보고 가까운 군집끼리 연결해 가는 방법. 상향식
# 분리형: 전체 자료를 하나의 군집으로 보고 분리해 나가는 방법

# linkage : 응집형 계층적 군집을 수행
from scipy.cluster.hierarchy import linkage
row_clusters = linkage(dist_vec, method='ward')

df = pd.DataFrame(row_clusters, columns=['클러스터id_1', '클러스터id_2', '거리', '클러스터멤버수'], index=['클러스터1', '클러스터2', '클러스터3', '클러스터4'])
print(df)

# linkage의 결과로 덴드로그램 작성
from scipy.cluster.hierarchy import dendrogram
row_dendr = dendrogram(row_clusters, labels=labels)
plt.tight_layout()
plt.ylabel('유클리드 거리')
plt.show()
plt.close()