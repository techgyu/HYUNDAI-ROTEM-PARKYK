# 미돌 기반 클러스터링 비모수적 알고리즘이다.
# 일부 공간에 있는 점의 경우, 서로 밀접하게 밀집된 점 (인근 이웃이 많은 점)을
# 그룹화하여 저밀도 지역(가장 가까운 이웃이 너무 멀리 떨어져 있음)에 혼자 있는 이상점
# 가장 많이 인용되는 클러스터링 알고리즘 중 하나이다.

import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, DBSCAN

x, y = make_moons(n_samples=2000, noise=0.05, random_state=0)
print(x)
print('실제 군집 id: ', y[:10])
plt.show()
plt.close()

# KMeans로 군집화
km = KMeans(n_clusters=2, init='k-means++', max_iter=300, random_state=0)
pred1 = km.fit_predict(x)
print('예측 군집 id: ', pred1[:10])

# 시각화
def plotResultFunc(x, pr):
    plt.scatter(x[pr==0, 0], x[pr==0, 1], c='blue', marker='o', s=40, label='cluster-1')
    plt.scatter(x[pr==1, 0], x[pr==1, 1], c='red', marker='s', s=40, label='cluster-2')
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], marker="+", s=50, label='cluster center', c='black')
    plt.legend()
    plt.show()

plotResultFunc(x, pred1)

print()

# DBSCAN로 군집화
dm = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
pred2 = dm.fit_predict(x)

plotResultFunc(x, pred2)

# 군집화 : 고객 세분화, 예상치 탐지, 추천 시스템... 등의 효과적