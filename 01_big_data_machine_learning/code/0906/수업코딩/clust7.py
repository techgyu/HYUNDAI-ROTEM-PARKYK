# iris dataset으로 지도학습(K-NN)/비지도학습(K-Means) 비교

from sklearn.datasets import load_iris
import numpy as np

iris_dataset = load_iris()
print(iris_dataset.keys())

print(iris_dataset['data'][:3])
print(iris_dataset['target'][:3])
print(iris_dataset['target_names'][:3])
print(iris_dataset['feature_names'])

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size=0.25, random_state=42)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape) # (112, 4) (38, 4)

print('지도학습 : K-NN ------------')
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

knnModel = KNeighborsClassifier(n_neighbors=3, weights = 'distance', metric = 'euclidean')
knnModel.fit(train_x, train_y) # featrue, label(tag, target, class)

predict_label = knnModel.predict(test_x)
print('예측값: ', predict_label) # [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0 0 0 1 0 0 2 1 0]
print('test acc : {:.3f}'.format(np.mean(predict_label == test_y))) # 0.973
print('acc : ', metrics.accuracy_score(test_y, predict_label))

# 새로운 데이터 분류
new_input = np.array([[6.1, 2.8, 4.7, 1.2]])
print(knnModel.predict(new_input)) # [1]
print(knnModel.predict_proba(new_input)) # 각 클래스에 속할 확률
dist, index = knnModel.kneighbors(new_input)
print('index : ', index, 'dist : ', dist)

print('\n비지도학습 : KMeans(데이터에 정답(label)이 없는 경우) ------------')
from sklearn.cluster import KMeans
kmeansModel = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=0)
kmeansModel.fit(train_x) # label X
print(kmeansModel.labels_)
print('0 cluster: ', train_y[kmeansModel.labels_ == 0]) # 군집 0에 속하는 실제값
print('1 cluster: ', train_y[kmeansModel.labels_ == 1]) # 군집 1에 속하는 실제값
print('2 cluster: ', train_y[kmeansModel.labels_ == 2]) # 군집 2에 속하는 실제값

# 이번엔 클러스터링에서 새로운 데이터 분류
new_input = np.array([6.1, 2.8, 4.7, 1.2]).reshape(1, -1)
clu_pred = kmeansModel.predict(new_input)
print(clu_pred)