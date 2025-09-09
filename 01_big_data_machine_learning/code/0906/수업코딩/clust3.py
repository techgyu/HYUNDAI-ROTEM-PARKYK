from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
print(iris_df.head(2))
print(iris_df.loc[0:4, ['sepal length (cm)', 'sepal width (cm)']])

from scipy.spatial.distance import pdist, squareform
dist_vec = pdist(iris_df.loc[:, ['sepal length (cm)', 'sepal width (cm)']], metric = 'euclidean')
print('dist_vec:', dist_vec)
print()
row_dist = pd.DataFrame(squareform(dist_vec))
print('row_dist : \n', row_dist)

from scipy.cluster.hierarchy import linkage
row_clusters = linkage(dist_vec, method = 'complete')
print('row_clusters : \n', row_clusters)

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters = 2, metric = 'euclidean', linkage = 'complete')
X = iris_df.loc[0:4, ['sepal length (cm)', 'sepal width (cm)']]
labels = ac.fit_predict(X)
print(X)