# iris(붓꽃) dataset : 꽃받침과 꽃잎의 너비와 길이로 꽃의 종류를 3가지로 분류

import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns

iris_data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/iris.csv')

print(iris_data.info())
print(iris_data.head(3))
print(iris_data.tail(3))

# plt.scatter(iris_data['Sepal.Length'], iris_data['Petal.Length'])
# plt.xlabel('Sepal Length')
# plt.ylabel('Petal Length')
# plt.title('Iris Sepal vs Petal Length')
# plt.show()

print()
print(iris_data['Species'].unique()) # 리스트를 이용하여 중복된 종(species) 이름을 제거하여 출력
print(set(iris_data['Species'])) # 집합을 이용하여 중복 제거된 종(species) 이름 출력

cols = []
species_to_color = {'setosa': 1, 'versicolor': 2, 'virginica': 3}
cols = iris_data['Species'].map(species_to_color)

plt.scatter(iris_data['Sepal.Length'], iris_data['Petal.Length'], c=cols)
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.title('Iris Sepal vs Petal Length')
plt.show()

# 데이터 분포와 산점도 그래프 출력
iris_col = iris_data.loc[:, 'Sepal.Length': 'Petal.Width']
print(iris_col)

# 산점도
scatter_matrix(iris_col, diagonal='kde')
plt.show()

# seaborn을 활용한 그래프
sns.pairplot(iris_data, hue='Species', height=2)
plt.show()


# rug plot: 각 데이터 포인트의 위치를 나타내는 선을 추가하여 데이터 분포를 시각화
x = iris_data['Sepal.Length']
sns.rugplot(x=x)
plt.show()

# kernel density
sns.kdeplot(x=x)
plt.show()