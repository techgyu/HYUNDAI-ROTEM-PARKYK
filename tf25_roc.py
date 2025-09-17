# iris 다항 분류
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

iris = load_iris()
print(type(iris))  # <class 'sklearn.utils._bunch.Bunch'>
print(iris.keys())
x = iris.data
y = iris.target
print(x[:3])
print(y[:3])
print(x.shape, y.shape) # (150, 4) (150,)
print(set(y))

names = iris.target_names
print(names)
feature_names = iris.feature_names
print(feature_names)

# label : onehot
onehot = OneHotEncoder(categories='auto') # to_categorical, numpy:np.eye(), pd.get_dummies()
print(y.shape) # (150,)
y = onehot.fit_transform(y[:, np.newaxis]).toarray()
print(y.shape) # (150, 3)
print(y[:3])

# feature: 표준화

print(x[:2])
scaler = StandardScaler()
x_scale = scaler.fit_transform(x)
print(x_scale[:2])

x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size=0.3)

n_features = x_train[1]
n_classes = y_train[1]
print(n_features, n_classes)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def create_model_func(input_dim, out_dim, out_nodes, n, model_name = 'model'):
    print(input_dim, out_dim, out_nodes, n, model_name)

[create_model_func(n_features, n_classes, 10, n, 'model_{}'.format(n)) for n in range(3)]
