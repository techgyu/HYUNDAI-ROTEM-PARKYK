import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import numpy as np
import tensorflow as tf

# data load
train_df = pd.read_excel('hd_carprice.xlsx', sheet_name='train')
test_df = pd.read_excel('hd_carprice.xlsx', sheet_name='test')
print(train_df)
print(test_df)

# feature
x_train = train_df.drop(['가격'], axis=1)
x_test = test_df.drop(['가격'], axis=1)
# label
y_train = train_df['가격']
y_test = test_df['가격']

print(x_train.head(2))
print(x_train.columns)
print(x_train.shape)

print(set(x_train.종류))
print(set(x_train.연료))
print(set(x_train.변속기))

# 종류, 연료, 변속기 열에 대해서는 Label Encoder(), OneHotEncoder() 적용
transformer = make_column_transformer(OneHotEncoder(), ['종류', '연료', '변속기'], remainder='passthrough')
# remainder = 'passthrough' : 기본은 drop, 열이 transformer에 전달
transformer.fit(x_train)
x_train = transformer.transform(x_train)
x_test = transformer.transform(x_test)
print(x_train[:2], x_train.shape)  # (700, 16)
print(y_train.shape)  # (700,)