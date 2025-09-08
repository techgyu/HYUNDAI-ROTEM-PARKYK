# Random Forest 분류/예측 알고리즘
# 분류 알고리즘으로 titanic dataset 사용해 survived냐 아니냐 예측
# bagging 사용 데이터 샘플링(bootstrap)을 통해 모델을 학습시키고, 결과를 집계(Aggregating)하는 방법
# 참고: 우수한 성능을 원한다면 Bossting, 오버피팅이 걱정된다면 Bagging의 방식을 써봐라

# titanic dataset : feature - (pclass, age, sex), label(survived)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/titanic_data.csv')
print(df.head(2))
print(df.info())
print(df.isnull().any())
df = df.dropna(subset=['Pclass', 'Age', 'Sex'])
print(df.shape)

# feature, label로 분리
df_x = df[['Pclass', 'Age', 'Sex']]
print(df_x.head(2)) # Sex를 숫자화
y = df['Survived']
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df_x.loc[:, 'Sex'] = encoder.fit_transform(df_x['Sex'])
print(df_x.head(2))

df_y = df['Survived']
print(df_y.head(2))
print()
train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.3, random_state=42)

model = RandomForestClassifier(criterion='entropy', n_estimators=500)
model.fit(train_x, train_y)
pred = model.predict(test_x)
print('예측값: ', pred[:10])
print('실제값: ', np.array(test_y[:10]))
print('맞춘 개수 :', sum(test_y == pred))
print('전체 대비 맞춘 비율 : ', sum(test_y == pred) / len(test_y) * 100)
print('분류 정확도 :', accuracy_score(test_y, pred))

# K-fold
cross_vali = cross_val_score(model, df_x, df_y, cv = 5)
print('교차 검증 정확도: ', cross_vali)
print('교차 검증 평균 정확도: ', np.mean(cross_vali))
print()
print('교차 검증 표준편차: ', np.std(cross_vali))

# 중요변수 확인
print('특성(변수) 중요도: ', model.feature_importances_)

import matplotlib.pyplot as plt
n_features = df_x.shape[1]
plt.barh(range(n_features), model.feature_importances_)
plt.xlabel('feature_importance_score')
plt.ylabel('features')
plt.yticks(np.arange(n_features), df_x.columns)
plt.ylim(-1, n_features)
plt.show()
plt.close()


