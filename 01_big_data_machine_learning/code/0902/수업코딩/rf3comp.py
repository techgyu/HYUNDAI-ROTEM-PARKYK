# 앙상블 기법의 모델들 성능비교
# - 랜덤 포레스트
# - 그래디언트 부스팅
# - XGBoost
# - LightGBM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/titanic_data.csv') 
# print(df.info())  
df.drop(columns=['PassengerId','Name','Ticket'], inplace=True)
print(df.describe())
print(df.isnull().sum())

# null 처리
df['Age'].fillna(df['Age'].mean(), inplace=True)
# Age 열에 결측치(NaN)가 있을 때, 그 결측값을 Age 열의 평균값으로 모두 채워줍니다.
# 즉, 결측치가 있던 부분이 평균값으로 대체되어 데이터가 완성됩니다.
# inplace=True는 원본 데이터프레임(df)에 바로 적용한다는 뜻입니다.
df['Cabin'].fillna('N', inplace=True)
df['Embarked'].fillna('N', inplace=True)
print(df.info())

print('1. Sex : ', df['Sex'].value_counts())
# 데이터프레임 df의 Sex 열에 있는 각 값(예: 'male', 'female')이 몇 번씩 나오는지(빈도)를 세어서 출력합니다.
print('2. Cabin : ', df['Cabin'].value_counts())
print('3. Embarked : ', df['Embarked'].value_counts())
df['Cabin'] = df['Cabin'].str[:1]
print()

print(df.groupby(['Sex', 'Survived'])['Survived'].count())
print('여성 생존율 : ',233/ (233+81))
print('남성 생존율 : ',109/ (109+468))

sns.barplot(x='Sex', y='Survived', data=df, ci=95)

# 성별 기준으로 Pclass별 생존 확률
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=df, ci=95)
plt.show()


# 나이별 기준으로 생존 확률
def getAgeFunc(age):
    msg = ''
    if age <= -1:
        msg = 'unknown'
    elif age <= 5:
        msg = 'baby'
    elif age <= 18:
        msg = 'teenager'
    elif age <= 65:
        msg = 'adult'
    else:
        msg = 'elder'
    return msg

df['Age_category'] = df['Age'].apply(lambda a : getAgeFunc(a))
print(df.head(2))

sns.barplot(x='Age_category', y='Survived', hue='Sex', data=df, order=['unknown', 'baby', 'teenager', 'adult', 'elder'])
plt.show()

del df['Age_category']

# 문자열 자료를 숫자화
from sklearn import preprocessing
def labelIncoder(datas):
    cols = ['Cabin', 'Sex', 'Embarked']
    for c in cols:
        lab = preprocessing.LabelEncoder()
        lab = lab.fit(datas[c])
        datas[c] = lab.transform(datas[c])
    return datas

df = labelIncoder(df)
print(df.head(3))
print(df['Cabin'].unique()) # [7 2 4 6 3 0 1 5 8]
print(df['Sex'].unique()) # [1 0]
print(df['Embarked'].unique()) # [3 0 2 1]

# feature / label
feature_df = df.drop(['Survived'], axis=1)
label_df = df['Survived']
print(feature_df.head(2))
print(label_df.head(2))

x_train, x_test, y_train, y_test = train_test_split(feature_df, label_df, test_size=0.2, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

logmodel = LogisticRegression(solver = 'lbfgs', max_iter=500).fit(x_train, y_train)

demodel = DecisionTreeClassifier().fit(x_train, y_train)
rfmodel = RandomForestClassifier().fit(x_train, y_train)

logpred = logmodel.predict(x_test)
print('logmodel acc :', accuracy_score(y_test, logpred))
print('demodel acc :', accuracy_score(y_test, demodel.predict(x_test)))
print('rfmodel acc :', accuracy_score(y_test, rfmodel.predict(x_test)))

