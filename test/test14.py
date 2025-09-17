# [문항14] titanic data를 사용하여 아래의 지시문에 따라 의사결정나무분류 모델을 작성하고 결과를 출력하시오.
# (배점:10)
# import pandas as pd
# data = pd.read_csv('titanic_data.csv', usecols=['Survived', 'Pclass', 'Sex', 'Age','Fare'])
# print(data.head(2), data.shape)    # (891, 12)
# data.loc[data["Sex"] == "male","Sex"] = 0
# data.loc[data["Sex"] == "female", "Sex"] = 1
# print(data["Sex"].head(2))
# print(data.columns)

# feature = data[["Pclass", "Sex", "Fare"]]
# label = data["Survived"]

# 이하 소스 코드를 적으시오.
# 1) train_test_split (7:3), random_state=12
# 2) 의사결정나무 클래스를 사용해 분류 모델 작성
# 3) 예측결과로 분류 정확도를 출력
# 답안 :

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. 데이터 로딩
data = pd.read_csv('titanic_data.csv', usecols=['Survived', 'Pclass', 'Sex', 'Age','Fare'])
print(data.head(2), data.shape)    # (891, 12)

# 2. 성별 수치화
data.loc[data["Sex"] == "male","Sex"] = 0
data.loc[data["Sex"] == "female", "Sex"] = 1
print(data["Sex"].head(2))

# 3. 전체 컬럼 확인
print(data.columns)

# 4. 특징, 라벨 설정
feature = data[["Pclass", "Sex", "Fare"]]
label = data["Survived"] # class names

# 5. 1) train_test_split (7:3), random_state=12
x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.3, random_state=12)

# 6. 2) 의사결정나무 클래스를 사용해 분류 모델 작성

model = DecisionTreeClassifier(criterion = 'entropy', 
                                    max_depth = 5, 
                                    random_state = 0 
                                    )

# max_depth: 트리의 최대 깊이를 지정
# min_samples_split: 노드를 분할하기 위한 최소한의 샘플 데이터 수로 과적합 제어
# min_samples_leaf: 말단 노드(leaf)가 되기 위한 최소한의 샘플 데이터 수. 과적합 제어
# min_weight_fraction_leaf: 말단 노드(leaf)가 되기 위한 최소한의 샘플 데이터 비율. 과적합 제어
# max_features: 최적 분할을 위한 최대 특성 수. 과적합 제어

print(model)
model.fit(x_train, y_train) # supervised learning

# 7. 3) 예측결과로 분류 정확도를 출력
y_pred = model.predict(x_test)

print('예측값 : ', y_pred)
print('실제값 : ', y_test)

print('총 갯수:%d, 오류 수: %d' % (len(y_test), (y_test != y_pred).sum())) # 총 갯수:45, 오류 수: 1
print('분류 정확도 확인 1 : ')
print("%.5f" % accuracy_score(y_test, y_pred)) # 1.0

print('분류 정확도 확인 2 : ')
con_mat =  pd.crosstab(y_test, y_pred, rownames=['예측값'], colnames=['관측값']) # 1.0
print(con_mat)

print('분류 정확도 확인 3 : ')
print('test : ', model.score(x_test, y_test)) #  0.75
print('train :', model.score(x_train, y_train)) # 두 개의 값 차이가 크면 과적합 의심 0.84