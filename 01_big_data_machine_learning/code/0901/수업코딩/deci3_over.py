# 과적합 방지 처리 방법: train/test split, KFold, GridSearchCV ...
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
import numpy as np
from sklearn.model_selection import StratifiedKFold

iris = load_iris()
print(iris.keys())
train_data = iris.data
train_label = iris.target
print(train_data[:3])
print(train_label[:3])

# 분류 모델
dt_clf = DecisionTreeClassifier()
print(dt_clf)
dt_clf.fit(train_data, train_label)
pred = dt_clf.predict(train_data)
print('예측값: ', pred)
print('실제값: ', train_label)
print('분류 정확도: ', accuracy_score(train_label, pred)) # 분류 정확도:  1.0

# 과적합 발생
print('과적합 방지 방법 1 : train/test로 분리')
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, shuffle=True, random_state=121)

dt_clf.fit(x_train, y_train)
pred2 = dt_clf.predict(x_test)
print('예측값: ', pred)
print('실제값: ', y_test)
print('분류 정확도: ', accuracy_score(y_test, pred2)) # 분류 정확도:  0.9555
# 과적합이 해소 - 일반화된 모델, 포용성이 있는 모델이 생성됨


print('과적합 방지 방법 2 : KFold 교차 검증(cross validation)')
# K-Fold 교차 검증이 가장 일반적임
# train dataset에 대해 K개의 data fold set을 만들어 K번 만큼 학습 도중에 검증 평가를 수행하는 방법
features = iris.data
labels = iris.target
dt_clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 4)
kfold = KFold(n_splits=5)
cv_acc = []
print('iris_shape :', features.shape) # (150, 4)
n_iter = 0
for train_index, test_index in kfold.split(features, labels):
    # print('n_iter: ', n_iter)
    # print('train_index: ', len(train_index))
    # print('test_index: ', len(test_index))
    # n_iter += 1
    # kfold.split으로 변환된 인덱스를 이용해 학습용, 검증용 데이터 추출
    xtrain, xtest = features[train_index], features[test_index]
    ytrain, ytest = labels[train_index], labels[test_index]
    # 학습 및 예측
    dt_clf.fit(xtrain, ytrain) # train
    pred = dt_clf.predict(xtest) # test(validation data)
    n_iter += 1
    # 반복할 때 마다 정확도 측정
    acc = np.round(accuracy_score(ytest, pred), 3)
    train_size = xtrain.shape[0]
    test_size = xtest.shape[0]
    print('반복수:{0}, 교차검증 정확도:{1}, 학습데이터 수:{2}, 검증데이터 수:{3}'.format(n_iter, acc, train_size, test_size))
    print('반복수:{}, 검증세트: {}, 인덱스:{}'.format(n_iter, test_size, test_index))
    cv_acc.append(acc)

print('평균 검증 정확도 : ', np.mean(cv_acc)) # 평균 검증 정확도 :  0.9065999999999999

print('----------')
# StratifiedKFold : 불균형한 분포를 가진 데이터 집합을 위한 K-Fold 방식
# 대출사기, 이메일, 강우량, 코로나 백신 검사

skfold = KFold(n_splits=5)
cv_acc = []
print('iris_shape :', features.shape) # (150, 4)
n_iter = 0
for train_index, test_index in kfold.split(features):
    xtrain, xtest = features[train_index], features[test_index]
    ytrain, ytest = labels[train_index], labels[test_index]
    # 학습 및 예측
    dt_clf.fit(xtrain, ytrain) # train
    pred = dt_clf.predict(xtest) # test(validation data)
    n_iter += 1
    # 반복할 때 마다 정확도 측정
    acc = np.round(accuracy_score(ytest, pred), 3)
    train_size = xtrain.shape[0]
    test_size = xtest.shape[0]
    print('반복수:{0}, 교차검증 정확도:{1}, 학습데이터 수:{2}, 검증데이터 수:{3}'.format(n_iter, acc, train_size, test_size))
    print('반복수:{}, 검증세트: {}, 인덱스:{}'.format(n_iter, test_size, test_index))
    cv_acc.append(acc)

print('평균 검증 정확도 : ', np.mean(cv_acc)) # 평균 검증 정확도 :  0.9200000000000002

print('교차 검증 함수로 처리: ------')
data = iris.data
label = iris.target
score = cross_val_score(dt_clf, data, label, scoring = 'accuracy', cv=5)

print('교차 검증별 정확도: ', np.round(score, 2)) # 교차 검증별 정확도:  [0.96666667 0.96666667 0.9        0.93333333 1.        ]
# print('평균 검증 정확도 : ', np.round(np.mean(score), 2)) 평균 검증 정확도 :  0.95

print('과적합 방지 방법 3: GridSearchCV - 최적의 파라미터를 제공')
parameters = {'max_depth': [1, 2, 3], 'min_samples_split': [2, 3]} # dict type
grid_dtree = GridSearchCV(dt_clf, param_grid=parameters, cv=3, refit=True)

grid_dtree.fit(x_train, y_train) # 자동으로 복수의 내부 모형을 생성, 실행해 가며 최적의 파라미터를 찾는다.

import pandas as pd
scoreDf = pd.DataFrame(grid_dtree.cv_results_)
pd.set_option('display.max_columns', None)
print(scoreDf)
print('best parameter : ', grid_dtree.best_params_)
print('best accuracy : ', grid_dtree.best_score_) # 0.9428571428571427

# 최적의 parameter 를 탐제한 모델이 제공
estimator = grid_dtree.best_estimator_
pred = estimator.predict(x_test)
print('예측값: ', pred)
print('테스트 데이터: ', accuracy_score(y_test, pred))


