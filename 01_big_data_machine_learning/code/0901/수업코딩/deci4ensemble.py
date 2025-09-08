# 앙상블(Ensemble)
# 하나의 샘플 데이터를 여러 개의 분류기를 통해 다수의 학습 모델을 만들어 
# 학습시키고 학습 결과를 결합하므로써 과적합을 방지하고 정확도를 높이는 학습 기법

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
import numpy as np

cancer = load_breast_cancer()
x, y = cancer.data, cancer.target
print(x[:3])
print(y[:3])
print(np.unique(y))

# 0과 1의 비율 확인
counter = Counter(y)
total = sum(counter.values())
for cls, cnt in counter.items():
    print(f"class {cls} : {cnt}개({cnt/total})")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, stratify=y)
# stratify=y : 레이블 분포가 train / test 고르게 유지하도록 층화 샘플링
# 불균형 데이터에서 모델 평가가 왜곡되지 않도록 함

print('전체 분포: ', Counter(y)) # Counter({np.int64(1): 357, np.int64(0): 212})
print('train 분포: ', Counter(y_train)) # Counter({np.int64(1): 285, np.int64(0): 170})
print('test 분포: ', Counter(y_test)) # Counter({np.int64(1): 72, np.int64(0): 42})

# 개별 모델 생성 (스케일링 - 표준화)
# make_pipeline을 이용해 전처리와 모델을 일체형으로 관리
logi = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=1000, random_state=12))
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
dti = DecisionTreeClassifier(max_depth=5, random_state=12)

voting = VotingClassifier(
    estimators=[('LR', logi), ('KNN', knn), ('DT', dti)],
    voting='soft'
)

# 개별 모델 성능 확인
# for clf in [logi, knn, dti]:
#     clf.fit(x_train, y_train)
#     pred = clf.predict(x_test)
#     print(f'{clf.__class__.__name__} 정확도: {accuracy_score(y_test, pred):.4f}')
name_models = [('LR', logi), ('KNN', knn), ('DT', dti)]
for name, clf in name_models:
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    print(f'{name} 정확도: {accuracy_score(y_test, pred):.4f}')

voting.fit(x_train, y_train)
vpred = voting.predict(x_test)
print(f'VotingClassifier 정확도: {accuracy_score(y_test, vpred):.4f}')

# Option : 교차 검증으로 안정성 확인
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=12)
cv_score = cross_val_score(voting, x, y, cv=cv, scoring='accuracy')
print(f'voting 5겹 cv 평균: {cv_score.mean():.4f} (+-) {cv_score.std():.4f}')

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
print(classification_report(y_test, vpred, digits=4))
print(confusion_matrix(y_test, vpred))
print(roc_auc_score(y_test, voting.predict_proba(x_test)[:, 1]))

# GridSearchCV로 최적의 파라미터 찾기
from sklearn.model_selection import GridSearchCV
param_grid = {
    'LR__logisticregression__C': [0.1, 1.0, 10.0],
    'KNN__kneighborsclassifier__n_neighbors': [3, 5, 7],
    'DT__decisiontreeclassifier__max_depth': [3, 5, 7, 9]
}

gs = GridSearchCV(estimator=voting, param_grid=param_grid, cv=cv, scoring='accuracy')
gs.fit(x_train, y_train)
print('best params: ', gs.best_params_)
print('best cv accuracy : ', gs.best_score_)
best_voting = gs.best_estimator_
print('test accuracy(best) : ', accuracy_score(y_test, best_voting.predict(x_test)))
print('test ROC_AUC(best) : ', roc_auc_score(y_test, best_voting.predict_proba(x_test)[:, 1]))