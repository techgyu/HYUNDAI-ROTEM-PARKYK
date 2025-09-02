# 앙상블(Ensemble)
# 하나의 샘플 데이터를 여러 개의 분류기를 통해 다수의 학습모델을 만들어 학습시키고
# 학습결과를 결합하므로써 과적합을 방지하고 정확도를 높이는 학습기법

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

print('전체 분포 : ', Counter(y))
print('train 분포 : ', Counter(y_train))
print('test 분포 : ', Counter(y_test))

# 개별 모델 생성 (스케일링-표준화)
# make_pipeline을 이용해 전처리와 모델을 일체형으로 관리
logi = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)

knn = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier()
)

tree = DecisionTreeClassifier(max_depth=5, random_state=12)

voting = VotingClassifier(
    estimators=[('LR', logi), ('KNN', knn), ('DT', tree)],
    voting='soft'
)

# 개별 모델 성능
for clf in [logi, knn, tree]:
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    print(f'{clf.__class__.__name__} 정확도 : {accuracy_score(y_test, pred):.4f}')

voting.fit(x_train, y_train)
vpred = voting.predict(x_test)

print(f'VotingClassifier 정확도 : {accuracy_score(y_test, vpred):.4f}')

# 옵션 : 교차 검증으로 안정성 확인
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=12)
cv_score = cross_val_score(voting, x, y, cv=cv, scoring='accuracy')
print(f'voting 5겹 cv 평균: {cv_score.mean():.4f} (+-) {cv_score.std():.4f}')