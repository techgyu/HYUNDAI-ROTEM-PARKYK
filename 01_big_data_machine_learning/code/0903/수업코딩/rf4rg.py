# RandomForestRegressor : 정량적 예측 모델
# california_housing dataset 사용

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

housing = fetch_california_housing(as_frame=True) # DataFrame 형태로 반환
print(housing.data[:2])
print(housing.target[:2])
print(housing.feature_names)
df = housing.frame # as_frame = True 때문에 가능
print(df.head(2))

# feature / label로 분리
print("dddddddddddddddd", df)
x = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# train / test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
rfmodel = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rfmodel.fit(x_train, y_train)

ypred = rfmodel.predict(x_test)
print(f'MSE: {mean_squared_error(y_test, ypred):.3f}')
print(f'결정계수: {r2_score(y_test, ypred):.3f}')

print('독립변수 중요도 순위 표')
importance = rfmodel.feature_importances_
indices = np.argsort(importance)[::-1]
ranking = pd.DataFrame({
    'Feature': x.columns[indices],
    'Importance': importance[indices]
})
print(ranking)

# 간단한 튜닝으로 최적의 파라미터 찾기
# GridSearchCV: 정확하게 최적값 찾기에 적당. 파라미터가 많으면 계산량 폭발적 증가
from sklearn.model_selection import RandomizedSearchCV
# 연속적 값 처리 가능, 최적 조합 못 찾을 수 있다.
param_list= {
    'n_estimators': [200, 400, 600],
    'max_depth': [None, 10, 20, 30],
    'min_samples_leaf': [1, 2, 4], # 리프 노드 최소 샘플 수
    'min_samples_split': [2, 5, 10], # 노드 분할 최소 샘플 수
    'max_features': [None, 'sqrt', 'log2', 1.0, 0.8, 0.6] # 최대 특성 수
}


search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42), # 기준 모델을 이걸로 하고
    param_distributions=param_list, # 파라미터를 바꿔가면서 학습을 수행한다.
    n_iter=10,
    scoring='r2',
    cv=3,
    random_state=42
)
#!!!!!!!!!!! 시간 오래 걸리는 거 정상임!!!!!!!!!!!!!
search.fit(x_train, y_train) # 탐색 수행(학습)

print('best parmas: ', search.best_params_) # 가장 성능 좋은 걸 여기에 저장한다.
best_model = search.best_estimator_
print('best cv r^2(교차검증 평균 결정계수): ', search.best_score_)
print('best model 결정계수: ', r2_score(y_test, best_model.predict(x_test)))
# best parmas:  {'n_estimators': 600, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 0.6, 'max_depth': 30}
# best cv r^2(교차검증 평균 결정계수):  0.8038261770514334
# best model 결정계수:  0.8132457689591202