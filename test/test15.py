# [문항15] https://www.kaggle.com/c/bike-sharing-demand/data 에서 
# train.csv를 다운받아 bike_dataset.csv 으로 파일명을 변경한다. 
# 이 데이터는 어느 지역의 2011년 1월 ~ 2012년 12월 까지 날짜/시간. 
# 기온, 습도, 풍속 등의 정보를 바탕으로 1시간 간격의 자전거 대여횟수가 기록되어 있다.


# train / test로 분류 한 후 대여횟수에 중요도가 높은 칼럼을 판단하여 feature를 선택한 후, 
# 대여횟수에 대한 회귀 예측(RandomForestRegressor)을 하시오.


# (배점:10)

# 칼럼 정보 :
#   'datetime', 'season'(사계절:1,2,3,4),  
#   'holiday'(공휴일(1)과 평일(0)), 
#   'workingday'(근무일(1)과 비근무일(0)),
#   'weather'(4종류:Clear(1), Mist(2), Snow or Rain(3), Heavy Rain(4)),
#   'temp'(섭씨온도), 'atemp'(체감온도), 'humidity'(습도), 'windspeed'(풍속),
#   'casual'(비회원 대여량), 
#   'registered'(회원 대여량), 
#   'count'(총대여량)
# 참고 : casual + registered 가 count 임.

# 출력 사항 : 예측값 / 실제값, 결정계수, 예측 결과
# 답안 :

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np

# 1. 데이터 로딩
data = pd.read_csv('bike_dataset.csv')
print(data)

# 2.1 전처리
data = data.dropna()

# 2. 대여횟수에 중요도가 높은 칼럼을 판단하여 feature를 선택
feature = data.drop(['count', 'datetime', 'casual', 'registered', 'season', 'weather', 'workingday', 'holiday', 'temp', 'windspeed'], axis=1)
label = data['count']

print(feature)

# 3. train / test로 분류
x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.3, random_state=42)

# 4. 대여횟수에 대한 회귀 예측(RandomForestRegressor)을 하시오.
rfmodel = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rfmodel.fit(x_train, y_train)

ypred = rfmodel.predict(x_test)

# 5. 결과 출력
# 예측값 / 실제값 
result_df = pd.DataFrame({'실제값': y_test.values, '예측값': ypred})
print(result_df)
#       실제값         예측값
# 0     127  358.781646
# 1      13   36.557524
# 2     163  127.646890
# 3     233  294.006708
# 4     222  276.231163
# ...   ...         ...
# 3261   86   60.404179
# 3262  157  142.941686
# 3263  281  170.321604
# 3264  152  290.249519
# 3265    2   28.197907
# 결정 계수
print(f'\n결정계수(R2): {r2_score(y_test, ypred):.3f}') #  0.228
print(f'MSE: {mean_squared_error(y_test, ypred):.3f}') # MSE: 25124.047
# 예측 결과 - 중요도 분석
print('독립변수 중요도 순위 표')
importance = rfmodel.feature_importances_
indices = np.argsort(importance)[::-1]
ranking = pd.DataFrame({
    'Feature': feature.columns[indices],
    'Importance': importance[indices]
})
print(ranking)

# 독립변수 중요도 순위 표
#     Feature  Importance
# 0     atemp    0.580219
# 1  humidity    0.419781