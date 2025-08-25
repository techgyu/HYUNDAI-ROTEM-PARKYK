# 회귀분석 문제 2) 
# testdata에 저장된 student.csv 파일을 이용하여 세 과목 점수에 대한 회귀분석 모델을 만든다. 
# 이 회귀문제 모델을 이용하여 아래의 문제를 해결하시오.  수학점수를 종속변수로 하자.
#   - 국어 점수를 입력하면 수학 점수 예측
#   - 국어, 영어 점수를 입력하면 수학 점수 예측

import pandas as pd 
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api
import matplotlib.pyplot as plt
import seaborn as sns

# Data Loading
students = pd.read_csv("./01_big_data_machine_learning/data/student.csv")

# Data Info
# print(students.select_dtypes(include='number').columns) # ['국어', '영어', '수학']
# print(students.select_dtypes(include='number').info())
# print(students.select_dtypes(include='number').corr())

# 01. 국어 점수를 입력하면 수학 점수 예측
# 산점도 확인: 데이터가 너무 퍼져있어서 추세선을 그려도 의미가 없음(수학을 종속 변수로)
result1 = smf.ols(formula='수학 ~ 국어', data=students).fit()
# print(result1.summary())

# input1 = float(input("국어 점수를 입력하시면, 수학 점수를 예측해 드립니다. \n 국어 점수 입력: "))
# print("예측된 수학 점수: ", result1.predict(pd.DataFrame({'국어': [input1]})).values)

# 02. 국어, 영어 점수를 입력하면 수학 점수 예측(수학을 종속 변수로)
result2 = smf.ols(formula='수학 ~ 국어 + 영어', data=students).fit()
# print(result2.summary())

# print("국어, 영어 점수를 입력하시면, 수학 점수를 예측해 드립니다.")
# input2 = float(input("국어 점수 입력: "))
# input3 = float(input("영어 점수 입력: "))
# print("예측된 수학 점수: ", result2.predict(pd.DataFrame({'국어': [input2], '영어': [input3]})).values)

# 03. 1번째 모델 검정
print("1번째 모델 검정 결과: ")
print("R-squared: ", result1.rsquared) # 0.587, 국어 점수로 수학 점수의 58.7% 설명 가능(중간 ~ 높은 수준)
print("p-value: ", result1.pvalues.iloc[1]) 
# 8.160795225697283e-05 < 0.05
# 국어 점수와 수학 점수 사이에 유의한 선형 관계가 있다.

# 04. 2번째 모델 검정
print("\n2번째 모델 검정 결과: ")
print("R-squared: ", result2.rsquared) # 0.659, 국어, 영어 점수로 수학 점수의 65.9% 설명 가능(중간 ~ 높은 수준)
print("p-value: ", result2.pvalues.iloc[1]) 
# 0.6634015534997675 > 0.05
# 국어, 영어 점수와 수학 점수 사이에 유의한 선형 관계가 없다.

# 결론: 국어 점수만으로 충분히 수학 점수를 예측할 수 있다.
