# 세 개 이상의 모집단에 대한 가설검정 – 분산분석
# ‘분산분석’이라는 용어는 분산이 발생한 과정을 분석하여 요인에 의한 분산과 요인을 통해 나누어진 각 집단 내의 분산으로 나누고 요인
# 에 의한 분산이 의미 있는 크기를 크기를 가지는지를 검정하는 것을 의미한다.
# 세 집단 이상의 평균비교에서는 독립인 두 집단의 평균 비교를 반복하여 실시할 경우에 제1종 오류가 증가하게 되어 문제가 발생한다.
# 이를 해결하기 위해 Fisher가 개발한 분산분석(ANOVA, ANalysis Of Variance)을 이용하게 된다.
# 분산의 성질과 원리를 이용하여 집단 간의 차이를 검정하는 방법이다.
# 즉, 평균을 직접 비교하지 않고 집단 내 분산과 집단 간 분산을 이용하여 집단의 평균이 서로 다른지
# 확인하는 방법이다.
# f-value = 그룹 간 분산(Between variance) / 그룹 내 분산(Within variance)

# * 서로 독립인 세 집단의 평균 차이 검정
# 실습 1) 세 가지 교육방법을 적용하여 1개월 동안 교육받은 교육생 80명을 대상으로 실기시험을 실시. three_sample.csv'

# 교육 방법 -> 실기 시험 평균에 영향을 줌
# 근데 교육 방법이 3개 이상이므로, t-test는 적용할 수 없음. 따라서 분산분석(ANOVA)을 실시한다.

# 독립 변수: 교육방법 (3가지 방법), 종속 변수: 실기 시험 점수
# 일원 분산 분석(One-way ANOVA) - 복수의 집단을 대상으로 집단을 구분하는 요인이 하나인 경우

# 대립 가설(Hypothesis 1): 세 가지 교육 방법에 따라 실기 시험 점수의 평균 차이가 있다.
# 귀무 가설(Hypothesis 0): 세 가지 교육 방법에 따라 실기 시험 점수의 평균 차이가 없다.

import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols #최소 제곱 법(기울과 절편을 구할 수 있다 -> 직선을 구함 -> 회귀 분석에서 중요함!)
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 데이터 구성
data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/three_sample.csv')

# 데이터 확인
print(data)
print(data.shape) # (80, 4)
print(data.describe()) # score의 의심 자료 발견!

# 이상치를 차트로 확인
# - 히스토그램
# plt.hist(data['score'])
# plt.title('Score Histogram')
# plt.show()
# plt.close()

# - 박스플롯
# plt.boxplot(data['score'])
# plt.title('Score Boxplot')
# plt.show()
# plt.close()

# 이상치 제거
data = data.query('score <= 100')
print("이상치 처리 후 데이터 개수:\n", len(data)) # 78개

# 데이터 분리
result = data[['method', 'score']]
print("method, score 추출 결과\n:", result)
m1 = result[result['method'] == 1]
m2 = result[result['method'] == 2]
m3 = result[result['method'] == 3]
print(m1[:3])
print(m2[:3])
print(m3[:3])

# 정규성 검정
score1 = m1['score']
score2 = m2['score']
score3 = m3['score']
print('score1: ', stats.shapiro(score1).pvalue)
print('score2: ', stats.shapiro(score2).pvalue)
print('score3: ', stats.shapiro(score3).pvalue) # 정규성 만족: p-value > 0.05

# 두 표본이 같은 분포를 따르는지 확인
print(stats.ks_2samp(score1, score2)) # 두 집단의 동일 분포 여부 확인
print(stats.ks_2samp(score1, score3)) # 두 집단의 동일 분포 여부 확인
print(stats.ks_2samp(score2, score3)) # 두 집단의 동일 분포 여부 확인

# 등분산성 검정(복수 집단 분산의 치우침 정도)
print("levene: ", stats.levene(score1, score2, score3).pvalue) # 등분산성 검정, 0.113
print("fligner: ", stats.fligner(score1, score2, score3).pvalue) # 등분산성 검정, 0.108
print("bartlett: ", stats.bartlett(score1, score2, score3).pvalue) # 등분산성 검정, 0.105

print('------------')
# 교차표 등 작성 가능 ...
reg = ols("score ~ C(method)", data=data).fit() # 단일 회귀 모델 작성
# reg = ols("data[score] ~ C(data['method'])", data=data).fit() # 단일 회귀 모델 작성

# 분산 분석표를 이용해 분산 결과 작성
table = sm.stats.anova_lm(reg, type=2) # anova linear regression model 생성
print(table) # p-value: 0.939639 > 0.05 이므로 귀무 채택

# ANOVA 사후 검정(Post-hoc test)
# 분산 분석은 집단의 평균의 차이 여부만 알려줄 뿐, 
# 각 집단 간의 평균 차이는 알려주지 않는다.
# 각 집단 간의 평균 차이를 확인하기 위해 사후 검정 실시
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Tukey의 사후 검정(집단별 평균 차이 유의성 확인)
turResult = pairwise_tukeyhsd(endog=data.score, groups=data.method)
print(turResult)  # 각 집단 쌍별로 평균 차이, p-value, 유의성 결과 출력

# 사후 검정 결과를 시각화 (신뢰구간 그래프)
turResult.plot_simultaneous(xlabel='mean', ylabel='group')
plt.show()        # 그래프 창 표시
plt.close()       # 그래프 창 닫기