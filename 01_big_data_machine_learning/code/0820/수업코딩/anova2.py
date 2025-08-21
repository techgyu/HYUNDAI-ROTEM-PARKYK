# 일원 분산 분석 연습
# 강남구에 있는 GS 편의점 3개 지역 알바생의 급여에 대한 평균의 차이가 있는가
# 대립 가설(H1): GS 편의점 3개 지역의 알바생 급여 평균에 차이가 있다.
# 귀무 가설(H0): GS 편의점 3개 지역의 알바생 급여 평균에 차이가 없다.
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/group3.txt"
# data = pd.read_csv(url, header=None)
# print(data)
data = np.genfromtxt(url, delimiter=',')
print(data, type(data), data.shape) # (22, 2)

# 3개 집단에 월급, 평균 얻기
gr1 = data[data[:, 1] == 1, 0]
gr2 = data[data[:, 1] == 2, 0]
gr3 = data[data[:, 1] == 3, 0]
print(gr1, ' ', np.mean(gr1)) # 316.625
print(gr2, ' ', np.mean(gr2)) # 256.444
print(gr3, ' ', np.mean(gr3)) # 278.0

# 정규성 검증
print(stats.shapiro(gr1).pvalue) # 0.333 > 0.05 정규성 만족
print(stats.shapiro(gr2).pvalue) # 0.656 > 0.05 정규성 만족
print(stats.shapiro(gr3).pvalue) # 0.832 > 0.05 정규성 만족

# 등분산성
print(stats.levene(gr1, gr2, gr3).pvalue) # 0.045 < 0.05 등분산성 불만족
print(stats.bartlett(gr1, gr2, gr3).pvalue) # 0.350 > 0.05 등분산성 만족

# 시각화
plt.boxplot([gr1, gr2, gr3], showmeans=True)
plt.show()
plt.close()

# Anova 검정 방법1 : anova_lm
df = pd.DataFrame(data, columns=['pay', 'group'])
print(df)

lmodel = ols('pay ~ C(group)', data=df).fit() # pay를 종속변수로, group을 독립변수로 설정
print(anova_lm(lmodel, type=2)) # 0.0435

# Anova 검정 방법2 : f_oneway -> 표 미 제공
f_statistic, p_value = stats.f_oneway(gr1, gr2, gr3)
print(f"f-statistic: {f_statistic}") # 3.711
print(f"p-value: {p_value}") # 0.043 < 0.05 귀무기각(알바생 급여 평균에 차이가 있다)

# 사후 검정
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukeyResult = pairwise_tukeyhsd(endog=df.pay, groups=df.group)
print(tukeyResult)
tukeyResult.plot_simultaneous(xlabel='mean', ylabel='group')
plt.show()
plt.close()