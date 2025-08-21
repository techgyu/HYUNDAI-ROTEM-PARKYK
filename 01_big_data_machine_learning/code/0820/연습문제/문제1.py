# [ANOVA 예제 1]
# 빵을 기름에 튀길 때 네 가지 기름의 종류에 따라 빵에 흡수된 기름의 양을 측정하였다.
# 기름의 종류에 따라 흡수하는 기름의 평균에 차이가 존재하는지를 분산분석을 통해 알아보자.
# 조건 : NaN이 들어 있는 행은 해당 칼럼의 평균값으로 대체하여 사용한다.
# kind quantity
# 1 64
# 2 72
# 3 68
# 4 77
# 2 56
# 1 NaN
# 3 95
# 4 78
# 2 55
# 1 91
# 2 63
# 3 49
# 4 70
# 1 80
# 2 90
# 1 33
# 1 44
# 3 55
# 4 66
# 2 77

# 대립가설(H1): 기름의 종류에 따라 빵에 흡수된 기름의 평균에 차이가 있다.
# 귀무가설(H0): 기름의 종류에 따라 빵에 흡수된 기름의 평균에 차이가 없다.

import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'kind': [1, 2, 3, 4, 2, 1, 3, 4, 2, 1, 2, 3, 4, 1, 2, 1, 1, 3, 4, 2],
    'quantity': [64, 72, 68, 77, 56, None, 95, 78, 55, 91, 63, 49, 70, 80, 90, 33, 44, 55, 66, 77]
})

# 결측치 처리
data = data.fillna(data['quantity'].mean())

# 데이터 분리
kind_1 = data[data['kind'] == 1]['quantity']
kind_2 = data[data['kind'] == 2]['quantity']
kind_3 = data[data['kind'] == 3]['quantity']
kind_4 = data[data['kind'] == 4]['quantity']

# 정규성 검정
print(stats.shapiro(kind_1).pvalue) # 0.333 > 0.05 정규성 만족
print(stats.shapiro(kind_2).pvalue) # 0.656 > 0.05 정규성 만족
print(stats.shapiro(kind_3).pvalue) # 0.832 > 0.05 정규성 만족
print(stats.shapiro(kind_4).pvalue) # 0.912 > 0.05 정규성 만족

# 등분산성 검정
print(stats.levene(kind_1, kind_2, kind_3, kind_4).pvalue) # 0.326 > 0.05 등분산성 만족
print(stats.bartlett(kind_1, kind_2, kind_3, kind_4).pvalue) # 0.193 > 0.05 등분산성 만족

# 일원분산분석
print(stats.f_oneway(kind_1, kind_2, kind_3, kind_4).pvalue) # 0.848 > (유의 수준) 0.05 (귀무가설 채택)

# ANOVA 사후 검정(Post-hoc test)
# 분산 분석은 집단의 평균의 차이 여부만 알려줄 뿐, 
# 각 집단 간의 평균 차이는 알려주지 않는다.
# 각 집단 간의 평균 차이를 확인하기 위해 사후 검정 실시

# Tukey의 사후 검정(집단별 평균 차이 유의성 확인)
turResult = pairwise_tukeyhsd(endog=data.quantity, groups=data.kind)
print(turResult)  # 각 집단 쌍별로 평균 차이, p-value, 유의성 결과 출력

# 사후 검정 결과를 시각화 (신뢰구간 그래프)
turResult.plot_simultaneous(xlabel='mean', ylabel='group')
plt.show()        # 그래프 창 표시
plt.close()       # 그래프 창 닫기


# 결론: 기름의 종류에 따른 빵에 흡수된 기름의 평균 차이는 없다.[귀무채택]