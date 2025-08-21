# 온도(3개의 집단)에 따른 음식점 매출액의 따른 평균 차이 검정
# 공통 칼럼이 년월일인 두 개의 파일을 조합을 해서 작업

# 대립가설(H1): 온도(3개의 집단)에 따른 음식점 매출액 평균 차이는 있다.
# 귀무가설(H0): 온도(3개의 집단)에 따른 음식점 매출액 평균 차이는 없다.

# 온도(3개의 집단)에 따른 음식점 매출액의 평균 차이 검정
import numpy as np
import pandas as pd
import scipy.stats as stats
from pingouin import welch_anova
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt

# 매출 자료 읽기
sales_data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tsales.csv', dtype={'YMD':'object'})
print(sales_data.head(3)) # 328 entries, 3 columns / YMD: 20190514
print(sales_data.info())

# 날씨 자료 읽기
wt_data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tweather.csv')
print(wt_data.head(3)) # 702 entries, 9 columns / tm:2018-06-01 
print(wt_data.info())

# sale에 매출액이 있으므로 데이터의 날짜를 기준으로 두개의 자료, 병합 작업 진행
wt_data.tm = wt_data.tm.map(lambda x: x.replace('-', ''))
print(wt_data.head(3)) # tm: 20180601

# 두 데이터 병합
frame = sales_data.merge(wt_data, how='left', left_on='YMD', right_on='tm')
print(frame.head(3), ' ', len(frame))
print(frame.columns) # ['YMD', 'AMT', 'CNT', 'stnId', 'tm', 'avgTa', 'minTa', 'maxTa', 'sumRn', 'maxWs', 'avgWs', 'ddMes']
data = frame.iloc[:, [0, 1, 7, 8]] # 날짜, 매출액, 최고기온, 강수량

# 온도 매핑
# bins = [1000, 3000, 5000, 7000, float('inf')]
# labels = ["1", "2", "3", "4"]
# df['jikwonpay'] = pd.cut(df['jikwonpay'], bins=bins, labels=labels, right=False)
print(data.maxTa.describe())
# 일별 최고 온도(연속형)변수를 이용해 명목형(구간화) 변수 추가
data['ta_gubun'] = pd.cut(data.maxTa,
                          bins = [-5, 8, 24, 37],
                          labels = [0, 1, 2]
                          )

print(data.head(3))
print(data.ta_gubun.unique()) # [2, 1, 0]
print(data.isnull().sum()) # 결측치 없음

# 최고 온도를 세 그룹으로 나눈 뒤, 등분산/정규성 검정
x1 = np.array(data[data.ta_gubun == 0].AMT)
x2 = np.array(data[data.ta_gubun == 1].AMT)
x3 = np.array(data[data.ta_gubun == 2].AMT)
print(x1[:5], len(x1))
print(x2[:5], len(x2))
print(x3[:5], len(x3))
print()
print(stats.levene(x1, x2, x3).pvalue) # 0.039(등분산 안 만족)
print("x1 정규성:", stats.shapiro(x1).pvalue) # 0.248(정규성 만족)
print("x2 정규성:", stats.shapiro(x2).pvalue) # 0.038(정규성 안 만족)
print("x3 정규성:", stats.shapiro(x3).pvalue) # 0.318(정규성 만족)

spp = data.loc[:, ['AMT', 'ta_gubun']]
print(spp.groupby('ta_gubun').mean()) # ta_gubun별 평균 매출액
print(pd.pivot_table(spp, 
                     index='ta_gubun', 
                     values='AMT', 
                     aggfunc='mean')
                     )
# ANOVA 진행
sp = np.array(spp)
group1 = sp[sp[:, 1] == 0, 0]
group2 = sp[sp[:, 1] == 1, 0]
group3 = sp[sp[:, 1] == 2, 0]

print(stats.f_oneway(group1, group2, group3).pvalue) # 0.000 < (유의 수준) 0.05 (귀무가설 기각)

# 참고 : 등분산성 만족 X : Welch's anova test
print(welch_anova(dv='AMT', between='ta_gubun', data=data))

# 참고 : 정규성 만족 X : Kruskal-Wallis H-test
print('kruskal: ', stats.kruskal(group1, group2, group3).pvalue) # 0.000 < (유의 수준) 0.05 (귀무가설 기각)

# 결론: 온도(3개의 집단)에 따른 음식점 매출액 평균 차이는 있다.[귀무기각]

# 사후 분석
tukeyResult = pairwise_tukeyhsd(endog=data.AMT, groups=spp['ta_gubun'])
print(tukeyResult)
tukeyResult.plot_simultaneous(xlabel='mean', ylabel='group')
plt.show()
plt.close()