# 집단 간 차이 분석: 평균 또는 비율의 차이를 분석하여 모집단의 특성을 추론합니다.
# T-test와 ANOVA의 차이점:
# - 두 집단의 평균 차이를 검정할 때는 T-test를 사용하여 T값으로 가설 검정을 합니다.
# - 세 집단 이상의 평균 차이를 검정할 때는 ANOVA를 사용하여 F값으로 가설 검정을 합니다.

# 핵심 아이디어:
# 집단 평균차이(분자)와 집단 내 변동성(표준편차, 표준오차 등, 분포)을 비교하여 
# 그 차이가 데이터의 불확실성(변동성)에 비해 얼마나 큰 지를 계산합니다.

# t 분포는 표본 평균을 이용해 정규 분포의 평균을 해석할 때 많이 사용한다.

# 대개의 경우, 표본의 크기는 30개 이하일 때 t 분포를 따른다.
# t 검정은 '두개 이하 집단의 평균의 차이가 우연의 의한 것인지 통계적으로 유의한 차이를
# 판단하는 통계적 절차이다.

# 실습1 - 어느 남성 집단의 평균키 검정
# 귀무(H0): 집단의 평균 키가 177이다.(모수) : 모집단의 평균과 샘플 데이터의 평균이 같다.
# 대립(H1): 집단의 평균 키가 177이 아니다.(비모수) : 모집단의 평균과 샘플 데이터의 평균이 다르다.
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# one_sample = [167.0, 182.7, 160.6, 176.8, 185.0]
# print(np.array(one_sample).mean()) # 174.42
# # 177.0과 174.42는 평균의 차이가 있느냐?

# result = stats.ttest_1samp(one_sample, popmean = 177.0)
# print('statistic:%.5f, pvalue: %.5f' % result) # statistic:-0.555, pvalue: 0.608 이므로 귀무가설 채택(174도 177로 본다)

# plt.boxplot(one_sample)
# sns.displot(one_sample, bins=10, kde=True, color='blue')
# plt.xlabel('data')
# plt.ylabel('count')
# plt.show()
# plt.close()


# print("---------------------------------------------------------------------------------")
# # 단일 모집단의 평균에 대한 가설검정(one samples t-test)
# # 실습2 - 단일 모집단의 평균에 대한 가설 검정
# # 중학교 1학년 1반 학생들의 시험결과가 담긴 파일을 읽어 국어 점수 평균
# # 귀무: 학생들의 국어 점수의 평균은 80이다.
# # 대립: 학생들의 국어 점수의 평균은 80아니다.
# data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/student.csv")
# print(data.head(3))
# print(data.describe())

# print(data['국어'].mean()) # 72.9
# # 아래 정규성 검정: one-sample t-test는 옵션
# print("정규성 검정: ", stats.shapiro(data.국어)) # pvalue=0.0129 이므로 정규성을 따르지 않는다.
# # 정규성 위배는 데이터 재가공 추천, 비모수 검정으로 대체 가능(Wilcoxon Signed-rank test)를 써야 더 안전
# # Wilcoxon Signed-rank test는 정규성을 가정하지 않음
# wilcox_result = stats.wilcoxon(data.국어 - 80) # 평균 80과의 차이를 검정
# print("wilcox_result: ", wilcox_result) # WilcoxonResult(statistic=np.float64(74.0), pvalue=np.float64(0.39777620658898905)) > 0.05이므로 귀무가설 채택

# res = stats.ttest_1samp(data['국어'], popmean=80)
# print('statistic:%.5f, pvalue: %.5f' % res) # statistic:-1.33218, pvalue: 0.19856 이므로 귀무가설 채택
# # 해석: 정규성 위배(p<0.05)이지만 t-test와 Wilcoxon 모두 귀무가설 기각 실패(평균 80과 유의한 차이 근거 부족).
# # t-test는 어느 정도 강건하나 비정규 분포이므로 비모수 결과(Wilcoxon)를 함께 제시하며 신중 해석.

# 실습3 - 여아 신생아 몸무게의 평균 검정 수행 babyboom.csv
# 여아 신생아의 몸무게는 평균이 2800(g)으로 알려져 왔으나 이보다 더 크다는 주장이 나왔다.
# 표본으로 여아 18명을 뽑아 체중을 측정하였다고 할 때 새로운 주장이 맞는지 검정해 보자.

# 대립 : 여아 신생의 몸무게는 평균이 2800(g)보다 크다.
# 귀무 : 여아 신생의 몸무게는 평균이 2800(g)이다.
data2 = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/babyboom.csv")
print(data2.head(3))
print(data2.describe())
fdata = data2[data2.gender == 1]
print(fdata.head(3))
print(len(fdata))
print(np.mean(fdata.weight), np.std(fdata.weight))
# 3132 vs 2800 둘 사이는 평균에 차이가 있는가?

# 정규성 검정 (하나의 집단일 때는 option)
print(stats.shapiro(fdata.iloc[:, 2])) # p-value = 0.0179 < 0.05 : 정규성 위배

# 정규성 시각화
# 1) histogram으로 확인
sns.displot(fdata.weight, kde=True) # 정규성을 만족하지 않아서 왜도로 나옴
plt.show()
plt.close()

# 2) Q-Q plot으로 확인 - 정규성을 확인하기 위해 사용
stats.probplot(fdata.weight, dist="norm", plot=plt)
plt.show()
plt.close()
# Wilcoxon Signed-rank test는 정규성을 가정하지 않음
wilcox_resBaby = stats.wilcoxon(fdata.weight - 2800) # 평균 2800과의 차이를 검정
print('wilcox_res:', wilcox_resBaby)# pvalue = 0.0342 < 0.05 이므로 귀무기각
print()

resBaby = stats.ttest_1samp(fdata.weight, popmean=2800) # pvalue = 0.03927 < 0.05 이므로 귀무기각
print('statistic:%.5f, pvalue: %.5f' % resBaby)
# 즉, 여 신생아의 평균 체중은 2800(g)보다 증가하였다.
