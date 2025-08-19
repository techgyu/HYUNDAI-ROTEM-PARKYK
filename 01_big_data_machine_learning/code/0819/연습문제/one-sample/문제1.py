# [one-sample t 검정 : 문제1]  
# 영사기에 사용되는 구형 백열전구의 수명은 250시간이라고 알려졌다. 
# 한국연구소에서 수명이 50시간 더 긴 새로운 백열전구를 개발하였다고 발표하였다. 
# 연구소의 발표결과가 맞는지 새로 개발된 백열전구를 임의로 수집하여 수명시간 관련 자료를 얻었다. 
# 한국연구소의 발표가 맞는지 새로운 백열전구의 수명을 분석하라.
#    305 280 296 313 287 240 259 266 318 280 325 295 315 278

# 대립: 새로 개발된 백열전구의 수명은 300시간보다 길다.
# 귀무: 새로 개발된 백열전구의 수명은 300시간보다 짧다.
import scipy.stats as stats

one_sample = [305, 280, 296, 313, 287, 240, 259, 266, 318, 280, 325, 295, 315, 278]

# 정규성 확인
result_shapiro = stats.shapiro(one_sample) # p-value = 0.0179 < 0.05 : 정규성 위배
print('statistic:%.5f, pvalue: %.5f' % result_shapiro)

if result_shapiro[1] > 0.05:
    print("데이터가 정규성을 만족한다. 따라서 독립표본 t-검정을 수행한다.")
else:
    print("데이터가 정규성을 만족하지 않는다. 따라서 비모수 검정을 수행한다.")

# 독립표본 t-검정
result_ttest_1samp = stats.ttest_1samp(one_sample, popmean = 300.0)

print('statistic:%.5f, pvalue: %.5f' % result_ttest_1samp) # statistic:6.06248, pvalue: 0.00004 이므로 귀무가설 기각

if result_ttest_1samp[1] > 0.05:
    print("귀무가설을 기각할 수 없다.")
else:
    print("귀무가설을 기각하고 대립 가설을 채택한다.")

# 따라서, 새로 개발된 백열 전구의 수명은 300시간보다 길다.