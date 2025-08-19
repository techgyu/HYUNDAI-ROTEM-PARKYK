# [one-sample t 검정 : 문제3] 
# https:\\www.price.go.kr\tprice\portal\main\main.do 에서 
# 메뉴 중  가격동향 -> 개인서비스요금 -> 조회유형:지역별, 품목:미용 자료(엑셀)를 파일로 받아 미용 요금을 얻도록 하자. 
# 정부에서는 전국 평균 미용 요금이 15000원이라고 발표하였다. 이 발표가 맞는지 검정하시오.

# 대립: 전국 평균 미용 요금은 15000원과 유의미한 차이가 있다.
# 귀무: 전국 평균 미용 요금은 15000원과 유의미한 차이가 없다.
import pandas as pd 
import scipy.stats as stats

data = pd.read_csv("./01_big_data_machine_learning/data/개인서비스지역별_동향.csv")
# print(data)

# 데이터 전 처리
data.drop(data.columns[0: 3], axis=1, inplace=True) # 불필요 열 제거
print("data1: \n",  data)
data = data.dropna(axis = 1) # 결측치 제거
print("data2: \n",  data)
prices = data.iloc[0]
print("data3: \n",  prices)
mean_prices = prices.mean()

# 정규성 확인
result_shapiro = stats.shapiro(mean_prices)
print('statistic:%.5f, pvalue: %.5f' % result_shapiro)

if result_shapiro[1] > 0.05:
    print("데이터가 정규성을 만족한다. 따라서 독립표본 t-검정을 수행한다.")
else:
    print("데이터가 정규성을 만족하지 않는다. 따라서 비모수 검정을 수행한다.")

# 정규성 검정 결론, 전국 미용실의 개별 요금 데이터를 갖고 한 검증이 아니기 때문에 무의미 함.

# 독립 표본 t-검정
result = stats.ttest_1samp(mean_prices, popmean = 15000) # statistic:6.67543, pvalue: 0.00001
print('statistic:%.5f, pvalue: %.5f' % result)

if result[1] > 0.05:
    print("귀무가설을 기각할 수 없다.")
else:
    print("귀무가설을 기각하고 대립 가설을 채택한다.")
# t-검정 결론: p-value가 유의수준 0.05보다 작으므로 전국 평균 미용 요금은 15000원과 유의미한 차이가 있다.

# Wilcoxon Signed-rank test는 정규성을 가정하지 않으로 설정 후 한번 더
wilcox_result = stats.wilcoxon(mean_prices - 15000) # 평균 15000과의 차이를 검정
print('wilcox_res:', wilcox_result)# (statistic=np.float64(0.0), pvalue=np.float64(3.0517578125e-05 = 0.00003...))
if wilcox_result[1] > 0.05:
    print("귀무가설을 기각할 수 없다.")
else:
    print("귀무가설을 기각하고 대립 가설을 채택한다.")
# 비모수 검정 결론: p-value가 유의수준 0.05보다 작으므로 전국 평균 미용 요금은 15000원과 유의미한 차이가 있다.