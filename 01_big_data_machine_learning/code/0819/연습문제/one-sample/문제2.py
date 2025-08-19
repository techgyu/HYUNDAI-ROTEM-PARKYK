# [one-sample t 검정 : 문제2] 
# 국내에서 생산된 대다수의 노트북 평균 사용 시간이 5.2 시간으로 파악되었다. 
# A회사에서 생산된 노트북 평균시간과 차이가 있는지를 검정하기 위해서 A회사 노트북 150대를 랜덤하게 선정하여 검정을 실시한다.
# 실습 파일 : one_sample.csv
# 참고 : time에 공백을 제거할 땐 ***.time.replace("     ", "")

# 대립: A 회사의 노트북 평균 사용 시간은 5.2시간과 유의미한 차이가 있다.
# 귀무: A 회사의 노트북 평균 사용 시간은 5.2시간과 유의미한 차이가 없다.
import pandas as pd
import scipy.stats as stats

data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/one_sample.csv")


# time 컬럼 공백 제거 후 숫자화 (안 되면 NaN)
data['time'] = pd.to_numeric(data['time'].astype(str).str.strip(), errors='coerce')

# 결측(공백 등) 제거
new_data = data.dropna(subset=['time'])
print(len(data)) # 결측치 날린 후 109개로만 처리

# 정규성 확인
result_shapiro = stats.shapiro(new_data.time) # p-value = 0.0179 < 0.05 : 정규성 위배
print('statistic:%.5f, pvalue: %.5f' % result_shapiro)

if result_shapiro[1] > 0.05:
    print("데이터가 정규성을 만족한다. 따라서 독립표본 t-검정을 수행한다.")
else:
    print("데이터가 정규성을 만족하지 않는다. 따라서 비모수 검정을 수행한다.")

# 독립 표본 t-검정
result = stats.ttest_1samp(new_data.time, popmean = 5.2)
print('statistic:%.5f, pvalue: %.5f' % result) # statistic:3.94606, pvalue: 0.00014

if result[1] > 0.05:
    print("귀무가설을 기각할 수 없다.")
else:
    print("귀무가설을 기각하고 대립 가설을 채택한다.")

# 따라서, A회사 노트북의 평균 사용 시간은 5.2시간과 유의미한 차이가 있다.