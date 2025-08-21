import pandas as pd

data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tsales.csv")
print(data)
print(data.dtypes)
# data['YMD']
# data[['YMD', 'AMT']]
# data
# data[조건] [[열 이름 목록]]

# 조건
print(
    data[data['YMD'] > 20200000]
    )

data2 = data[data['YMD'] > 20200000]
data2[['AMT', 'CNT']]

부서 = 관리부
부서 = 에이콘
부서 = 현대로템

busername == '관리부'
busername == '에이콘'
busername == '현대로템'


data['busername'] == '관리부'


data[data['busername'] == '관리부']

print(data)

