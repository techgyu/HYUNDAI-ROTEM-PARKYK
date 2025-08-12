import numpy as np

# 1번
data = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
print(np.flip(data))


import urllib
import requests
import bs4 as BeautilfulSoup
# # 2번
# try:
#       url = "http://www.naver.com"
#       page = urllib.request.urlopen(url)
           
#       soup = BeautifulSoup(page.read(), "html.parser") 
#       title = soup..find_all('li')
#       for i in range(0, 10):
#               print(str(i + 1) + ") " + title[i].a['title'])
# except Exception as e:
#       print('에러:', e)

# 3번
# import pandas as pd

# data = {
#     'product':['아메리카노','카페라떼','카페모카'],
#     'maker':['스벅','이디아','엔젤리너스'],
#     'price':[5000,5500,6000]
# }

# df = pd.DataFrame(data)
# print(df)
# df.('test', conn, if_exists='append', ②________________)

# 4번
from pandas import DataFrame

df = DataFrame(np.arange(12).reshape((4, 3)), index = ['1월', '2월', '3월', '4월'], columns = ['강남', '강북', '서초'])
print(df)

# 5번
# plt.show()

# 6번
# data = DataFrames(items)
# data.to_csv(test.csv, index=None , header=None)

# 7번
from pandas import DataFrame
import pandas as pd
frame = DataFrame({'bun':[1,2,3,4], 'irum':['aa','bb','cc','dd']}, index=['a','b', 'c','d'])
# 실행 결과 1
print(frame.T)
# 실행 결과 2
frame2 = frame.drop('d')   # 인덱스가 'd'인 행 삭제
print(frame2)

# 실행결과 1 :
#         a  b  c  d
# bun    1  2  3  4
# irum  aa  bb  cc  dd

# 실행결과 2 :
#   bun irum
# a    1  aa
# b    2  bb
# c    3  cc

# 8번
import pandas as pd
import pandas as Series
df = pd.read_csv('C:/github_personal/HYUNDAI-ROTEM-PARKYK/01_big_data_machine_learning/data/employee_data.csv')
print(df)

# 9번
data = {
    'juso':['강남구 역삼동', '중구 신당동', '강남구 대치동'],
    'inwon':[23, 25, 15]
}
df = DataFrame(data)
print(df)
results = Series.Series([x.split()[0] for x in df.juso])
print(results)

# 출력 결과 :
# 0    강남구
# 1    중구
# 2    강남구
# dtype: object

# 10번
# x 변수에는 1 2 3 4 5(list type)으로 1차원 배열
x = [1, 2, 3, 4, 5]
print(x, type(x))
# y 변수에는 2차원 배열(3행 1열)의 요소로 1, 2, 3을 저장 y 변수는 reshape 함수 사용
y = np.arange(1, 4).reshape(3, 1)
print(y)

print(x + y)

# 12번
df = DataFrame(np.random.randn(36).reshape(9, 4), columns = ['가격1', '가격2', '가격3', '가격4'])
print(df)
print(df['가격1'].mean())
print(df['가격2'].mean())
print(df['가격3'].mean())
print(df['가격4'].mean())

# 13번
from pandas import DataFrame
data = {"a": [80, 90, 70, 30], "b": [90, 70, 60, 40], "c": [90, 60, 80, 70]}

print(list(data["a"]))
print(list(data["b"]))
print(list(data["c"]))
a = data["a"]
b = data["b"]
c = data["c"]
d = [a, b, c]
print(d)
data = DataFrame(d, index=['국어', '영어', '수학']).T
print(data)
# 칼럼(열)의 이름을 순서대로 "국어", "영어", "수학"으로 변경한다.
# 아래 문제는 제시한 columns와 index 명을 사용한다.
# 1) 모든 학생의 수학 점수를 출력하기
print(data['수학'])
# 2) 모든 학생의 수학 점수의 표준편차를 출력하기
print(data['수학'].std())
# 3) 모든 학생의 국어와 영어 점수를 Series 타입이 아니라 DataFrame type으로 출력하기 (배점:10)
print(data[['국어', '영어']])

# 14번
import matplotlib.pyplot as plt
df = np.random.randn(1000)
# print(df)

# 3. 히스토그램 (가격 분포)
plt.figure()
plt.hist(df, bins=20)
plt.title('good')
plt.show()

# 15번
import pandas as pd
import pandas as Series
df = pd.read_csv('sales_data.csv')
df = df.pivot_table()