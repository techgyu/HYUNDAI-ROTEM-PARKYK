# 데이터 프레임 병합(머지)
import numpy as np
from pandas import Series
from pandas import DataFrame
import pandas as pd
import random

df1 = DataFrame({
    'data': range(7), 
    'key': ['b', 'b', 'b', 'c', 'a', 'a', 'b']
})

print(df1)

df2 = DataFrame({
    'key': ['a', 'b', 'd'],
    'data': range(3), 
})

print(df2)

print(pd.merge(df1, df2, on='key'))

print(pd.merge(df1, df2, on='key', how='inner'))

print(pd.merge(df1, df2, on='key', how='outer'))

print(pd.merge(df1, df2, on='key', how='left'))

print(pd.merge(df1, df2, on='key', how='right'))

print('공통 칼럼이 없는 경우')
df3 = DataFrame({'key2': ['a', 'b', 'c'], 'data2': range(3)})
print(df3)
print(pd.merge(df1, df3, left_on='key', right_on='key2'))
print(pd.concat([df1, df2], axis=0, ignore_index=True))

s1 = pd.Series([0, 1], index=['a', 'b'])
s2 = pd.Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = pd.Series([5, 6], index=['f', 'g'])
print(pd.concat([s1, s2, s3], axis = 0))

print('그룹화: pivot_table')
data = {
    'city': ['강남', '강북', '강남', '강북'],
    'year': [2000, 2001, 2002, 2003],
    'pop': [3.3, 2.5, 3.0, 2.0]
}

df = DataFrame(data)
print(df)
print(df.pivot(index='city', columns='year', values='pop')) # 정렬 알고리즘의 pivot과 다른 개념

print(df.set_index(['city', 'year']).unstack()) # pivot과 유사한 결과
print(df.describe()) # 데이터 요약 통계

print()
print(df.pivot_table(index='city', columns='year', values='pop', aggfunc=np.sum)) # pivot_table 사용

print("pivot_table: pivot과 groupby의 중간적 성격")
print(df.pivot_table(index='city')) # pivot_table로 city별로 그룹화

print(df.pivot_table(index=['city'], aggfunc='mean')) # pivot_table로 city별로 그룹화
print(df.pivot_table(index=['city', 'year'], aggfunc=[len, 'sum'])) # pivot_table로 city, year별로 그룹화
print(df.pivot_table(index='city', values='pop', aggfunc='mean')) # city별로 pop의 평균값 계산
print(df.pivot_table(values='pop', index='year', columns='city')) # year별로 city의 pop 값 피벗
print(df.pivot_table(index='city', columns='year', values='pop', aggfunc=np.sum, fill_value=0)) # 결측치 0으로 대체

hap = df.groupby('city')
print(hap)
print(hap.sum()) # city별로 합계 계산
print(df.groupby('city').sum()) # city별로 합계 계산
print(df.groupby(['city', 'year']).mean()) # city별로 평균 계산