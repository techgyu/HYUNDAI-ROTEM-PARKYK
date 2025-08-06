from pandas import Series, DataFrame
import pandas as pd
import numpy as np

s1 = Series([1, 2, 3], index=['a', 'b', 'c'])

s2 = Series([4, 5, 6, 7], index=['a', 'b', 'd', 'c'])

print("s1:\n", s1)
print("s2:\n", s2)

print(s1 + s2) # 인덱스가 일치하는 값끼리 더함
print(s1.add(s2)) # 인덱스가 일치하지 않는 경우 0으로 채워서 더함
print(s1.multiply(s2)) # 인덱스가 일치하지 않는 경우 1로 채워서 곱함
df1 = DataFrame(np.arange(9).reshape(3, 3), index=['서울', '대전', '대구'], columns=list('kbs'))
df2 = DataFrame(np.arange(12).reshape(4, 3), index=['서울', '대전', '제주', '수원'], columns=list('kbs'))
print("DataFrame df1:\n", df1)
print("DataFrame df2:\n", df2)

print(df1 + df2) # 인덱스가 일치하는 값끼리 더함
print(df1.add(df2, fill_value=0)) # 인덱스가 일치하지 않는 경우 0으로 채워서 더함
print(df1.multiply(df2, fill_value=1)) # 인덱스가 일치하지 않는 경우 1로 채워서 곱함

ser1 = df1.iloc[0]  # 서울 행을 Series로 선택
print("Series ser1:\n", ser1)
ser2 = df2.iloc[1]  # 대전 행을 Series로 선택
print("Series ser2:\n", ser2)
print(df1 - ser1) # df1에서 ser1을 빼기 (브로드캐스팅)


print('기술적 통계(평균, 분산, 표준편차 등)')
df = DataFrame([[1.4, np.nan], [7, -4.5], [np.nan, None], [0.5, -1]], columns=['one', 'two'])
print("DataFrame df:\n", df)
print("평균:\n", df.mean())  # 각 열의 평균
print("분산:\n", df.var())  # 각 열의 분산
print("표준편차:\n", df.std())  # 각 열의 표준
print("최대값:\n", df.max())  # 각 열의 최대값
print("최소값:\n", df.min())  # 각 열의 최소값
print("합계:\n", df.sum())  # 각 열의 합계
print("중앙값:\n", df.median())  # 각 열의 중앙값
print("최대값의 인덱스:\n", df.idxmax())  # 각 열의 최대값의 인덱스
print("최소값의 인덱스:\n", df.idxmin())  # 각 열의 최소값의 인덱스
print("상관계수:\n", df.corr())  # 각 열의 상관계수
print("공분산:\n", df.cov())  # 각 열의 공분산
print("df.isnull: \n", df.isnull())  # 결측값 여부 확인
print("df.notnull: \n", df.notnull())  # 결측값이 아닌 여부 확인
print("df.drop(0): \n", df.drop(0))  # 0번째 행 제거
print("df.dropna(): \n", df.dropna())  # 결측값이 있는 행 제거
print("df.dropna(how='any'): \n", df.dropna(how='any'))  # 결측값이 하나라도 있는 행 제거
print("df.dropna(how='all'): \n", df.dropna(how='all'))  # 모든 값이 결측인 행 제거
print("df.dropna(subset=['one']): \n", df.dropna(subset=['one']))  # 'one' 열에 결측값이 있는 행 제거
print("df.dropna(axis=1): \n", df.dropna(axis=1))  # 결측값이 있는 열 제거
# print("df.dropna(axis='rows'): \n", df.dropna(axis='rows'))  # 결측값이 있는 행 제거
print("df.dropna(axis='columns'): \n", df.dropna(axis='columns'))
print(df.fillna(0))  # 결측값을 0으로 채움

print('기술적 통계(평균, 분산, 표준편차 등)')
print(df.sum())  # 각 열의 합계
print(df.sum(axis=1))  # 각 행의 합계
print(df.describe())  # 기술적 통계 요약
print(df.info())  # DataFrame 정보

print('재구조화, 구간 설정, 그룹 별 연산(agg 함수)')
df = DataFrame(1000 + np.arange(6).reshape(2, 3), index=['서울', '대전'], columns=['2020', '2021', '2022'])
print("DataFrame df:\n", df)
print("df.T:\n", df.T)  # 전치
# stack, unstack
df_row = df.stack() # 행을 열로 변환
print("df_row:\n", df_row)  # 행을 열로 변환
df_col = df_row.unstack()  # 열을 행으로 변환(복원)
print("df_col:\n", df_col)  # 열을 행으로 변환(복원)

print()
# 구간 설정: 연속형 자료를 범주화
price = [10.3, 5.5, 7.8, 3.6] 
cut = [3, 7, 9, 11] # 구간 기준 값
result_cut = pd.cut(price, cut)  # 구간 설정
print("price:", price)
print("cut:", cut)
print("result_cut:", result_cut)  # 구간 설정 결과
print(pd.value_counts(result_cut))  # 구간별 빈도수

datas = pd.Series(np.arange(1, 1001))
print("datas:\n", datas)
print("datas.head(10):\n", datas.head(10))  # 처음 10개 데이터
print("datas.tail(10):\n", datas.tail(10))  # 마지막 10개 데이터
print("datas.sample(10):\n", datas.sample(10))  # 무작위로 10개 데이터 샘플링
print("datas.sample(10, random_state=1):\n", datas.sample(10, random_state=1))  # 무작위로 10개 데이터 샘플링 (랜덤 시드 설정)
print("datas.sample(frac=0.1):\n", datas.sample(frac=0.1))  # 전체 데이터의 10% 샘플링

result_cut2 = pd.qcut(datas, 3)  # 분위수 구간 설정 (datas와 길이 같게!)
print("result_cut2:", result_cut2)  # 분위수 구간 설정 결과
print(pd.value_counts(result_cut2))  # 분위수 구간별 빈도수
print("result_cut2.categories:", result_cut2.cat.categories)  # 분위수 구간의 범주

print('-----------')
group_col = datas.groupby(result_cut2)  # 구간별 그룹화 (이제 에러 없음)
print("group_col:\n", group_col)  # 구간별 그룹화 결과
print("group_col.mean():\n", group_col.mean())  # 구간별 평균
print("group_col.sum():\n", group_col.sum())  # 구간

print("group_col.agg(['mean', 'sum', 'count', 'std', 'min'])")  # 구간별 평균, 합계, 개수
print(group_col.agg(['mean', 'sum', 'count', 'std', 'min']))  # 구간별 평균, 합계, 개수

def myFunc(gr):
    return{'count:': gr.count(),
        'mean': gr.mean(),
        'std': gr.std(),
        'min': gr.min(),
        'max': gr.max()
        }

print(group_col.apply(myFunc))  # 사용자 정의 함수 적용
print("group_col.apply(myFunc).unstack():\n", group_col.apply(myFunc).unstack())  # 사용자 정의 함수 적용 후 unstack
print("group_col.apply(myFunc).unstack().T:\n", group_col.apply(myFunc).unstack().T)  # 사용자 정의 함수 적용 후 unstack 및 전치

