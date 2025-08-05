# 재색인(reindexing)
# 기존의 인덱스를 새로운 인덱스로 변경하는 작업

import pandas as pd
import numpy as np

# Series의 재색인
print("\nSeries 재색인")
data1 = pd.Series([1, 3, 2], index=[1, 4, 2])
print("생성한 Series:\n", data1)
data2 = data1.reindex([1, 2, 4])
print("재색인된 Series:\n", data2)

print("재색인할 때 값")
data3 = data2.reindex([0, 1, 2, 3, 4, 5]) # 없는 인덱스는 NaN으로 채움
print("재색인 후 채운 Series:\n", data3)

# 대응 값이 없는 (NaN) 인덱스는 결측값인데 777로 채울 수 있음
data4 = data3.fillna(777)
print("결측값을 777로 채운 Series:\n", data4)

# 한번에 NaN 값을 다른 값으로 채우는 방법, NaN이 들어가지 않아서 int형으로 유지됨
data3 = data2.reindex([0, 1, 2, 3, 4, 5], fill_value=777)
print("재색인 후 NaN을 777로 채운 Series:\n", data3)

data3 = data2.reindex([0, 1, 2, 3, 4, 5], method='ffill')  # 이전 값으로 다음 값을 채움 (forward fill, 'ffill'은 'pad'와 동일)
print("재색인 후 이전 값으로 채운 Series:\n", data3)

data3 = data2.reindex([0, 1, 2, 3, 4, 5], method='pad')  # 이전 값으로 다음 값을 채움 ('pad'는 'ffill'의 별칭, 동작 동일)
print("재색인 후 이전 값으로 채운 Series:\n", data3)

data3 = data2.reindex([0, 1, 2, 3, 4, 5], method='bfill')  # 다음 값으로 이전 값을 채움 (backward fill)
print("재색인 후 다음 값으로 채운 Series:\n", data3) # 5번은 6번을 가져와야 하는데 6번이 없으므로 NaN으로 남음

# bool, 슬라이싱 관련 method: loc(), iloc()
# 복수 인덱싱: loc() 라벨 지원, iloc() 숫자 지원
print("\nSeries bool 처리")

df = pd.DataFrame(np.arange(12).reshape(4, 3), index=['1월', '2월', '3월', '4월'], columns=['강남', '강북', '서초'])
print("생성한 DataFrame:\n", df)

print(df['강남']) # 강남 열을 선택
print(df['강남'] > 3) # 강남이 3보다 큰지 여부를 bool로 반환
print(df[df['강남'] > 3]) # 강남이 3보다 큰 행만 출력
print(df.loc[:'2월', '강남']) # 2월까지의 강남 열을 선택
print(df.iloc[:2, 0]) # 0번째 열(강남)의 2월까지의 행을 선택
print()
print(df.iloc[1:3, 0:2]) # 1월부터 3월까지의 강남과 강북 열을 선택
print(df.iloc[2, :]) # 3월의 모든 열을 선택
print(df.iloc[:3, 2], type(df.iloc[:3, 2])) # 1월부터 3월까지의 서초 열을 선택
print(df.iloc[:3, 1:3]) # 1월부터 3월까지의 강북과 서초 열을 선택