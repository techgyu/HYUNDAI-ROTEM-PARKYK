# pandas : 행과 열이 단순 정수형 인덱스가 아닌 레이블로 식별되는 데이터 구조를 제공하는 라이브러리
# 시계열 축약 연산, 누락 데이터 처리, SQL, 시각화 등 다양한 기능을 제공

import pandas as pd  # pandas 전체를 pd라는 별칭(alias)으로 불러옴. pd.DataFrame, pd.Series 등으로 사용.
# from pandas import Series, DataFrame  # pandas에서 Series, DataFrame 클래스만 직접 불러옴. Series(), DataFrame()으로 바로 사용 가능.
import numpy as np  # numpy 전체를 np라는 별칭으로 불러옴. np.array() 등으로 사용.

# Series : 1차원 배열과 유사한 객체로, 인덱스와 값으로 구성
# - 넘파이의 1차원 배열과 비슷하지만, 각 값에 대응하는 인덱스(레이블)를 가질 수 있음
# - 인덱스는 정수뿐만 아니라 문자열 등 다양한 레이블 사용 가능
# - 데이터(값)와 인덱스(레이블)로 구성되어 있어, 인덱스를 통해 값에 쉽게 접근 가능
# - 결측치(NaN) 처리, 브로드캐스팅, 벡터 연산 등 다양한 기능 지원
# - 예시:
#   import pandas as pd
#   s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
#   print(s['b'])  # 20

# list: 순서가 있고 중복을 허용하는 자료형
print("\nlist를 Series 객체로 변환")
obj = pd.Series([3, 7, -5, 4])
print("list: Series 객체(인덱스 미 지정): \n", obj)
obj = pd.Series([3, 7, -5, 4], index=['d', 'b', 'a', 'c'])
print("list: Series 객체(인덱스 지정): \n", obj)

# set: 순서가 없고 중복을 허용하지 않는 자료형
# obj = pd.Series({3, 7, -5, 4}) # 집합(set)은 순서가 없어서 불가능

# 튜플: 순서가 있으므로 Series 객체로 변환 가능
print("\n튜플을 Series 객체로 변환")
obj = pd.Series((3, 7, -5, 4))  # 튜플은 순서가 있으므로 가능
print("tuple: Series 객체(인덱스 미 지정): \n", obj)
obj = pd.Series((3, 7, -5, 4), index=['d', 'b', 'a', 'c'])
print("tuple: Series 객체(인덱스 지정): \n", obj)

# pandas Series 객체 연산
print("\nSeries 객체 연산")
print("python sum:\n", obj.sum())
print("numpy sum:\n", np.sum(obj))  # Series 객체는 numpy의 sum 함수로도 합산 가능
print("values:\n", obj.values)  # Series 객체의 값들만 추출
print("index:\n", obj.index)  # Series 객체의 인덱스(레이블)들만 추출

# pandas Series 슬라이싱 연산
print("\nSeries 객체 슬라이싱")
print("인덱스 1부터 2까지 슬라이싱:\n", obj[1:3])  # 인덱스 1부터 2까지 슬라이싱
print("인덱스 'a'에 해당하는 값 추출:\n", obj['a'])  # 인덱스 'a'에 해당하는 값 추출
print("인덱스 'a'에 해당하는 값 추출: obj[['a']]:\n", obj[['a']])  # 인덱스 'a'에 해당하는 값 추출

print("인덱스 'a'와 'b'에 해당하는 값 추출:\n", obj[['a', 'b']])  # 인덱스 'a'와 'b'에 해당하는 값 추출
print("인덱스 'a'부터 'b'까지 슬라이싱:\n", obj['a':'b'])  # 인덱스 'a'부터 'b'까지 슬라이싱


# pandas Series는 넘파이 배열과 달리, 인덱스를 직접 지정하면 그 인덱스(레이블)에 해당하는 값을 반환함.
# 만약 인덱스를 문자열 등으로 지정했다면 obj[0]은 '0'이라는 레이블을 찾으려는 것이지, 첫 번째 값을 의미하지 않음.
# 따라서 숫자 위치 기반 접근은 obj.iloc[pos]처럼 iloc 속성을 사용해야 함.

# print("obj[0]:\n", obj[0])  # 인덱스 0에 해당하는 값 추출(오류 발생) set.iloc[pos] 사용 추천 받음
print("obj.iloc[0]:\n", obj.iloc[0])  # 인덱스 0에 해당하는 값 추출, 숫자 인덱스 접근은 iloc 사용

# print("obj[[2, 3]]:\n", obj[[2, 3]])  # 인덱스 2와 3에 해당하는 값 추출(오류 발생)
print("obj.iloc[[2, 3]]:\n", obj.iloc[np.array([2, 3])])  # (재 확인 필요!!!!!!: 인덱스 2와 3에 해당하는 값 추출, numpy 배열로 인덱스 지정

print("대소 비교 연산:\n", obj > 0)  # 각 값이 0보다 큰지 비교하여 불리언 시리즈 반환
print("인덱스 'a'가 Series에 있는지 확인:\n", "a" in obj)  # 인덱스 'a'가 Series에 있는지 확인

# dict type으로 Series 객체 생성
# set을 빼고는 list, dict, tuple 모두 순서가 있음
print("\ndict type으로 Series 객체 생성")
names = {'mouse': 5000, 'keyboard': 2000, 'monitor': 30000}
print("생성한 dictionary 확인:\n", names, type(names))

obj3 = pd.Series(names) # dict type으로 Series 객체 생성
print("dict type으로 Series 객체 생성:\n", obj3)
print("길이 확인:\n", len(obj3),  # Series 객체의 길이(요소 개수) 확인
      "값 확인:\n", obj3.values,
      "인덱스 확인:\n", obj3.index)

obj3.index = pd.Index(['마우스', '키보드', '모니터'])  # 권장 방식  # 인덱스(레이블) 변경
print(obj3['마우스'])  # 인덱스 '마우스'에 해당하는 값 추출

# DataFrame : Series 객체를 여러 개 모아 2차원 배열 형태로 구성된 데이터 구조
# - 행(row)과 열(column)로 구성되어 있으며, 각 행과 열은 인덱스(레이블)를 가짐
# - 행은 Series 객체로 표현되며, 열은 Series 객체의 모음으로 표현됨
# - 데이터베이스의 테이블과 유사한 구조로, 행과 열 모두 인덱스를 가질 수 있음
# - 다양한 데이터 타입을 가질 수 있으며, 결측치(NaN) 처리, 브로드캐스팅, 벡터 연산 등 다양한 기능 지원

df = pd.DataFrame(obj3)  # Series 객체를 DataFrame으로 변환
print("생성한 dataframe 확인: \n", df)  # DataFrame 출력

data = {
    'name' : ['홍길동', '한국인', '신기해', '공기밥', '한가해'],
    'address' : ('역삼동', '신당동', '역삼동', '역삼동', '신사동'),
    'age' : [23, 25, 33, 30, 35]
}

frame = pd.DataFrame(data) # 딕셔너리로 DataFrame 생성
print("딕셔너리로 DataFrame 생성: \n", frame)  # DataFrame 출력

print("DataFrame의 열(컬럼) 확인: \n", frame['name'])  # DataFrame의 열(컬럼) 확인
print("DataFrame의 열(컬럼) 확인 (속성 접근): \n", frame.name)  # DataFrame의 열(컬럼) 확인 (속성 접근)

print("DataFrame의 열(컬럼) 확인 (속성 접근) 타입: \n", type(frame.name)) 

print("DataFrame 생성 시 열 순서 지정:")
pd.DataFrame(data, columns=['name', 'address', 'age'])  # DataFrame 생성 시 열 순서 지정

print("\ndata에 NaN 값 추가")
frame2 = pd.DataFrame(data, columns=['name', 'age', 'address', 'tel'], index=['a', 'b', 'c', 'd', 'e'])
print("생성한 DataFrame 확인: \n", frame2)  # NaN 값이 있는 DataFrame 출력

# Data 값 수정
print(frame2)
frame2['tel'] = '010-1234-5678'  # tel 열 추가, 모든 행에 동일한 값 할당
print("tel 열 추가 후 DataFrame 확인: \n", frame2)  # tel 열 추가 후 DataFrame 출력

val = pd.Series(['222-2222', '333-3333', '444-4444'], index=['b', 'c', 'e'])
frame2.tel = val  # 기존 'tel' 열에 새로운 Series 객체 할당
print("기존 'tel' 열에 새로운 Series 객체 할당 후 DataFrame 확인: \n", frame2)  # 기존 'tel' 열에 새로운 Series 객체 할당 후 DataFrame 출력

print("전치된 DataFrame 확인: \n", frame2.T) # DataFrame을 전치(transpose)하여 행과 열을 교환한 결과 출력

# 왜 numpy.ndarray로 나오는 것인지? series인데, 어디서 변환된 것인지?

print(frame2.values, type(frame2.values))  # DataFrame의 값들만 추출하여 출력
print(frame2.values[0, 1])  # DataFrame의 첫 번째 행, 두 번째 열의 값 추출
print(frame2.values[0: 2]) # DataFrame의 첫 번째와 두 번째 행의 값들 추출

# 행 삭제
frame3 = frame2.drop('d', axis=0) # 'd' 인덱스가 d인 행 삭제, axis=0은 행을 의미
print(frame3) # 행 삭제 후 DataFrame 출력

# 열 삭제
frame4 = frame2.drop('tel', axis=1)  # 'tel' 열 삭제
print(frame4)  # 열 삭제 후 DataFrame 출력

print("정렬 ---")
# DataFrame의 'age' 열을 기준으로 오름차순 정렬
print(frame2.sort_index(axis=0, ascending=True))  # 행(axis 0) 인덱스 기준 오름차순 정렬
print(frame2.sort_index(axis=1, ascending=True))  # 열(axis 1) 인덱스 기준 오름차순 정렬
print(frame2.sort_index(axis=0, ascending=False))  # 행(axis 0) 인덱스 기준 내림차순 정렬
print(frame2.sort_index(axis=1, ascending=False))  # 열(axis 1) 인덱스 기준 내림차순 정렬

print(frame2["address"].value_counts()) # 'address' 열의 값 개수 세기

print("문자열 자르기")
data = {
    "address": ["강남구 역삼동", "중구 신당동", "강남구 대치통"],
    "human_count": [23, 25, 15]
}

fr = pd.DataFrame(data)  # DataFrame 생성
print("문자열 자르기 전 DataFrame 확인: \n", fr)  # 문자열 자르기 전 DataFrame 출력
result1 = pd.Series([x.split()[0] for x in fr.address]) # 공백을 구분자로 문자열 자르기
print("문자열 자른 결과: \n", result1)  # [0]을 기준으로 문자열 자른 결과 출력

result2 = pd.Series([x.split()[1] for x in fr.address]) # 공백을 구분자로 문자열 자르기
print("문자열 자른 결과: \n", result2)  # [1]을 기준으로 문자열 자른 결과 출력

print(result1, result1.value_counts())  # result1의 값과 각 값의 개수 출력
print(result2, result2.value_counts())  # result2의 값과 각 값의 개수 출력

