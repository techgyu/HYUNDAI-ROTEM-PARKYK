# Pandas의 DataFrame 관련 연습문제
import numpy as np
from pandas import Series
from pandas import DataFrame
import pandas as pd
import random

# 문제 1 - a)표준정규분포를 따르는 9 X 4 형태의 DataFrame을 생성하시오.
print("\n문제 1 - a)표준정규분포를 따르는 9 X 4 형태의 DataFrame을 생성하시오.")
q1_frame = DataFrame(np.random.randn(9, 4))
print("frame: \n", q1_frame)

# 문제 1 - b)a에서 생성한 DataFrame의 칼럼 이름을 - No1, No2, No3, No4로 지정하시오.
print("\n문제 1 - b)a에서 생성한 DataFrame의 칼럼 이름을 - No1, No2, No3, No4로 지정하시오.")
q1_frame.columns = ['No1', 'No2', 'No3', 'No4']
print("frame: \n", q1_frame)

# 문제 1 - c)각 컬럼의 평균을 구하시오. mean() 함수와 axis 속성 사용
print("\n문제 1 - c)각 컬럼의 평균을 구하시오. mean() 함수와 axis 속성 사용")
print("각 컬럼별 평균:\n", q1_frame.mean(axis=0))

# 문제 2
#     numbers
# a | 10
# b | 20
# c | 30
# d | 40
# 문제 2 - a)DataFrame으로 위와 같은 자료를 만드시오. column(열) name은 numbers, row(행) name은 a~d이고 값은 10~40.
print("\n문제 2 - a)DataFrame으로 위와 같은 자료를 만드시오. column(열) name은 numbers, row(행) name은 a~d이고 값은 10~40.")
q2_frame = DataFrame([10, 20, 30, 40], columns=['numbers'], index=['a', 'b', 'c', 'd'])
print("q2_frame: \n", q2_frame)

# 문제 2 - b)c row의 값을 가져오시오.
print("\n문제 2 - b)c row의 값을 가져오시오.")
print("q2_frame.loc['c']", q2_frame.loc['c'])

# 문제 2 - c)a, d row들의 값을 가져오시오.
print("\n문제 2 - c)a, d row들의 값을 가져오시오.")
print("q2_frame.loc['a', 'd']", q2_frame.loc[['a', 'd']])

# 문제 2 - d)numbers의 합을 구하시오.
print("q2_frame['numbers'].sum():", q2_frame['numbers'].sum())

# 문제 2 - e)numbers의 값들을 각각 제곱하시오. 아래 결과가 나와야 함.
#     numbers
# a | 100
# b | 400
# c | 900
# d | 1600
print("문제 2 - e)numbers의 값들을 각각 제곱하시오. 아래 결과가 나와야 함.")
q2_frame = q2_frame.multiply(q2_frame)
print("q2_frame:\n", q2_frame)

# 문제 2 - f)float 라는 이름의 칼럼을 추가하시오. 값은 1.5, 2.5, 3.5, 4.5 아래 결과가 나와야 함.
#     numbers   float
# a | 100     |  1.5
# b | 400     |  2.5
# c | 900     |  3.5
# d | 1600    |  4.5
print("문제 2 - f)float 라는 이름의 칼럼을 추가하시오. 값은 1.5, 2.5, 3.5, 4.5 아래 결과가 나와야 함.")
temp_frame = DataFrame([1.5, 2.5, 3.5, 4.5], columns=['float'], index=['a', 'b', 'c', 'd'])
q2_frame['float'] = temp_frame['float']
print("q2_frame:\n", q2_frame)

# 문제 2 - g)names 라는 이름의 다음과 같은 칼럼을 위의 결과에 또 추가하시오. Series 클래스 사용
#       names
# d  |   길동
# a  |   오정
# b  |   팔계
# c  |   오공
print("문제 2 - g)names 라는 이름의 다음과 같은 칼럼을 위의 결과에 또 추가하시오. Series 클래스 사용")
temp_frame = DataFrame(['길동', '오정', '팔계', '오공'], columns=['names'], index=['a', 'b', 'c', 'd'])
q2_frame['names'] = temp_frame['names']
print("q2_frame:\n", q2_frame)

# 문제 3 - 1) 5 * 3 형태의 랜덤 정수형 DataFrame을 생성하시오.(범위: 1 이상 20 이하, 난수)
print("문제 3 - 1) 5 * 3 형태의 랜덤 정수형 DataFrame을 생성하시오.(범위: 1 이상 20 이하, 난수)")
q3_frame = pd.DataFrame(np.random.randint(1, 21, size=(5, 3)))
print("q3_frame:\n", q3_frame)
print(q3_frame.shape)

# 문제 3 - 2) 생성된 DataFrame의 칼럼 이름을 A, B, C로 설정하고, 행 인덱스를 r1, r2, r3, r4, r5로 설정하시오.
print("문제 3 - 2) 생성된 DataFrame의 칼럼 이름을 A, B, C로 설정하고, 행 인덱스를 r1, r2, r3, r4, r5로 설정하시오.")
q3_frame.columns = ['A', 'B', 'C']
q3_frame.index = pd.Index(['r1', 'r2', 'r3', 'r4', 'r5'])
print("q3_frame:\n", q3_frame)

# 문제 3 - 3) A 컬럼의 값이 10보다 큰 행만 출력하시오.
print("문제 3 - 3) A 컬럼의 값이 10보다 큰 행만 출력하시오.")
print(q3_frame[q3_frame['A'] > 10])

# 문제 3 - 4) 새로 D라는 컬럼을 추가하여, A와 B의 합을 저장하시오.
print("문제 3 - 4) 새로 D라는 컬럼을 추가하여, A와 B의 합을 저장하시오.")
q3_frame['D'] = q3_frame['A'] + q3_frame['B']
print("q3_frame:\n", q3_frame)

# 문제 3 - 5) 행 인덱스가 r3인 행을 제거하되, 원본 DataFrame이 실제로 바뀌도록 하시오.
print("문제 3 - 5) 행 인덱스가 r3인 행을 제거하되, 원본 DataFrame이 실제로 바뀌도록 하시오.")
q3_frame.drop('r3', inplace=True)
print("q3_frame:\n", q3_frame)

# 문제 3 - 6) 아래와 같은 정보를 가진 새로운 행(r6)을 DataFrame 끝에 추가하시오.
#       A       B       C       D
#r6     15      10      2       A+B
print("문제 3 - 6) 아래와 같은 정보를 가진 새로운 행(r6)을 DataFrame 끝에 추가하시오.")
q3_frame.loc['r6'] = [15, 10, 2, 15+10]
print("q3_frame:\n", q3_frame)

# 문제 4) 다음과 같은 재고 정보를 가지고 있는 딕셔너리 Data가 있다고 하자.
# data = {
#     'product': ['Mouse', 'Keyboard', 'Monitor', 'Laptop'],
#     'price':   [12000,     25000,      150000,    900000],
#     'stock':   [  10,         5,          2,          3 ]
# }
print("문제 4) 다음과 같은 재고 정보를 가지고 있는 딕셔너리 Data가 있다고 하자.")
data = {
    'product': ['Mouse', 'Keyboard', 'Monitor', 'Laptop'],
    'price':   [12000,     25000,      150000,    900000],
    'stock':   [  10,         5,          2,          3 ]
}
print("data:\n", data)

# 문제 4 - 1) 위 딕셔너리로부터 DataFrame을 생성하시오. 단, 행 인덱스는 p1, p2, p3, p4가 되도록 하시오.
print("문제 4 - 1) 위 딕셔너리로부터 DataFrame을 생성하시오. 단, 행 인덱스는 p1, p2, p3, p4가 되도록 하시오.")
q4_frame = DataFrame(data, index = ['p1', 'p2', 'p3', 'p4'])
print("q4_frame: \n", q4_frame)

# 문제 4 - 2) price와 stock을 이용하여 'total'이라는 새로운 컬럼을 추가하고, 값은 'price x stock'이 되도록 하시오.
print("문제 4 - 2) price와 stock을 이용하여 'total'이라는 새로운 컬럼을 추가하고, 값은 'price x stock'이 되도록 하시오.")
q4_frame['total'] = q4_frame['price'] * q4_frame['stock']
print("q4_frame: \n", q4_frame)

# 문제 4 - 3) 컬럼 이름을 다음과 같이 변경하시오. 원본 갱신
# - product → 상품명,  price → 가격,  stock → 재고,  total → 총가격
print("문제 4 - 3) 컬럼 이름을 다음과 같이 변경하시오. 원본 갱신")
q4_frame.rename(
        columns={
            'product': '상품명',
            'price': '가격',
            'stock': '재고',
            'total': '총가격'
    },
    inplace=True
)
print("q4_frame: \n", q4_frame)

# 문제 4 - 4) 재고(재고 칼럼)가 3 이하인 행의 정보를 추출하시오.
print("문제 4 - 4) 재고(재고 칼럼)가 3 이하인 행의 정보를 추출하시오.")
print(q4_frame[q4_frame['재고'] <= 3])

# 문제 4 - 5) 인덱스가 p2인 행을 추출하는 두 가지 방법(loc, iloc)을 코드로 작성하시오.
print("문제 4 - 5) 인덱스가 p2인 행을 추출하는 두 가지 방법(loc, iloc)을 코드로 작성하시오.")
print("q4_frame.loc['p2']", q4_frame.loc['p2'])
print("q4_frame.iloc[2]", q4_frame.iloc[1])

# 문제 4 - 6) 인덱스가 p3인 행을 삭제한 뒤, 그 결과를 확인하시오.(원본이 실제로 바뀌지 않도록, 즉 drop()의 기본 동작으로)
print("문제 4 - 6) 인덱스가 p3인 행을 삭제한 뒤, 그 결과를 확인하시오.(원본이 실제로 바뀌지 않도록, 즉 drop()의 기본 동작으로)")
print("q4_frame.drop('p3', inplace=False):", q4_frame.drop('p3', inplace=False))

# 문제 4 - 7) 위 DataFrame에 아래와 같은 행(p5)을 추가하시오.
#         상품명       가격       재고     총가격
#  p5     USB메모리    15000      10      가격*재고
print("문제 4 - 7) 위 DataFrame에 아래와 같은 행(p5)을 추가하시오.")
q4_frame.loc['p5'] = ['USB메모리', 15000, 10, 15000*10]
print("q4_frame: \n", q4_frame)