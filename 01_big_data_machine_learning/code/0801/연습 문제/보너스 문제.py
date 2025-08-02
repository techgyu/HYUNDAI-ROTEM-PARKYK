import numpy as np
print('----------numpy 문제 추가 ~~~~~~~~~~~~~~~~~~~~~-----------')

print('Q1) 브로드캐스팅과 조건 연산')
# 다음 두 배열이 있을 때,
# a = np.array([[1], [2], [3]])
# b = np.array([10, 20, 30])
# 두 배열을 브로드캐스팅하여 곱한 결과를 출력하시오.
# 그 결과에서 값이 30 이상인 요소만 골라 출력하시오.
print("1. 배열 생성")
a = np.array([[1], [2], [3]])
b = np.array([10, 20, 30])
print("2. 두 배열을 브로드캐스팅하여 곱한 결과를 출력하시오.")
print(a * b)

print("3. 그 결과에서 값이 30 이상인 요소만 골라 출력하시오.")
print((a * b)[(a * b) > 30])

# --------------------------------------------------------------

print('Q2) 다차원 배열 슬라이싱 및 재배열')
#  - 3×4 크기의 배열을 만들고 (reshape 사용),  
#  - 2번째 행 전체 출력
#  - 1번째 열 전체 출력
#  - 배열을 (4, 3) 형태로 reshape
#  - reshape한 배열을 flatten() 함수를 사용하여 1차원 배열로 만들기
print("1. 3×4 크기의 배열을 만들고 (reshape 사용)")
arr = np.random.randn(12).reshape(3, 4)
print("arr:\n", arr)

print("2. 2번째 행 전체 출력:\n", arr[1])

print("3. 1번째 열 전체 출력:\n", arr[:, 0])

print("4. 배열을 (4, 3) 형태로 reshape:\n", arr.reshape(4, 3))

print("5. reshape한 배열을 flatten() 함수를 사용하여 1차원 배열로 만들기:\n", arr.flatten())

# --------------------------------------------------------------

print('Q3) 1부터 100까지의 수로 구성된 배열에서 3의 배수이면서 5의 배수가 아닌 값만 추출하시오.')
# 그런 값들을 모두 제곱한 배열을 만들고 출력하시오.
print("1. 1부터 100까지의 수로 구성된 배열을 생성")
arr = np.random.randint(1, 100, 100)
print("arr:\n", arr)
print(" 3의 배수이면서 5의 배수가 아닌 값만 추출")
new_arr = [] # 얘는 파이썬 기본 리스트
for i in range(len(arr)):
    if (arr[i] % 3) == 0 and (arr[i] % 5) != 0:
        new_arr.append(arr[i])
print("new_arr: \n", *new_arr) # arr은 넘파이 배열이므로, *를 붙여야 타입을 빼고 출력함
print('넘파이 extract 함수 사용:\n', np.extract((arr % 3 == 0) & (arr % 5 != 0), arr))

# --------------------------------------------------------------

print('Q4) 다음과 같은 배열이 있다고 할 때,')
# arr = np.array([15, 22, 8, 19, 31, 4])
# 값이 10 이상이면 'High', 그렇지 않으면 'Low'라는 문자열 배열로 변환하시오.
# 값이 20 이상인 요소만 -1로 바꾼 새로운 배열을 만들어 출력하시오. (원본은 유지)
# 힌트: np.where(), np.copy()

print("1. 배열을 생성")
arr = np.array([15, 22, 8, 19, 31, 4])
print("arr:\n", arr)

print("2. 값이 10 이상이면 'High', 그렇지 않으면 'Low'라는 문자열 배열로 변환")
print(np.where(arr >= 10, 'High', 'Low'))

print("3. 값이 20 이상인 요소만 -1로 바꾼 새로운 배열을 만들어 출력 (원본 유지)")
new_arr = np.where(arr > 20, -1, arr)
print("new_arr:\n", new_arr)

# --------------------------------------------------------------

print('Q5) 정규분포(평균 50, 표준편차 10)를 따르는 난수 1000개를 만들고, 상위 5% 값만 출력하세요.')
# 힌트 :  np.random.normal(), np.percentile()
print("1.  정규분포(평균 50, 표준편차 10)를 따르는 난수 1000개 생성")
arr = np.random.normal(50, 10, 1000)
print("arr:\n", arr)

print("2.  상위 5% 값만 출력")
print("상위 5% 기준 값", np.percentile(arr, 95))
print('넘파이 extract 함수 사용:\n', np.extract((arr > np.percentile(arr, 95)), arr))