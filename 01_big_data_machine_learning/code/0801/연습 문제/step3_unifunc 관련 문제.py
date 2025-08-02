import numpy as np
print('----------3) step3 : unifunc 관련문제-----------')
#   표준정규분포를 따르는 난수를 이용하여 4행 5열 구조의 다차원 배열을 생성한 후
#   아래와 같이 넘파이 내장함수(유니버설 함수)를 이용하여 기술통계량을 구하시오.
#   배열 요소의 누적합을 출력하시오.

# <<출력 예시>>
# ~ 4행 5열 다차원 배열 ~
# [[ 0.56886895  2.27871787 -0.20665035 -1.67593523 -0.54286047]
#            ...
#  [ 0.05807754  0.63466469 -0.90317403  0.11848534  1.26334224]]

# ~ 출력 결과 ~
# 평균 :
# 합계 :
# 표준편차 :
# 분산 :
# 최댓값 :
# 최솟값 :

# 1사분위 수 :           percentile()
# 2사분위 수 :
# 3사분위 수 :
# 요소값 누적합 :      cumsum()

print("1. 표준정규분포를 따르는 난수를 이용하여 4행 5열 구조의 다차원 배열을 생성")
arr = np.random.randn(4, 5)
print("arr:\n", arr)

print("2. 아래와 같이 넘파이 내장함수(유니버설 함수)를 이용하여 기술통계량을 구하시오.")
print("2.1 평균:")
print(np.mean(arr))

print("2.2 합계:")
print(np.sum(arr))

print("2.3 표준편차:")
print(np.std(arr))

print("2.4 분산:")
print(np.var(arr))

print("2.5 최댓값:")
print(np.max(arr))

print("2.6 최솟값:")
print(np.min(arr))

print("2.7 1사분위 수:")
print(np.percentile(arr, 25))

print("2.8 2사분위 수:")
print(np.percentile(arr, 50))

print("2.9 3사분위 수:")
print(np.percentile(arr, 75))

print("2.10 요소값 누적합 :")
print(np.cumsum(arr))