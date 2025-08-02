# 내적 연산
import numpy as np
v = np.array([9, 10])
w = np.array([11, 12])
x = np.array([[1, 2], 
              [3, 4]])
y = np.array([[5, 6], 
              [7, 8]])

print(v * w) # 원소별 곱셈 [9*11, 10*12] = [99, 120]

# 파이썬 내적 연산: 속도 느림
print(v.dot(w)) # 내적 연산 9*11 + 10*12 = 99 + 120 = 219

# 넘파이 내적 연산: 속도 빠름(c기반)
print(np.dot(w, v)) # 내적 연산: 11*9 + 12*10 = 99 + 120 = 219

# 넘파이 내적 연산: 속도 빠름(c기반), 서로 다른 크기의 배열일 때
# 머신 러닝할 때 구조를 맞춰주는 과정에서 자주 사용
# (중요) 행렬 곱에서는 앞 행렬(x)의 열 개수와 뒤 벡터(v)의 원소 개수가 같아야 곱셈이 가능함
#   예) x가 (2, 2)면 v는 (2,)여야 함. (2, 3) * (3,) 가능, (2, 2) * (3,) 불가능
print(np.dot(x, v)) # 내적 연산: [1,2]*[9,10] + [3,4]*[9,10] = [1*9+2*10, 3*9+4*10] = [29, 67]

# 넘파이 내적 연산: 행렬 곱
print(np.dot(x, y)) # 내적 연산: [[1, 2]*[5, 6] + [3, 4]*[7, 8]] = [[19, 22], [43, 50]]

print('유용한 함수 정리')
print(x) # 배열 x 출력
print('np.sum(x):', np.sum(x))  # 배열 x의 모든 원소 합계
print('np.sum(x, axis=0):', np.sum(x, axis=0))  # 열 방향 합계
print('np.sum(x, axis=1):', np.sum(x, axis=1))  # 행 방향 합계

print('np.mean(x):', np.mean(x))  # 배열 x의 평균
print('np.max(x):', np.max(x))  # 배열 x의 최대값
print('np.argmax(x) index:', np.argmax(x))  # 배열 x의 최대값 위치 index 반환
print('np.min(x):', np.min(x))  # 배열 x의 최소값
print('np.argmin(x) index:', np.argmin(x))  # 배열 x의 최소값 위치 index 반환

print('np.cumsum(x) 누적 합:', np.cumsum(x))  # 배열 x의 누적 합계
print('np.cumprod(x) 누적 곱:', np.cumprod(x))  # 배열 x의 누적 곱셈

names1 = np.array(['tom', 'james', 'oscar', 'johnson', 'harry', 'tom', 'oscar']) # 중복된 이름 포함
names2 = np.array(['tom', 'page', 'john']) # name1과 name2의 교집합은 tom

print('name1의 고유 원소 집합:', np.unique(names1))  # 중복 제거 후 고유한 원소 반환
print('name2의 고유 원소 집합:', np.unique(names2))  # 중복 제거 후 고유한 원소 반환

print('names1과 name2의 교집합:', np.intersect1d(names1, names2))  # 교집합, 사전 순으로 정렬됨
print('names1과 name2의 교집합 assum_unique=True:', np.intersect1d(names1, names2, assume_unique=True))  # 교집합, 사전 순으로 정렬됨
print('names1과 name2의 합집합:', np.union1d(names1, names2))  # 합집합, 사전 순으로 정렬됨
print('names1과 name2의 차집합:', np.setdiff1d(names1, names2))  # 차집합, 사전 순으로 정렬됨


print('transpose 연산')
# 전치 연산, 행과 열을 바꿈
print('x의 전치 행렬:\n', x.T)  # 원래 x의 행렬, [[1, 2], [3, 4]] -> x의 전치 행렬, [[1, 3], [2, 4]]
arr = np.arange(1, 16).reshape(3, 5) # 1부터 15까지의 숫자를 3행 5열로 변형
print('arr:\n', arr)  # arr 출력
print('arr의 전치 행렬:\n', arr.T)  # arr의 전치 행렬

# 내적
# print(np.dot(arr, arr))  # arr과 arr의 내적 연산(행렬 개수가 안 맞아서 불가능))
print('arr과 arr의 전치 행렬의 내적 연산:\n', np.dot(arr, arr.T))  # arr과 arr의 전치 행렬의 내적 연산(행렬 개수가 맞아서 가능)

print('arr.flatten() 2차원 배열을 1차원으로 평탄화:', arr.flatten())  # 2차원 배열을 1차원으로 평탄화
print('arr.ravel(): 2차원 배열을 1차원으로 평탄화:', arr.ravel())  # 2차원 배열을 1차원으로 평탄화, ravel은 원본 배열을 변경하지 않음
