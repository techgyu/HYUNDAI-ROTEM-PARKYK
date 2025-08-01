import numpy as np
print('-----------2) step2 : indexing 관련문제-----------')
print('\n 문2-1) 6행 6열의 다차원 zero 행렬 객체를 생성한 후 다음과 같이 indexing 하시오.')
#    조건1> 36개의 셀에 1~36까지 정수 채우기

#    조건2> 2번째 행 전체 원소 출력하기 

#               출력 결과 : [ 7.   8.   9.  10.  11.  12.]

#    조건3> 5번째 열 전체 원소 출력하기

#               출력결과 : [ 5. 11. 17. 23. 29. 35.]

#    조건4> 15~29 까지 아래 처럼 출력하기

#               출력결과 : 

#               [[15.  16.  17.]

#               [21.  22.  23]

#               [27.  28.  29.]]

arr = np.zeros((6, 6)) # 6행 6열의 다차원 zero 행렬 객체를 생성
print('6행 6열의 다차원 zero 행렬 객체를 생성: \n', arr)
arr.flat[:] = np.arange(1, 37) # 조건1> 36개의 셀에 1~36까지 정수 채우기
print('36개의 셀에 1~36까지 정수 채우기: \n', arr) 
print('2번째 행 전체 원소 출력하기', arr[1]) # 조건2> 2번째 행 전체 원소 출력하기
print('5번째 열 전체 원소 출력하기', arr[:, 4]) # 조건3> 5번째 열 전체 원소 출력하기
# 조건4> 15~29 까지 아래 처럼 출력하기
print('15~29 까지 아래 처럼 출력하기:\n', np.extract((arr >= 15) & (arr <= 29), arr).reshape(3, 5))

#-------------------------------------------------------------------------------------------------------

print('문2-2) 6행 4열의 다차원 zero 행렬 객체를 생성한 후 아래와 같이 처리하시오.')
#조건1> 20~100 사이의 난수 정수를 6개 발생시켜 각 행의 시작열에 난수 정수를 저장하고, 
# 두 번째 열부터는 1씩 증가시켜 원소 저장하기
#조건2> 첫 번째 행에 1000, 마지막 행에 6000으로 요소값 수정하기

# <<출력 예시>>

# 1. zero 다차원 배열 객체
#   [[ 0.  0.  0.  0.]
#         ...
#    [ 0.  0.  0.  0.]]

# 2. 난수 정수 발생
# random.randint(s, e, n)

# 3. zero 다차원 배열에 난수 정수 초기화 결과. 두 번째 열부터는 1씩 증가시켜 원소 저장하기
# [[  90.   91.   92.   93.]
#  [  40.   41.   42.   43.]
#  [ 100.  101.  102.  103.]
#  [  22.   23.   24.   25.]
#  [  52.   53.   54.   55.]
#  [  71.   72.   73.   74.]]

# 4. 첫 번째 행에 1000, 마지막 행에 6000으로 수정
#  [[ 1000.  1000.  1000.  1000.]
#   [   40.    41.    42.    43.]
#   [  100.   101.   102.   103.]
#   [   22.    23.    24.    25.]
#   [   52.    53.    54.    55.]
#   [ 6000.  6000.  6000.  6000.]]

# 1. zero 다차원 배열 객체
arr = np.zeros((6, 4)) # zero 다차원 배열 객체
print('1. zero 다차원 배열 객체: \n', arr)

# #조건1> 20~100 사이의 난수 정수를 6개 발생시켜
# 각 행의 시작열에 난수 정수를 저장하고, 
# 두 번째 열부터는 1씩 증가시켜 원소 저장하기
print("2.1 20~100 사이의 난수 정수를 6개 발생:")
rand_six_num = np.random.randint(20, 100, 6)
print('rand_six_num:', rand_six_num)

print("2.2 각 행의 시작열에 난수 정수를 저장")
for i in range(6):
    arr[i][0] = rand_six_num[i]
print('arr: \n', arr)

print("2.3 두 번째 열부터는 1씩 증가시켜 원소 저장하기")
z = 1
for i in range(6):
    for t in range(1, 4):
        rand_six_num[i] += 1
        arr[i][t] = rand_six_num[i]
print('arr: \n', arr)

#조건2> 첫 번째 행에 1000, 마지막 행에 6000으로 요소값 수정하기
print("3,1 첫 번째 행에 1000")
for i in range(4):
    arr[0][i] = 1000
print('arr: \n', arr) 

print("3.2 마지막 행에 6000으로 요소값 수정하기")
for i in range(4):
    arr[5][i] = 6000
print('arr: \n', arr) 