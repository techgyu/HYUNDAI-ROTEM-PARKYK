import numpy as np
print('-----------1) step1 : array 관련 문제-----------')
print('\nstep1: 정규분포를 따르는 난수를 이용하여 5행 4열 구조의 다차원 배열 객체를 생성하고, 각 행 단위로 합계, 최댓값을 구하시오.')

arr = np.random.randn(5, 4)

print('arr: \n', arr)
print('1행만 선택: \n', arr[0])
print('1행 합계: \n', np.sum(arr[0]))
print('1행 최댓값: \n', np.max(arr[0]))
print('2행 합계: \n',  np.sum(arr[1]))
print('2행 최댓값: \n',  np.max(arr[1]))
print('3행 합계: \n',  np.sum(arr[2]))
print('3행 최댓값: \n', np.max(arr[2]))