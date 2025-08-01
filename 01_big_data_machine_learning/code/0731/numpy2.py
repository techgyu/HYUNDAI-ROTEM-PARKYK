# numpy 기본 기능
import numpy as np

ss = ['tom', 'james', 'oscar', 5]
print(ss, type(ss))
ss2 = np.array(ss)
print(ss2, type(ss2))

# 메모리 비교
li = list(range(1, 10))
print(li)
print(id(li[0])) # 리스트로 처리하면 각각의 요소가 별도의 메모리 주소를 가짐
print(id(li[1]))
print(id(li[2]))
print(id(li[3]))
print(li * 10)
print('^' * 10)

for i in li:
    print(i * 10, end=' ')

# 람다로 처리
list(map(lambda x: print(x * 10, end=' '), li))

print('---')

num_arr = np.array(li)
print(id(num_arr[0]), ' ', id(num_arr[1])) # 넘파이로 처리하면 메모리 절약(주소 동일)
print(num_arr * 10)

print()
a = np.array([1, 2, 0, '3'])
print(a, type(a), a.dtype, a.shape, a.ndim, a.size)
print(a[0], a[1])
b = np.array([[1, 2, 3], [4, 5, 6]])
print(b.shape, ' ', b[0], ' ', b[[0]])

print(b[0, 0], ' ', b[1, 2])

print()
c = np.zeros((2, 2)) # 2x2 행렬의 모든 요소를 0으로 초기화
print(c)

print()
d = np.ones((2, 2)) # 2x2 행렬의 모든 요소를 1로 초기화
print(d)

print()
e = np.full((2, 2), 7) # 2x2 행렬의 모든 요소를 7로 초기화
print(e)

print()
f = np.eye(3, 3) # 3x3 단위 행렬 생성
print(f)

print()
print(np.random.rand(5)) # 0~1 사이의 난수로 5x1 행렬 생성(균등 분포)
print()
print(np.random.randn(5)) # 평균 0, 표준편차 1인 난수로 5x1 행렬 생성(정규 분포)

# 분포 확인용 시각화 코드 추가
import matplotlib.pyplot as plt

# 균등 분포 확인
# rand_data = np.random.rand(10000)
# plt.hist(rand_data, bins=100, color='skyblue', edgecolor='black')
# plt.title('Uniform Distribution: np.random.rand')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.show()

# 정규 분포 확인
# randn_data = np.random.randn(10000)
# plt.hist(randn_data, bins=100, color='salmon', edgecolor='black')
# plt.title('Normal Distribution: np.random.randn')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.show()

np.random.seed(42) # 난수 생성 시드 설정, 동일한 난수 생성 보장
print(np.random.randn(2, 3))

print('\n배열 인덱싱 ------')
a = np.array([1, 2, 3, 4, 5])
print(a)
print(a[1])  # 인덱싱, 첫 번째 요소
print(a[1:]) # 슬라이싱, 두 번째 요소부터 끝까지
print(a[1:5]) # 슬라이싱, 두 번째부터 다섯 번째 요소까지
print(a[1:5:2]) # 슬라이싱, 두 번째부터 다섯 번째 요소까지 2칸씩 건너뛰기
print(a[-2:]) # 마지막에서 두 번째 요소

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]) # 다차원 배열 생성
print(a)
print(a[:])
print(a[1:]) # 두 번째 행부터 끝까지
print(a[1:, 0:2]) # 두 번째 행부터 끝까지, 첫 번째 열부터 두 번째 열까지
print(a[0, 0], ' ', a[0][0], ' ', a[[0]])

print()
aa = np.array((1, 2, 3))
print(aa)
bb = aa[1:3] # 슬라이싱, 두 번째부터 세 번째 요소(논리적 메모리 확보)
print(bb, ' ', bb[0], ' ', bb[1]) # 슬라이싱, 두 번째부터 세 번째 요소까지
bb[0] = 33
print(bb)
print(aa)
cc = aa[1:3].copy() # 슬라이싱 후 copy()로 새로운 배열 생성
print(cc)
cc[0] = 55
print(cc)  # [55  3]
print(aa)  # [ 1 33  3]

print('------------')
# 3x3 2차원 배열 생성
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 행 인덱싱/슬라이싱 예시
r1 = a[1, :]    # 두 번째 행만 1차원 배열로 추출 (shape: (3,))
r2 = a[1:2, :]  # 두 번째 행만 2차원 배열로 추출 (shape: (1, 3))
print(r1, r1.shape)  # [4 5 6] (3,)
print(r2, r2.shape)  # [[4 5 6]] (1, 3)

# 열 인덱싱/슬라이싱 예시
c1 = a[:, 1]   # 전체 행의 1열만 1차원 배열로 추출 (shape: (3,))
c2 = a[:, 1:2] # 전체 행의 1열만 2차원 배열로 추출 (shape: (3, 1))

print(c1, c1.shape)  # [2 5 8] (3,)
print(c2, c2.shape)  # [[2]
                    #  [5]
                    #  [8]] (3, 1)

print()
# 전체 배열 출력
print(a)  # [[1 2 3]
          #  [4 5 6]
          #  [7 8 9]]

# 불리언 인덱싱: 조건에 맞는 요소만 True로 표시
bool_idx = a >= 5
print(bool_idx)  # [[False False False]
                 #  [False  True  True]
                 #  [ True  True  True]]
# 불리언 인덱싱으로 조건에 맞는 값만 추출
print(a[bool_idx])  # [5 6 7 8 9]

