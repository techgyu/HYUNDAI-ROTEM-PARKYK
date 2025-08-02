
# ===============================
# numpy 실습 종합 예제 (번호/주석 정리)
# ===============================
import numpy as np  # 항상 맨 위에 위치
import random       # random도 위에 위치

# 1. 내적 연산 및 행렬 곱
print('===== 1. 내적 연산 및 행렬 곱 =====')
v = np.array([9, 10])
w = np.array([11, 12])
x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])
print('1-1) 원소별 곱셈:', v * w) # [99, 120]
print('1-2) 파이썬 내적 연산:', v.dot(w)) # 219
print('1-3) 넘파이 내적 연산:', np.dot(w, v)) # 219
print('1-4) 행렬*벡터 내적:', np.dot(x, v)) # [29, 67]
print('1-5) 행렬*행렬 내적:', np.dot(x, y)) # [[19, 22], [43, 50]]

# 2. 유용한 함수 정리
print('\n===== 2. 유용한 함수 정리 =====')
print('x =\n', x)
print('2-1) 전체 합계:', np.sum(x))
print('2-2) 열 방향 합계:', np.sum(x, axis=0))
print('2-3) 행 방향 합계:', np.sum(x, axis=1))
print('2-4) 평균:', np.mean(x))
print('2-5) 최대값:', np.max(x))
print('2-6) 최대값 위치:', np.argmax(x))
print('2-7) 최소값:', np.min(x))
print('2-8) 최소값 위치:', np.argmin(x))
print('2-9) 누적 합:', np.cumsum(x))
print('2-10) 누적 곱:', np.cumprod(x))

# 3. 집합 연산
print('\n===== 3. 집합 연산 =====')
names1 = np.array(['tom', 'james', 'oscar', 'johnson', 'harry', 'tom', 'oscar'])
names2 = np.array(['tom', 'page', 'john'])
print('3-1) name1 고유값:', np.unique(names1))
print('3-2) name2 고유값:', np.unique(names2))
print('3-3) 교집합:', np.intersect1d(names1, names2))
print('3-4) 교집합(assume_unique=True):', np.intersect1d(names1, names2, assume_unique=True))
print('3-5) 합집합:', np.union1d(names1, names2))
print('3-6) 차집합:', np.setdiff1d(names1, names2))

# 4. 전치 연산 및 평탄화
print('\n===== 4. 전치 연산 및 평탄화 =====')
print('4-1) x의 전치:\n', x.T)
arr = np.arange(1, 16).reshape(3, 5)
print('4-2) arr:\n', arr)
print('4-3) arr의 전치:\n', arr.T)
print('4-4) arr.flatten():', arr.flatten())
print('4-5) arr.ravel():', arr.ravel())
print('4-6) arr과 arr.T 내적:\n', np.dot(arr, arr.T))

# 5. 브로드캐스팅
print('\n===== 5. 브로드캐스팅 =====')
x = np.arange(1, 10).reshape(3, 3)
y = np.array([1, 0, 1])
print('5-1) x:\n', x)
print('5-2) y:', y)
z = np.empty_like(x)
for i in range(3):
    z[i] = x[i] + y
print('5-3) x + y (for문):\n', z)
kbs = np.tile(y, (3, 1))
print('5-4) tile로 확장한 y:\n', kbs)
print('5-5) x + kbs:', x + kbs)
print('5-6) 브로드캐스팅 x + y:', x + y)
a = np.array([0, 1, 2])
b = np.array([5, 5, 5])
print('5-7) a + b:', a + b)
print('5-8) a + 5:', a + 5)

# 6. 파일 입출력
print('\n===== 6. 넘파이 파일 입출력 =====')
np.save('numpy4etc.npy', x)
np.savetxt('numpy4etc.txt', x)
temp = np.load('numpy4etc.npy')
print('6-1) npy 파일 불러오기:\n', temp)
mydatas = np.loadtxt('numpy4etc.txt', delimiter=' ')
print('6-2) txt 파일 불러오기:\n', mydatas)

# 7. 배열 결합/분할, append/insert/delete
print('\n===== 7. 배열 결합/분할, append/insert/delete =====')
aa = np.eye(3)
print('7-1) 단위행렬 aa:\n', aa)
bb = np.c_[aa, aa[2]]
print('7-2) 열 방향 결합 bb:\n', bb)
cc = np.r_[aa, [aa[2]]]
print('7-3) 행 방향 결합 cc:\n', cc)
a = np.array([1, 2, 3])
print('7-4) np.c_ a:', np.c_[a])
a = a.reshape(3, 1)
print('7-5) a reshape:', a)
print('7-6) append:', np.append(a, [4, 5]))
print('7-7) insert:', np.insert(a, 0, [6, 7]))
print('7-8) delete:', np.delete(a, 1))
aa = np.arange(1, 10).reshape(3, 3)
print('7-9) 2차원 aa:\n', aa)
print('7-10) insert(행):', np.insert(aa, 1, 99))
print('7-11) insert(axis=0):', np.insert(aa, 1, 99, axis=0))
print('7-12) insert(axis=1):', np.insert(aa, 1, 99, axis=1))
bb = np.arange(1, 10).reshape(3, 3)
cc = np.append(aa, bb)
print('7-13) append(차원축소):', cc)
cc = np.append(aa, bb, axis=0)
print('7-14) append(axis=0):\n', cc)
print('7-15) append(스칼라):', np.append(aa, 88))
print('7-16) append(행):', np.append(aa, [[88, 88, 88]], axis=0))
print('7-17) append(열):', np.append(aa, [[88], [88], [88]], axis=1))
print('7-18) delete(차원축소):', np.delete(aa, 1))
print('7-19) delete(axis=0):', np.delete(aa, 1, axis=0))
print('7-20) delete(axis=1):', np.delete(aa, 1, axis=1))

# 8. 조건 연산 where
print('\n===== 8. 조건 연산 where =====')
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
condData = np.array([True, False, True])
print('8-1) 조건 배열:', condData)
print('8-2) where(cond, x, y):', np.where(condData, x, y))
print('8-3) x >= 2 index:', np.where(x >= 2))
print('8-4) x[x>=2]:', x[np.where(x >= 2)])
print('8-5) where(x>=2, "T", "F"):', np.where(x >= 2, 'T', 'F'))
print('8-6) where(x>=2, x, x+100):', np.where(x >= 2, x, x + 100))
bb = np.random.randn(4, 4)
print('8-7) bb:\n', bb)
print('8-8) bb >= 0이면 7, 아니면 원래값:\n', np.where(bb >= 0, 7, bb))

# 9. 배열 결합/분할 split, hsplit, vsplit
print('\n===== 9. 배열 결합/분할 split, hsplit, vsplit =====')
kbs = np.concatenate([x, y])
print('9-1) concatenate:', kbs)
x1, x2 = np.split(kbs, 2)
print('9-2) split:', x1, x2)
a = np.arange(1, 17).reshape(4, 4)
print('9-3) a:\n', a)
x1, x2 = np.hsplit(a, 2)
print('9-4) hsplit:', x1, x2)
x1, x2 = np.vsplit(a, 2)
print('9-5) vsplit:', x1, x2)

names1 = np.array(['tom', 'james', 'oscar', 'johnson', 'harry', 'tom', 'oscar']) # 중복된 이름 포함
aa = np.eye(3) # 3x3 단위 행렬 생성
aa = np.where(x >= 2)
bb = np.random.randn(4, 4) # 4x4 랜덤 배열 생성, 정규분포(가우시안분포)를 따르는 난수
x1, x2 = np.split(kbs, 2) # kbs를 두 개의 배열로 분할
a = np.arange(1, 17).reshape(4, 4) # 1부터 15까지의 숫자를 4행 4열로 변형
x1, x2 = np.vsplit(a, 2) # a를 두 개의 배열로 수직 분할
datas = np.array([1, 2, 3, 4, 5, 6, 7]) # 데이터 배열 생성
