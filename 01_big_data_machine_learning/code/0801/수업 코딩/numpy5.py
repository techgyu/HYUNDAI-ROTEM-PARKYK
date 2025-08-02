import numpy as np

aa = np.eye(3) # 3x3 단위 행렬 생성
print('aa: \n', aa)
bb = np.c_[aa, aa[2]] # aa를 열 방향으로 연결
print('bb: \n', bb)
cc = np.r_[aa, [aa[2]]] # aa를 행 방향으로 연결
print('cc: \n', cc)

# reshape
a = np.array([1, 2, 3])
print('np.c_:\n', np.c_[a]) # a를 열 방향으로 연결
a = a.reshape(3, 1) # a를 3행 1열로 변형
print('a: \n', a)

print('--append, insert, delete--')
# 1차원
print(a)
b = np.append(a, [4, 5]) # a에 [4, 5]를 추가
print(b)
c = np.insert(a, 0, [6, 7]) # a의 0번째 위치에 [6, 7]를 삽입
print('c: \n', c)

d = np.delete(a, 1) # a의 1번째 원소를 삭제
print('d: \n', d)

# 2차원
aa = np.arange(1, 10).reshape(3, 3) # 1부터 9까지의 숫자를 3행 3열로 변형
print('aa: \n', aa)
print(np.insert(aa, 1, 99)) # aa의 1번째 행에 99를 삽입, 삽입 후 차원 축소
print(np.insert(aa, 1, 99, axis=0)) # aa의 1번째 열에 99를 삽입, 차원 유지
print(np.insert(aa, 1, 99, axis=1)) # aa의 마지막 행에 99를 추가, 차원 유지

print(aa)
bb = np.arange(1, 10).reshape(3, 3) # 1부터 9까지의 숫자를 3행 3열로 변형
print('bb: \n', bb)
cc = np.append(aa, bb) # aa와 bb를 연결, 차원 축소
print('cc: \n', cc) # aa와 bb를 연결, 차원 축소
cc = np.append(aa, bb, axis=0) # aa와 bb를 행 방향으로 연결, 차원 유지
print('cc: \n', cc) # aa와 bb를 행 방향으로 연결, 차원 유지

print("np.append 연습")
print(np.append(aa, 88)) # aa에 88을 추가, 차원 축소
print(np.append(aa, [[88, 88, 88]], axis = 0)) # aa의 마지막 행에 [88, 88, 88]을 추가, 차원 유지
print(np.append(aa, [[88], [88], [88]], axis = 1)) # aa의 마지막 열에 [88], [88], [88]을 추가, 차원 유지

print("np.delete 연습")
print(np.delete(aa, 1)) # aa의 1번째 원소를 삭제, 차원 축소
print(np.delete(aa, 1, axis=0)) # aa의 1번째 행을 삭제, 차원 유지
print(np.delete(aa, 1, axis=1)) # aa의 1번째 열을 삭제, 차원 유지

# 조건 연산 where(조건, 참일 때 값, 거짓일 때 값)
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
print('x:', x)
print('y:', y)
condData = np.array([True, False, True]) # 조건 배열
print('condData:\n', condData)
result = np.where(condData, x, y) # 조건에 따라 x 또는 y의 값을 선택
print('result:\n', result) # 조건에 따라 x 또는 y의 값을 선택

aa = np.where(x >= 2)
print(aa)
print(x[aa]) # 조건에 맞는 x의 원소 출력
print(np.where(x >= 2, 'T', 'F')) # 조건에 따라 'T' 또는 'F' 출력
print(np.where(x >= 2, x, x + 100)) # 조건에 따라 x 또는 x + 100 출력

bb = np.random.randn(4, 4) # 4x4 랜덤 배열 생성, 정규분포(가우시안분포)를 따르는 난수
# 중심 극한 정리란? : 여러 확률분포에서 임의로 뽑은 표본들의 평균을 많이 모으면, 그 분포가 원래 분포가 무엇이든 간에 평균이 정규분포를 따르게 된다는 통계학의 중요한 이론
print('bb:\n', bb)

print('처리 결과: \n', np.where(bb >= 0, 7, bb)) # bb의 원소가 0 이상이면 7, 아니면 bb의 원소 그대로

print('배열 결합 / 분할')
kbs = np.concatenate([x, y]) # x와 y를 연결
print('kbs:\n', kbs) # x와 y를 연결한 결과 출력

x1, x2 = np.split(kbs, 2) # kbs를 두 개의 배열로 분할
print('x1:\n', x1) # kbs의 첫 번째 부분 출력
print('x2:\n', x2) # kbs의 두 번째 부분 출력

a = np.arange(1, 17).reshape(4, 4) # 1부터 15까지의 숫자를 4행 4열로 변형
print('a:\n', a) # a 출력
x1, x2 = np.hsplit(a, 2) # a를 두 개의 배열로 수평 분할
print('x1:\n', x1) # a의 첫 번째 부분 출력
print('x2:\n', x2) # a의 두 번째 부분 출력

x1, x2 = np.vsplit(a, 2) # a를 두 개의 배열로 수직 분할
print('x1:\n', x1) # a의 첫 번째 부분 출력
print('x2:\n', x2) # a의 두 번째 부분 출력

print('복원, 비복원 추출') # 무작위로 원소를 선택하는 방법

datas = np.array([1, 2, 3, 4, 5, 6, 7]) # 데이터 배열 생성

# 복원 추출: 선택한 원소를 다시 선택할 수 있는 경우
print('복원 추출: ')
for _ in range(5):
    print(datas[np.random.randint(0, len(datas) - 1)], end=' ') # 무작위로 원소를 선택하여 출력
import random
# 비복원 추출 전용 함수: sample()
print('비복원 추출 sample 함수 사용: ')
print(*random.sample(list(datas), 5))  # datas에서 5개를 비복원 추출(숫자만 공백 구분으로 출력)

print(random.sample(list(datas), 5))

# 결과: [1, 5, 7, 6, 4] (리스트 전체가 출력, 대괄호 있음)
# 리스트의 자료형이 np.int64면 [np.int64(1), np.int64(5), ...]처럼 보일 수 있음
# print(*random.sample(list(datas), 5))

# 결과: 1 5 7 6 4 (숫자만 공백 구분, 대괄호 없음)
# *은 리스트의 각 원소를 print의 개별 인자로 전달(언패킹)
# np.int64여도 print가 알아서 숫자만 출력
# 즉,

# *이 없으면: 리스트 전체가 한 번에 출력 → 대괄호, 자료형까지 보일 수 있음
# *이 있으면: 리스트의 각 원소가 따로 출력 → 숫자만 공백 구분, 깔끔하게 나옴

# 복원 추출 지원 함수: choice()
print('복원 추출 choice 함수 사용: ')
print(np.random.choice(range(1, 45), 6)) # 무작위로 6개의 원소를 선택하여 출력

# 비복원 추출: 선택한 원소를 다시 선택할 수 없는 경우
print('\n비복원 추출 choice 함수 사용: ')
print(np.random.choice(range(1, 45), 6, replace=False)) # 무작위로 6개의 원소를 선택하여 출력

# 가중치를 부여한 랜덤 추출
print('가중치를 부여한 랜덤 추출: ')
ar = 'air book cat d e f god'
ar = ar.split(' ') # 문자열을 공백으로 분할하여 리스트로 변환
print(ar)
print(np.random.choice(ar, 3, p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4]))