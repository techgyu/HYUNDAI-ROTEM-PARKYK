# 브로드캐스팅에 대해 다룸
# 브로드캐스티이란? 크기가 다른 배열 간의 연산 시 배열의 구조 자동 변환
# 작은 배열과 큰 배열 연산 시 작은 배열은 큰 배열의 크기에 맞춰 자동으로 확장됨
import numpy as np
# 브로드캐스팅 제약 조건
# 1. 작은 배열의 차원이 큰 배열보다 작아야 함

x = np.arange(1, 10).reshape(3, 3)
y = np.array([1, 0, 1])
print('x:\n', x)  # x 출력
print('y:\n', y)  # y 출력

# 두 배열의 요소 더하기
# 1) 새로운 배열을 이용(브로드 캐스트 미 사용)
z = np.empty_like(x) # x와 같은 크기의 빈 배열 생성
print('z:\n', z)  # z 출력, 안에 들어간 데이터는 쓰레기 값
for i in range(3):
    z[i] = x[i] + y # x의 각 행에 y를 더함
print('x + y = z:\n', z)  # z 출력

# 2) tile을 이용
kbs = np.tile(y, (3, 1)) # tile 메소드를 이용하여 y를 3행으로 반복하여 kbs 생성
print('y:\n', y)  # y 출력
print('kbs:\n', kbs)  # kbs 출력
z = x + kbs  # x와 kbs를 더함
print('z:\n', z)  # z 출력

# 3) 브로드캐스팅을 이용
print('x:\n', x)  # x 출력
print('y:\n', y)  # y 출력
kbs = x + y
print('z:\n', kbs) # 브로드캐스팅을 이용하여 x와 y를 더함

a = np.array([0, 1, 2])
b = np.array([5, 5, 5])
print(a + b) # a와 b를 더함, 브로드캐스팅 적용됨
print(a + 5) # a에 5를 더함, 브로드캐스팅 적용됨

print('\n 넘파이로 파일 i/o')
np.save('numpy4etc.npy', x) # x 배열을 numpy4etc.npy 파일로 저장, binary 형태로 저장
np.savetxt('numpy4etc.txt', x) # x 배열을 numpy4etc.txt 파일로 저장, 텍스트 형태로 저장
temp = np.load('numpy4etc.npy') # numpy4etc.npy 파일에서 x 배열을 불러옴
print('temp:\n', temp)  # 불러온 배열 출력

mydatas = np.loadtxt('numpy4etc.txt', delimiter=' ')  # numpy4etc.txt 파일에서 데이터를 불러옴
print('mydatas:\n', mydatas)  # 불러온 데이터 출력