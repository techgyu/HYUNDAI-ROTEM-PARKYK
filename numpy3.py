
# numpy 배열 연산 예제
import numpy as np  # numpy 라이브러리 임포트

# 2x2 크기의 float16 타입 배열 x, y 생성
x = np.array([[1, 2], [3, 4]], dtype=np.float16)  # x: [[1, 2], [3, 4]]
y = np.array([[5, 6], [7, 8]], dtype=np.float16)  # y: [[5, 6], [7, 8]]


# x 배열 출력 및 데이터 타입 확인
print('x =')  # x =
print(x)      # [[1. 2.]
              #  [3. 4.]]
print('x dtype:', x.dtype)  # x의 데이터 타입 출력  # x dtype: float16
print()


# y 배열 출력 및 데이터 타입 확인
print('y =')  # y =
print(y)      # [[5. 6.]
              #  [7. 8.]]
print('y dtype:', y.dtype)  # y의 데이터 타입 출력  # y dtype: float16
print()


# x와 y의 요소별(같은 위치) 덧셈 연산
print('x + y =')  # x + y =
print(x + y)      # [[ 6.  8.]
                  #  [10. 12.]]
print()


# 5~8까지의 정수로 2x2 배열 z 생성 (reshape으로 2행 2열 변환)
z = np.arange(5, 9).reshape(2, 2)  # z: [[5, 6], [7, 8]]
print('z =')      # z =
print(z)          # [[5 6]
                  #  [7 8]]
print('z dtype:', z.dtype)  # z의 데이터 타입 출력  # z dtype: int32
print()

# x와 z의 요소별 덧셈 연산
print('x + z =')
print(x + z)
# numpy의 add 함수로도 동일하게 요소별 덧셈 가능
print('np.add(x, z) =')
print(np.add(x, z))

# numpy의 subtract, multiply, divide 함수로 요소별 연산
print('np.subtract(x, z) =')  # 요소별 뺄셈
print(np.subtract(x, z))
print('np.multiply(x, z) =')  # 요소별 곱셈
print(np.multiply(x, z))
print('np.divide(x, z) =')  # 요소별 나눗셈
print(np.divide(x, z))

# 대용량 배열 연산 속도 비교
import time
big_arr = np.random.rand(1000000)  # 0~1 사이 난수 100만 개 생성
print('big_arr: ', big_arr)

# 파이썬 내장 sum 함수로 합계 구하고 시간 측정
start = time.time()
sum(big_arr)
end = time.time()
print(f'sum(big_arr) time: {end - start:.6f} seconds')

# numpy의 sum 함수로 합계 구하고 시간 측정 (더 빠름)
start = time.time()
np.sum(big_arr)
end = time.time()
print(f'np.sum(big_arr) time: {end - start:.6f} seconds')

# 요소별 곱셈 연산 예시
print(x)
print(y)
print('x * y =')
print(x * y)  # 요소별 곱셈 (동일 위치 값끼리 곱함)
print('np.multiply(x, y) =')
print(np.multiply(x, y))  # 위와 동일

print()
# 행렬 곱셈(내적) 연산 예시
print('x / y =')
print(x.dot(y))  # 행렬 곱셈 (내적), 2차원 배열에서만 의미 있음
print('np.dot(x, y) =')
print(np.dot(x, y))  # 위와 동일

