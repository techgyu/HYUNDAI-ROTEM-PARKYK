import tensorflow as tf  # 텐서플로우 라이브러리 임포트
import numpy as np  # 넘파이 라이브러리 임포트

# 상수 노드 생성: 값이 변하지 않는 텐서
node1 = tf.constant(3, dtype=tf.float32)  # 값 3, float32 타입
node2 = tf.constant(4.0)  # 값 4.0, 기본 float32 타입
print(node1)  # node1의 텐서 정보 출력
print(node2)  # node2의 텐서 정보 출력

# 두 상수 노드의 합 계산
imsi = tf.add(node1, node2)  # node1 + node2 결과 텐서
print(imsi)  # 합산 결과 텐서 정보 출력

print()  # 구분을 위한 빈 줄 출력

# 변수 노드 생성: 값이 변경 가능한 텐서
node3 = tf.Variable(3, dtype=tf.float32)  # 값 3, float32 타입
node4 = tf.Variable(4.0)  # 값 4.0, 기본 float32 타입
tf.print(node3)  # node3의 값 출력 (tf.print는 실제 값 출력)
tf.print(node4)  # node4의 값 출력

# 두 변수 노드의 합 계산
imsi2 = tf.add(node3, node4)  # node3 + node4 결과 텐서
print(imsi2)  # 합산 결과 텐서 정보 출력

# node4에 node3 값을 더해서 node4를 갱신
node4.assign_add(node3)  # node4 = node4 + node3
print(node4)  # 갱신된 node4의 텐서 정보 출력

print()
a = tf.constant(5)
b = tf.constant(10)
c = tf.multiply(a, b)  # a * b 계산
result = tf.cond(a < b, lambda: tf.add(10, c), lambda: tf.square(c)) # 조건에 따라 다른 연산 수행
tf.print("Result:", result)  # 결과를 바로 출력

print('---------------')
v = tf.Variable(1)

@tf.function # autograph 기능에 의해 기본 Graph 객체 환경에서 작업(코드는 필요에 의해 자동 변환)
def find_nextFunc():
    v.assign(v + 1)
    if tf.equal(v % 2, 0):
        v.assign(v + 10)

find_nextFunc()
print(v.numpy())  # 변수 v의 현재 값을 numpy()로 출력
print(type(find_nextFunc))

print('1부터 3까지 합 출력 함수 작성')
def func1():
    imsi = tf.constant(0) # imsi = 0과 동일
    su = 1
    for _ in range(3):
        # imsi = tf.add(imsi, su)
        # imsi = imsi + su
        imsi += su
    return imsi

kbs = func1()
print(kbs.numpy(), ' ', np.array(kbs))  # 텐서 값을 numpy 배열로 변환하여 출력

print()
imsi = tf.constant(0) # 전역 변수로 선언
@tf.function
def func2():
    # imsi = tf.constant(0)
    global imsi
    su = 1
    for _ in range(3):
        imsi += su
    return imsi

mbc = func2()
print(mbc.numpy(), ' ', np.array(mbc))  # 텐서 값을 numpy 배열로 변환하여 출력

print()
imsi = tf.Variable(0) # 전역 변수로 선언
@tf.function
def func3():
    # imsi = tf.Variable(0) # 상태를 가지는 객체(값이 동적)
    su = 1
    for _ in range(3):
        imsi.assign_add(su)  # imsi = imsi + su
    return imsi

sbs = func3()
print(sbs.numpy(), ' ', np.array(sbs))  # 텐서 값을 numpy 배열로 변환하여 출력


print('구구단 출력-----------')

@tf.function
def gugu1(dan):
    su = tf.constant(0)
    for _ in range(9):
        su += tf.add(su, 1)
        # print(su)
        # tf.print(su)
        # print(su.numpy())
        # print('{}*{}={:2}'.format(dan, su, dan * su))
        # tf.print('{}*{}={:2}'.format(dan, su, dan * su))
        tf.print(dan, '*', su, '=', dan * su)

gugu1(3)

print()
@tf.function
def gugu2(dan):
    for i in range(1, 10):
        result = tf.multiply(dan, i) # 원소 곱 tf.matmul() : 행렬 곱
        tf.print(dan, '*', i, '=', result)

