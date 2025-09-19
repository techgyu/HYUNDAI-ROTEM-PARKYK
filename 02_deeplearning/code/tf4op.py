
import tensorflow as tf
import numpy as np

x = tf.constant(7)
y = tf.constant(3)

# 삼항연산 : cond
# x > y이면 x+y, 아니면 x-y를 반환
result1 = tf.cond(x > y, lambda:tf.add(x,y),lambda:tf.subtract(x,y))
print(result1)

# case 조건문: 여러 조건 중 첫 번째가 참이면 f1() 실행, 아니면 f2() 실행
f1 = lambda:tf.constant(1)
print(f1())  # f1() 결과 출력
f2 = lambda:tf.constant(2)
a = tf.constant(3)
b = tf.constant(4)
result2 = tf.case([(tf.less(a, b), f1)], default=f2)
print(result2)

# 관계연산: 두 값의 관계를 비교해 True/False 반환
print(tf.equal(1,2))         # 1 == 2
print(tf.not_equal(1,2))    # 1 != 2
print(tf.greater(1, 2))     # 1 > 2
print(tf.greater_equal(1, 2)) # 1 >= 2
print(tf.less(1, 2))        # 1 < 2

# 논리 연산: 논리값(True/False)에 대한 AND, OR, NOT 연산
print(tf.logical_and(True, False))  # AND
print(tf.logical_or(True, False))   # OR
print(tf.logical_not(True))         # NOT

# 유일 합집합: 중복을 제거한 유일값(val)과 각 원소의 인덱스(idx) 반환
kbs = tf.constant([1, 2, 2, 2, 2, 3])
val, idx = tf.unique(kbs)
print(val.numpy())  # 유일값 출력
print(idx.numpy())  # 각 원소가 유일값 중 몇 번째인지 인덱스 출력

# reduce ~: 배열의 평균, 축별 평균 등 통계 연산
ar = [[1., 2.], [3., 4.]]
print(tf.reduce_mean(ar).numpy()) # 전체 평균
print(tf.reduce_mean(ar, axis = 0).numpy())  # 각 열의 평균
print(tf.reduce_mean(ar, axis = 1).numpy())  # 각 행의 평균

# ...