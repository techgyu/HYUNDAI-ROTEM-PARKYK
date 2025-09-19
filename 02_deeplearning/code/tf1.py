import tensorflow as tf  # 텐서플로 라이브러리 임포트
print('즉시 실행모드: ',tf.executing_eagerly())  # 즉시 실행 모드가 활성화되어 있는지 확인
# https://cafe.daum.net/flowlife/S2Ul/16

# Tensor 생성 예시
print(1,type(1))  # 파이썬의 상수(정수)와 타입 출력
print(tf.constant(1),type(tf.constant(1)))  # 0차원(스칼라) 텐서 생성 및 타입 출력
print(tf.constant([1]),type(tf.constant([1])))  # 1차원 텐서 생성 및 타입 출력
print(tf.constant([[1]]),type(tf.constant([[1]])))  # 2차원 텐서 생성 및 타입 출력
print()  # 빈 줄 출력
a=tf.constant([1,2])  # 1차원 텐서 a 생성
b=tf.constant([3,4])  # 1차원 텐서 b 생성
c=a+b  # 텐서 a와 b를 더함(요소별 덧셈)
print(c)  # 결과 텐서 c 출력
d=tf.constant([3])  # 1차원 텐서 d 생성(브로드캐스팅 예시)
e=c+d  # c와 d를 더함(브로드캐스팅 적용)
print(e)  # 결과 텐서 e 출력
f=tf.add(c,d)  # tf.add 함수로 c와 d를 더함(속도 빠름)
print(f)  # 결과 텐서 f 출력

print(7)  # 정수 7 출력
print(tf.convert_to_tensor(7,dtype=tf.float32))  # 7을 float32 타입 텐서로 변환
print(tf.cast(7,dtype=tf.float32))  # 7을 float32 타입으로 형변환
print(tf.constant(7.0))  # 7.0(실수)로 텐서 생성
print(tf.constant(7,dtype=tf.float32))  # 7을 float32 타입으로 텐서 생성

# numpy의 ndarray와 tensor 사이에 type 자동 변환됨
import numpy as np  # 넘파이 임포트
arr=np.array([1,2])  # 넘파이 배열 생성
print(arr,type(arr))  # 넘파이 배열과 타입 출력
tfarr=tf.add(arr,5)  # 넘파이 배열과 5를 더하면 5가 자동으로 텐서로 변환됨
print(tfarr)  # 결과 텐서 출력
tf.print(tfarr)  # tf.print로 텐서 값 출력
print(tfarr.numpy())  # 텐서를 넘파이 배열로 변환해 출력
print(np.add(tfarr,3))  # 텐서에 3을 더해 넘파이 배열로 출력

# 텐서플로는 텐서를 Graph 영역내에서 실행하는 것이 일반적이다
g1=tf.Graph()  # 별도의 그래프 객체 생성
with g1.as_default():  # g1을 기본 그래프로 설정
  c1=tf.constant(1,name='c_one')  # 그래프 내 상수 텐서 생성
  print(c1)  # 텐서 객체 출력
  print(type(c1))  # 텐서 타입 출력
  print(c1.op.node_def)  # 텐서의 노드 정의 출력
  # 파이썬과 다르게 그래프 영역내에서 연산이 이루어짐

  
