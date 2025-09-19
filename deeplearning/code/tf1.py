import tensorflow as tf
print('즉시 실행모드: ',tf.executing_eagerly())
  # https://cafe.daum.net/flowlife/S2Ul/16

# Tensor 생성
print(1,type(1)) # 파이썬의 상수임
print(tf.constant(1),type(tf.constant(1))) # 스칼라임 0텐서. 0 dimensoin 텐서
print(tf.constant([1]),type(tf.constant([1])) ) # 1d tensor, 1차원배열이라생각
print(tf.constant([[1]]),type(tf.constant([[1]])) ) # 2d tensor
print()
a=tf.constant([1,2])
b=tf.constant([3,4])
c=a+b
print(c)
d=tf.constant([3]) # 브로드캐스팅: 작은 배열이 큰 배열의 크기에 맞게 "가상으로" 확장되어 연산이 가능해집니다.
e=c+d
print(e)
f=tf.add(c,d) # 텐서플로가 제공하는 더하기 해야 속도 빠르다
print(f)

print(7)
print(tf.convert_to_tensor(7,dtype=tf.float32)) # 첫번째 셀 수행했기땜에 오류 안 뜸
print(tf.cast(7,dtype=tf.float32))
print(tf.constant(7.0))
print(tf.constant(7,dtype=tf.float32))

# numpy의 ndarray와 tensor 사이에 type 자동 변환됨
import numpy as np
arr=np.array([1,2])
print(arr,type(arr))
tfarr=tf.add(arr,5) # 5가 자동으로 tensor 로 변환
print(tfarr)
tf.print(tfarr)
print(tfarr.numpy())
print(np.add(tfarr,3))

# 텐서플로는 텐서를 Graph 영역내에서 실행하는 것이 일반적이다
g1=tf.Graph() # 별도의 그래프 생성
with g1.as_default():
  c1=tf.constant(1,name='c_one')
  print(c1)
  print(type(c1))
  print(c1.op.node_def)
  # 파이선과 다르게 그래프 영역내에서 ..

  
