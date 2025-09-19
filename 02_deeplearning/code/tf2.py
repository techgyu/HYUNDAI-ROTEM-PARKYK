# 텐서플로에서 변수
import tensorflow as tf
f=tf.Variable(1.0) # 변수에 기억된다, 스칼라 기억함
v=tf.Variable(tf.ones((2,))) # 1차원
m=tf.Variable(tf.ones((2,1))) # 2차원
print(f)
print(v)
print(m)

tf.print(m)
print()
# 치환
v1=tf.Variable(1)
v1.assign(10) # 변수에 값을 할당
print(v1) # 스칼라를 준것

v2=tf.Variable(tf.ones(shape=(1)))
v2.assign([20])
print(v2)

v3=tf.Variable(tf.ones(shape=(1,2)))
v3.assign([[30,40]])
print(v3)

v1=tf.Variable([3],dtype=tf.float32)
v1=tf.Variable([5],dtype=tf.float32)
v3=(v1*v2)+10
print(v3)
var=tf.Variable([1,2,3,4,5],dtype=tf.float32)
result=var+10
print(result)

w=tf.Variable(tf.ones(shape=(1,)))
b=tf.Variable(tf.ones(shape=(1,)))
w.assign([2])
b.assign([3])

# 파이선의 일반함수
def func1(x):
  return w*x+b

out_a1=func1([3])
print('\n out_a1,type(func1)',out_a1,type(func1))

# 그래프영역 밖에서 진행하므로 의미X ,
@tf.function # auto graph 기능
def func2(x):
  return w*x+b

out_a2=func2([1,2])
print(out_a2,type(func2))
print('\n out_a2,type(func2)',out_a2,type(func2))

# | 구분    | 파이썬      | 텐서플로우 Graph                             |
# | ----- | -------- | --------------------------------------- |
# | 실행 시점 | 즉시 실행    | 그래프를 정의 후 실행(Session/TF2는 @tf.function) |
# | 결과    | 바로 값 계산됨 | Tensor 객체(노드)로 존재, 실행해야 값 나옴            |
# | 장점    | 직관적, 단순  | 속도 최적화, GPU/분산처리 유리                     |
# | 예시    | `3+5=8`  | `c = a+b` (노드), 실행해야 8                  |

# 난수 최소 0 최대 1
rand=tf.random.uniform([4],0,1) # 균등 분포
print(rand)
rand2=tf.random.normal([4],0,1)
print(rand2)

aa=tf.ones((2,1))
print(aa.numpy())
m=tf.Variable(tf.zeros((2,1)))
print(m.numpy())
m.assign(aa) # 치환
print(m.numpy())
m.assign_add(aa) # 더하기 후 치환
print(m.numpy())
m.assign_sub(aa) # 빼기 후 치환
print(m.numpy())
