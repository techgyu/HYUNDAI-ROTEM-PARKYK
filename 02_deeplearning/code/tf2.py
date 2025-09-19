
# 텐서플로에서 변수 사용 예시
import tensorflow as tf  # 텐서플로 임포트
f=tf.Variable(1.0)  # 스칼라(float) 변수 생성
v=tf.Variable(tf.ones((2,)))  # 1차원 벡터 변수 생성
m=tf.Variable(tf.ones((2,1)))  # 2차원 행렬 변수 생성
print(f)  # 변수 f 출력
print(v)  # 변수 v 출력
print(m)  # 변수 m 출력

tf.print(m)  # 텐서플로 방식으로 m 출력
print()  # 빈 줄 출력
# 치환(assign) 예시
v1=tf.Variable(1)  # 정수형 변수 v1 생성
v1.assign(10)  # v1에 10 할당
print(v1)  # v1 출력

v2=tf.Variable(tf.ones(shape=(1)))  # 1차원 벡터 변수 v2 생성
v2.assign([20])  # v2에 20 할당
print(v2)  # v2 출력

v3=tf.Variable(tf.ones(shape=(1,2)))  # 2차원 행렬 변수 v3 생성
v3.assign([[30,40]])  # v3에 [30,40] 할당
print(v3)  # v3 출력

v1=tf.Variable([3],dtype=tf.float32)  # float32 타입 1차원 변수 v1 생성
v1=tf.Variable([5],dtype=tf.float32)  # v1을 5로 재할당
v3=(v1*v2)+10  # v1과 v2 곱한 뒤 10 더함
print(v3)  # v3 출력
var=tf.Variable([1,2,3,4,5],dtype=tf.float32)  # 1차원 변수 var 생성
result=var+10  # var의 각 요소에 10 더함
print(result)  # result 출력

w=tf.Variable(tf.ones(shape=(1,)))  # 가중치 변수 w 생성
b=tf.Variable(tf.ones(shape=(1,)))  # 편향 변수 b 생성
w.assign([2])  # w에 2 할당
b.assign([3])  # b에 3 할당

# 파이썬의 일반 함수 정의
def func1(x):  # 입력 x에 대해
  return w*x+b  # w*x+b 계산

out_a1=func1([3])  # func1에 3을 넣어 계산
print('\n out_a1,type(func1)',out_a1,type(func1))  # 결과와 함수 타입 출력

# 그래프 영역 밖에서 진행하므로 의미 없음
@tf.function  # auto graph 기능: 그래프 영역에서 실행
def func2(x):  # 입력 x에 대해
  return w*x+b  # w*x+b 계산

out_a2=func2([1,2])  # func2에 [1,2]를 넣어 계산
print(out_a2,type(func2))  # 결과와 함수 타입 출력
print('\n out_a2,type(func2)',out_a2,type(func2))  # 결과와 함수 타입 출력

# | 구분    | 파이썬      | 텐서플로우 Graph                             |
# | ----- | -------- | --------------------------------------- |
# | 실행 시점 | 즉시 실행    | 그래프를 정의 후 실행(Session/TF2는 @tf.function) |
# | 결과    | 바로 값 계산됨 | Tensor 객체(노드)로 존재, 실행해야 값 나옴            |
# | 장점    | 직관적, 단순  | 속도 최적화, GPU/분산처리 유리                     |
# | 예시    | `3+5=8`  | `c = a+b` (노드), 실행해야 8                  |

# 난수 최소 0 최대 1
rand=tf.random.uniform([4],0,1)  # 0~1 균등분포 난수 4개 생성
print(rand)  # rand 출력
rand2=tf.random.normal([4],0,1)  # 평균0, 표준편차1 정규분포 난수 4개 생성
print(rand2)  # rand2 출력

aa=tf.ones((2,1))  # 2x1 행렬(모두 1) 생성
print(aa.numpy())  # 넘파이 배열로 출력
m=tf.Variable(tf.zeros((2,1)))  # 2x1 행렬(모두 0) 변수 생성
print(m.numpy())  # 넘파이 배열로 출력
m.assign(aa)  # m에 aa를 할당
print(m.numpy())  # m 출력
m.assign_add(aa)  # m에 aa를 더해 재할당
print(m.numpy())  # m 출력
m.assign_sub(aa)  # m에서 aa를 빼서 재할당
print(m.numpy())  # m 출력
