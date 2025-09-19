# 텐서플로 gradient tape
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Keras.optimizers 패키지에 있는 Adam,SGD,RMSprop,... 사용
opti=tf.keras.optimizers.SGD(learning_rate=0.01)

x=tf.Variable(5.0)
w=tf.Variable(0.0)

@tf.function
def train_step():
  # GradientTape 연산과정을 기억해뒀다가 나중에 자동으로 미분(gradient)을 계산함
  with tf.GradientTape() as tape:
    y=tf.multiply(w,x) # b=0으로 간주
    loss=tf.square(tf.subtract(y,50)) # 예측값, 실제값 빼서 제곱
  grad=tape.gradient(loss,w) # 자동으로 미분이 계산됨
  opti.apply_gradients([(grad,w)])
  return loss

for i in range(10):
  loss=train_step()
  print(f' i: {i} \n w:{w.numpy()} \n loss: {loss.numpy()}')

# 선형회귀 모형 작성
# Keras.optimizers 패키지에 있는 Adam,SGD,RMSprop,... 사용
opti=tf.keras.optimizers.SGD(learning_rate=0.01)

x=tf.Variable(5.0)
w=tf.Variable(0.0)

@tf.function
def train_step2():
  # GradientTape 연산과정을 기억해뒀다가 나중에 자동으로 미분(gradient)을 계산함
  with tf.GradientTape() as tape:
    y=tf.multiply(w,x) # b=0으로 간주
    loss=tf.square(tf.subtract(y,50)) # 예측값, 실제값 빼서 제곱
  grad=tape.gradient(loss,w) # 자동으로 미분이 계산됨
  opti.apply_gradients([(grad,w)])
  return loss

for i in range(10):
  loss=train_step2()
  print(f' i: {i} \n w:{w.numpy()} \n loss: {loss.numpy()}')

  # Keras.optimizers 패키지에 있는 Adam,SGD,RMSprop,... 사용
# 👉 tf.keras.optimizers 쓰면 SGD, Adam, RMSprop 등 고급 최적화 알고리즘을 바로 활용 가능해요.
opti=tf.keras.optimizers.SGD(learning_rate=0.01)

tf.random.set_seed(2)
w=tf.Variable(tf.random.normal((1,)))
b=tf.Variable(tf.random.normal((1,)))

@tf.function
def train_step3(x, y):
  with tf.GradientTape() as tape:
    hypo = tf.add(tf.multiply(w, x), b)
    loss = tf.reduce_mean(tf.square(tf.subtract(hypo, y)))
  grad = tape.gradient(loss, [w, b])
  opti.apply_gradients(zip(grad, [w, b]))
  return loss

x=[1.,2.,3.,4.,5.] #feature
y=[1.2,2.0,3.0,3.5,5.5] #label

w_vals=[]
cost_vals=[]

# for문 돌리고 결과 시각화
for i in range(1,101):
  cost_val=train_step3(x,y)
  cost_vals.append(cost_val.numpy())
  w_vals.append(w.numpy())
  if i%10==0:
    print(cost_val)
print(cost_vals)
print(w_vals)

plt.plot(w_vals,cost_vals)
plt.show()

print('cost가 최소일때 w',w.numpy())
print('cost가 최소일때 b',b.numpy())

y_pred=tf.multiply(x,w)+b
print('y_pred',y_pred)
plt.plot(x,y,'ro',label='real')
plt.plot(x,y_pred,'b-',label='pred')
plt.xlabel('x')
plt.ylabel('y')
# plt.show()
plt.savefig('pred.png')  # 그래프를 파일로 저장
plt.close()

# 새값으로 예측하기
new_x=[3.5,9.0]
new_pred=tf.multiply(new_x,w)+b
print('예측 결과',new_pred.numpy())

