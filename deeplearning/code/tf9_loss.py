# 텐스플로는 자동 미분(주어진 입력 변수에 대한 연산의 gradient를 계산하는 것)을 위한
# tf.GradientTape API를 제공한다.

import tensorflow as tf
import numpy as np

# keras.optimzers 패키지에 있는 Adam, SGD, RMSprop .. 사용


opti = tf.keras.optimizers.SGD(learning_rate=0.01)  # 확률적 경사 하강법

tf.random.set_seed(2)
w  = tf.Variable(tf.random.normal((1,))) # 가중치
b = tf.Variable(tf.random.normal((1,)))  # 편향

@tf.function
def train_step3(x, y):
    # GradientTape : 연산과정을 기억해 뒀다가 나중에 자동으로 미분
    with tf.GradientTape() as tape:
        hypo = tf.add(tf.multiply(w, x), b)  # 예측값 계산
        loss = tf.reduce_mean(tf.square(tf.subtract(hypo, y)))  # 손실 함수 계산 (예: MSE)
    grad = tape.gradient(loss, [w, b])  # 자동 미분

    opti.apply_gradients(zip(grad, [w, b]))  # 옵티마이저로 가중치 갱신

    return loss

x = [1., 2., 3., 4., 5.]
y = [1.2, 2.0, 3.0, 3.5, 5.5]

w_vals = []
cost_vals = []

for i in range(1, 101):
    cost_val = train_step3(x, y)
    cost_vals.append(cost_val.numpy())
    w_vals.append(w.numpy())
    if i % 10 == 0:
        print(cost_val)
print(cost_vals)
print(w_vals)

import matplotlib.pyplot as plt
plt.plot(w_vals, cost_vals, 'o--')
plt.xlabel('w')
plt.ylabel('cost')
# plt.show()
plt.savefig('cost.png')  # 그래프를 파일로 저장
plt.close()

print('cost가 최소일 때 w :', w.numpy())
print('cost가 최소일 때 b :', b.numpy())

y_pred = tf.multiply(x, w)
print('y_pred: ', y_pred)

plt.plot(x, y, 'ro', label='real')  # 실제값
plt.plot(x, y_pred, 'b-', label='pred')  # 예측값
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
# plt.show()
plt.savefig('pred.png')  # 그래프를 파일로 저장
plt.close()

# 새 값으로 예측하기
new_x = [3.5, 9.0]
new_pred = tf.multiply(new_x, w) + b
print('예측 결과: ', new_pred.numpy())