# 단층 신경망(뉴런, 노드) - Perceptron
# input의 가중치합에 임계값을 기준으로 2가지 output 중 하나를 출력하는 간단한 구조다.

# 단층 신경망으로 논리회로 분류
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

feature = np.array([[0,0], [0, 1], [1, 0], [1,1]]) 
print(feature)
# label = np.array([0, 0, 0, 1]) # AND gate
# label = np.array([0, 1, 1, 1]) # OR gate
label = np.array([0, 1, 1, 0]) # XOR gate
ml = Perceptron(max_iter=1000, eta0=0.1).fit(feature, label) # eta0 = learning_rate(학습률)
pred = ml.predict(feature) # 학습 전 예측
print('pred : ', pred)
print('acc :', accuracy_score(label, pred))
