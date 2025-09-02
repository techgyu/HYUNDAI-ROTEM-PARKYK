# 의사결정 나무(Decision Tree) - CART
# 예측 분류 모두 가능하나 분류가 주 목적이다.
# 비모수 검정 : 선형성, 정규성, 등분산성 가정 필요 없음.
# 유의수준 판단 기준이 없다. 과적합으로 예측 정확도가 낮을 수 있다.

# 키와 머리카락 길이로 성별 구분 모델 작성
import matplotlib.pyplot as plt
from sklearn import tree

# x = [[180, 15], [177, 42], [156, 35], [174, 65], [161, 28], [160, 5], [170, 12], [176, 75], [170, 22], [175, 28]]
# y = ['man', 'woman', 'woman', 'man', 'woman', 'man', 'man', 'man', 'man', 'woman']

x=[[180, 15], [177, 40], [156, 35], [174, 65], [161, 28], [160, 5], [170, 12], [176, 75], [170, 22], [175, 28]] # 싸이키런이기땜에 독립 2차원
y=['man', 'woman', 'woman', 'man', 'woman', 'man', 'man', 'man', 'man', 'woman'] # 종속 1차원
feature_names = ['height', 'hair length']
class_names = ['man', 'woman']

model = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 5, random_state = 0 )

# max_depth: 트리의 최대 깊이를 지정
# min_samples_split: 노드를 분할하기 위한 최소한의 샘플 데이터 수로 과적합 제어
# min_samples_leaf: 말단 노드(leaf)가 되기 위한 최소한의 샘플 데이터 수. 과적합 제어
# min_weight_fraction_leaf: 말단 노드(leaf)가 되기 위한 최소한의 샘플 데이터 비율. 과적합 제어
# max_features: 최적 분할을 위한 최대 특성 수. 과적합 제어

model.fit(x, y)

print('훈련 데이터 정확도 : {:.3f}'.format(model.score(x, y)))
print("예측 결과: ", model.predict(x))
print("실제값: ", y)

#시각화
plt.figure(figsize=(10,6))
tree.plot_tree(model,feature_names=feature_names
               ,class_names=class_names,filled=True,rounded=True,fontsize=12)
plt.show()