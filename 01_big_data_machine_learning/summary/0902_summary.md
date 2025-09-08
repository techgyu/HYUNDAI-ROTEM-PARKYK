# 랜덤 포레스트(Random Forest)

랜덤 포레스트는 여러 개의 결정 트리(Decision Tree)를 조합하여 예측 성능을 높이는 앙상블 학습 방법입니다. 각 트리는 데이터의 일부 샘플과 일부 특성(feature)을 무작위로 선택하여 학습합니다. 이렇게 만들어진 트리들의 예측 결과를 다수결(분류) 또는 평균(회귀)으로 결합하여 최종 결과를 도출합니다.

## 주요 특징
- **과적합 감소**: 여러 트리의 예측을 결합하므로 단일 트리보다 과적합 위험이 낮음
- **샘플링 방식**: 각 트리는 원본 데이터에서 중복을 허용하는 부트스트랩 샘플링(bootstrap sampling)으로 데이터를 선택
- **특성 무작위 선택**: 각 노드에서 분할할 때 일부 특성만 고려하여 트리 간 다양성 증가
- **병렬 처리 가능**: 각 트리의 학습이 독립적이므로 병렬 처리가 용이함

## 장점
- 높은 예측 성능
- 다양한 데이터에 적용 가능
- 변수 중요도(feature importance) 제공
- 이상치(outlier)에 강함

## 단점
- 모델 해석이 어려움(블랙박스)
- 많은 트리 사용 시 메모리와 계산 비용 증가

## 주요 파라미터
- `n_estimators`: 생성할 트리의 개수
- `max_features`: 각 노드에서 분할에 사용할 특성의 수
- `max_depth`: 각 트리의 최대 깊이
- `min_samples_split`: 노드를 분할하기 위한 최소 샘플 수

## 사용 예시 (scikit-learn)
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
```

