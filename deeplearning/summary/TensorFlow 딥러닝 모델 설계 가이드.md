# TensorFlow 딥러닝 모델 설계 가이드

## Dense 층 파라미터 구조 비교

딥러닝 모델에서 Dense 층의 유닛 수를 어떻게 배치할지는 모델 성능에 큰 영향을 미칩니다. 대표적인 3가지 구조를 비교해보겠습니다.

### 1. 크게 → 작게 (Decreasing/Pyramid Structure)
```python
# 예시: 512 → 256 → 128 → 64
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
```

**특징**
- 입력을 넓게 받아 점점 압축하면서 중요한 특징만 추출
- 정보의 계층적 압축과 추상화

**장점**
- 고차원 입력에서 효과적인 특징 추출
- Autoencoder의 encoder 구조와 유사한 차원 축소 효과
- 과적합 방지에 도움 (점진적 정보 압축)
- 계산 효율성 (뒤로 갈수록 연산량 감소)

**단점**
- 초기 층에서 많은 파라미터로 인한 메모리 사용량 증가
- 초기 설정이 부적절하면 정보 손실 가능성

**추천 상황**
- 고차원 입력 데이터 (이미지, 텍스트 등)
- 복잡한 패턴에서 핵심 특징 추출이 필요한 경우
- 일반적인 분류/회귀 문제

### 2. 균일 (Uniform Structure)
```python
# 예시: 128 → 128 → 128 → 128
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
```

**특징**
- 모든 은닉층이 동일한 유닛 수
- 일관된 표현력 유지

**장점**
- 구조가 단순하고 직관적
- 하이퍼파라미터 튜닝 부담 감소
- 파라미터 수 예측이 용이
- 베이스라인 모델로 적합

**단점**
- 데이터 특성에 따른 최적화 부족
- 표현력의 다양성 제한

**추천 상황**
- 베이스라인 모델 구축
- 초기 실험 및 성능 비교 기준점
- 단순한 데이터 구조

### 3. 작게 → 크게 (Increasing/Inverted Pyramid)
```python
# 예시: 64 → 128 → 256 → 512
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
```

**특징**
- 점진적으로 유닛 수를 증가시켜 복잡한 표현 학습
- 정보의 확장과 조합

**장점**
- 초기 연산 부담 감소
- 뒤쪽 층에서 복잡한 특징 조합 가능
- 생성 모델의 디코더 구조에 적합

**단점**
- 뒤쪽 층에서 과적합 위험 증가
- 후반부 연산량 급증
- 그래디언트 소실 가능성

**추천 상황**
- 생성 모델 (GAN, VAE의 디코더)
- 입력이 단순하고 출력이 복잡한 경우
- 특징 확장이 필요한 특수 목적

### 4. 실무 권장사항

**일반적인 선택**: 크게 → 작게 (피라미드 구조)
- CNN, Transformer 등 대부분의 성공적인 아키텍처에서 채택
- 효율적인 정보 처리와 과적합 방지의 균형
- 컴퓨팅 리소스 효율성

**구조별 활용도**
- **크게→작게**: 80% (일반적인 분류, 회귀 문제)
- **균일**: 15% (베이스라인, 단순 구조)
- **작게→크게**: 5% (생성 모델, 특수 목적)

---

## 이항 분류에서 불균형 데이터 처리

불균형 데이터(예: 90% vs 10%)는 실무에서 자주 마주치는 문제입니다. 올바른 처리 방법을 알아보겠습니다.

### 1. 데이터 분할 시 클래스 비율 유지 (Stratify)

**왜 필요한가?**
- 무작위 분할 시 test set에 소수 클래스가 과소/과다 포함될 위험
- 학습과 평가 결과의 신뢰성 확보

```python
from sklearn.model_selection import train_test_split

# 클래스 비율을 유지하며 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
    shuffle=True,
    random_state=42,
    stratify=y  # 핵심: 클래스 비율 유지
)

# 분할 결과 확인
print("전체 데이터 클래스 비율:", np.bincount(y) / len(y))
print("훈련 데이터 클래스 비율:", np.bincount(y_train) / len(y_train))
print("테스트 데이터 클래스 비율:", np.bincount(y_test) / len(y_test))
```

### 2. 학습 데이터 불균형 보정 방법

**주의**: `stratify`는 분할 시에만 도움을 줄 뿐, 실제 학습에서의 불균형은 여전히 존재합니다.

#### 방법 A: 클래스 가중치 (Class Weight) - 추천
가장 간단하고 효과적인 방법입니다.

```python
from sklearn.utils import class_weight
import numpy as np

# 자동으로 균형 가중치 계산
class_weights = class_weight.compute_class_weight(
    'balanced', 
    classes=np.unique(y_train), 
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights))
print(f"클래스 가중치: {class_weights_dict}")
# 출력 예시: {0: 0.56, 1: 5.0} (소수 클래스에 높은 가중치)

# 모델 훈련 시 적용
model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weights_dict  # 가중치 적용
)
```

#### 방법 B: 오버샘플링 (SMOTE)
```python
from imblearn.over_sampling import SMOTE

# SMOTE를 이용한 합성 샘플 생성
smote = SMOTE(random_state=42)
x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)

print(f"원본 훈련 데이터: {np.bincount(y_train)}")
print(f"SMOTE 후 데이터: {np.bincount(y_train_balanced)}")
```

#### 방법 C: 언더샘플링
```python
from imblearn.under_sampling import RandomUnderSampler

# 다수 클래스 샘플 줄이기
undersampler = RandomUnderSampler(random_state=42)
x_train_balanced, y_train_balanced = undersampler.fit_resample(x_train, y_train)
```

### 3. 성능 평가 지표

불균형 데이터에서는 정확도(Accuracy)만으로는 모델 성능을 제대로 평가할 수 없습니다.

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 다양한 지표로 평가
y_pred = model.predict(x_test)
y_pred_classes = (y_pred > 0.5).astype(int)

print("=== 성능 평가 ===")
print("정확도:", accuracy_score(y_test, y_pred_classes))
print("AUC-ROC:", roc_auc_score(y_test, y_pred))
print("\n상세 리포트:")
print(classification_report(y_test, y_pred_classes))
print("\n혼동 행렬:")
print(confusion_matrix(y_test, y_pred_classes))
```

**추천 지표**
- **Precision**: 양성 예측의 정확도
- **Recall**: 실제 양성의 탐지율  
- **F1-Score**: Precision과 Recall의 조화평균
- **AUC-ROC**: 전체적인 분류 성능

### 4. 실무 권장사항

1. **항상 stratify=y 사용**: 데이터 분할 시 필수
2. **class_weight 우선 시도**: 가장 간단하고 효과적
3. **성능 지표 다양화**: Accuracy 외에 Precision, Recall, F1-Score 확인
4. **임계값 조정**: 필요에 따라 0.5 외의 threshold 사용 고려

```python
# 임계값 조정 예시
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
# 최적 임계값 찾기 (F1-Score 최대화)
f1_scores = 2 * precision * recall / (precision + recall)
optimal_threshold = thresholds[np.argmax(f1_scores)]
print(f"최적 임계값: {optimal_threshold:.3f}")
```

**결론**: 불균형 데이터 처리는 stratify로 시작해서 class_weight로 보정하고, 적절한 평가 지표로 검증하는 것이 핵심입니다.