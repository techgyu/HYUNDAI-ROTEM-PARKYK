# 문제2) 21세 이상의 피마 인디언 여성의 당뇨병 발병 여부에 대한 dataset을 이용하여 당뇨 판정을 위한 분류 모델을 작성한다.
# 피마 인디언 당뇨병 데이터는 아래와 같이 구성되어 있다.
#   Pregnancies: 임신 횟수
#   Glucose: 포도당 부하 검사 수치
#   BloodPressure: 혈압(mm Hg)
#   SkinThickness: 팔 삼두근 뒤쪽의 피하지방 측정값(mm)
#   Insulin: 혈청 인슐린(mu U/ml)
#   BMI: 체질량지수(체중(kg)/키(m))^2
#   DiabetesPedigreeFunction: 당뇨 내력 가중치 값
#   Age: 나이
#   Outcome: 5년 이내 당뇨병 발생여부 - 클래스 결정 값(0 또는 1)
# 당뇨 판정 칼럼은 outcome 이다.   1 이면 당뇨 환자로 판정
# train / test 분류 실시
# 모델 작성은 Sequential API, Function API 두 가지를 사용한다.
# loss, accuracy에 대한 시각화도 실시한다.
# 출력결과는 Django framework를 사용하시오.

import pandas as pd  # 데이터프레임 조작, CSV 파일 읽기
from sklearn.model_selection import train_test_split  # 데이터를 훈련/테스트 세트 분할
from sklearn.preprocessing import StandardScaler, PolynomialFeatures  # 데이터 정규화 및 다항식 특성 생성
from sklearn.ensemble import VotingClassifier  # 앙상블 학습을 위한 투표 분류기
from sklearn.metrics import classification_report, confusion_matrix  # 모델 성능 평가 지표
from tensorflow.keras.models import Sequential  # 순차적 레이어 구성을 위한 모델
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization  # 신경망 레이어들
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # 학습 최적화 콜백들
from tensorflow.keras.optimizers import Adam  # Adam 최적화 알고리즘
from tensorflow.keras.regularizers import l2  # L2 정규화로 과적합 방지
import matplotlib.pyplot as plt  # 시각화를 위한 그래프 라이브러리
import numpy as np  # 수치 연산을 위한 라이브러리
import tensorflow as tf  # 딥러닝 프레임워크

# 0. 랜덤 시드 설정 - 실험 재현성을 위해 난수 생성기 초기화
np.random.seed(42)  # NumPy 난수 시드 고정
tf.random.set_seed(42)  # TensorFlow 난수 시드 고정

# 1. 데이터 로딩 및 분석
data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/pima-indians-diabetes.data.csv')

# 컬럼명 추가 (피마 인디언 당뇨병 데이터는 컬럼명이 없어서 직접 지정)
data.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

print("=== 데이터 정보 ===")  # 데이터 정보 출력 헤더
print(data.info())
print("\n=== 기본 통계 ===")  # 기본 통계 정보 출력 헤더
print(data.describe())
print("\n=== 클래스 분포 ===")  # 클래스 분포 출력 헤더
print(data['Outcome'].value_counts())  # 당뇨 발병(1)과 비발병(0)의 개수 분포 확인

# 2. 입 출력 분리
x = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]  # 입력 특성
y = data['Outcome']  # 출력(타겟): 당뇨병 여부 (0 또는 1)

# 3. train / test 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
# 데이터를 8:2 비율로 훈련/테스트 세트로 분할, 클래스 비율 유지(stratify)

# 3.1 고급 데이터 전처리
# 표준화
scaler = StandardScaler()  # 표준화 객체 생성 (평균=0, 표준편차=1로 변환)
x_train_scaled = scaler.fit_transform(x_train)  # 훈련 데이터로 스케일링 학습 후 변환
x_test_scaled = scaler.transform(x_test)  # 훈련 데이터의 스케일링 파라미터로 테스트 데이터 변환

# 다항식 특성 추가 (차수 2까지)
poly = PolynomialFeatures(degree=2, include_bias=False)  # 2차 다항식 특성 생성기 (편향 제외)
x_train_poly = poly.fit_transform(x_train_scaled)  # 훈련 데이터에 다항식 특성 적용
x_test_poly = poly.transform(x_test_scaled)  # 테스트 데이터에 동일한 다항식 특성 적용

print(f"원본 특성 수: {x_train_scaled.shape[1]}")  # 원본 특성 개수 출력 (3개)
print(f"다항식 특성 수: {x_train_poly.shape[1]}")  # 다항식 확장 후 특성 개수 출력 (9개)

# 3.2 고급 콜백 설정
# 정확도에 대한 조기 종료
early_stopping = EarlyStopping(
    monitor='val_accuracy',  # 검증 정확도를 모니터링
    patience=15,  # 성능 개선이 없으면 15 에포크 후 학습 중단
    restore_best_weights=True,  # 최고 성능 시점의 가중치로 복원
    verbose=1  # 조기 종료 시 메시지 출력
)

# 손실에 대한 학습률 감소
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',  # 검증 손실을 모니터링
    factor=0.2,  # 학습률을 0.2배로 감소
    patience=10,  # 성능 개선이 없으면 10 에포크 후 학습률 감소
    min_lr=0.0001,  # 최소 학습률 설정
    verbose=1  # 학습률 감소 시 메시지 출력
)

# 4.1 모델 작성(Sequential API) - 다항식 특성 사용
Sequential_model1 = Sequential([
    Dense(128, input_dim=x_train_poly.shape[1], activation='relu', kernel_regularizer=l2(0.001)),
    # 첫 번째 은닉층: 128개 뉴런, indput_dim: 입력으로 받을 특성 개수, ReLU 활성화, L2 정규화 적용
    # L2 정규화: 과적합 방지, 일반화 성능 향상, 괄호 안의 숫자를 조정하여 정규화 강도 조절 가능
    BatchNormalization(),  # 배치 정규화로 학습 안정성 향상
    Dropout(0.4),  # 40% 드롭아웃으로 과적합 방지
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    # 두 번째 은닉층: 64개 뉴런, ReLU 활성화, L2 정규화 적용
    BatchNormalization(),  # 배치 정규화 적용
    Dropout(0.3),  # 30% 드롭아웃 적용
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    # 세 번째 은닉층: 32개 뉴런, ReLU 활성화, L2 정규화 적용
    BatchNormalization(),  # 배치 정규화 적용
    Dropout(0.2),  # 20% 드롭아웃 적용
    Dense(16, activation='relu'),  # 네 번째 은닉층: 16개 뉴런, ReLU 활성화
    Dense(1, activation='sigmoid')  # 출력층: 1개 뉴런, 시그모이드 활성화 (이진 분류)
])

Sequential_model1.compile(
    loss='binary_crossentropy',  # 이진 분류용 손실 함수(합격 VS 불합격)
    optimizer=Adam(learning_rate=0.001),  # Adam 옵티마이저, 학습률 0.001
    metrics=['accuracy']  # 평가 지표로 정확도 사용
)

history1 = Sequential_model1.fit(
    x_train_poly, y_train,  # 훈련 데이터와 타겟
    epochs=300,  # 최대 300 에포크 학습
    batch_size=8,  # 배치 크기 8로 설정
    validation_split=0.2,  # 훈련 데이터의 20%를 검증용으로 분할
    callbacks=[early_stopping, reduce_lr],  # 조기 종료와 학습률 감소 콜백 적용
    verbose=0  # 학습 과정 출력
)
loss1, accuracy1 = Sequential_model1.evaluate(x_test_poly, y_test, verbose=0)
# 테스트 데이터로 모델 성능 평가

# 4.2 고급 모델 작성(Function API) - 다항식 특성 사용
inputs = tf.keras.Input(shape=(x_train_poly.shape[1],))  # 입력층 정의: 다항식 특성 개수만큼의 입력
x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(inputs)
# 첫 번째 은닉층: 128개 뉴런, ReLU 활성화, L2 정규화
x = BatchNormalization()(x)  # 배치 정규화 적용
x = Dropout(0.4)(x)  # 40% 드롭아웃 적용
x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
# 두 번째 은닉층: 64개 뉴런, ReLU 활성화, L2 정규화
x = BatchNormalization()(x)  # 배치 정규화 적용
x = Dropout(0.3)(x)  # 30% 드롭아웃 적용
x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
# 세 번째 은닉층: 32개 뉴런, ReLU 활성화, L2 정규화
x = BatchNormalization()(x)  # 배치 정규화 적용
x = Dropout(0.2)(x)  # 20% 드롭아웃 적용
x = Dense(16, activation='relu')(x)  # 네 번째 은닉층: 16개 뉴런, ReLU 활성화
outputs = Dense(1, activation='sigmoid')(x)  # 출력층: 시그모이드 활성화

Functional_model2 = tf.keras.Model(inputs=inputs, outputs=outputs)  # 함수형 모델 생성
Functional_model2.compile(
    loss='binary_crossentropy',  # 이진 분류용 손실 함수
    optimizer=Adam(learning_rate=0.001),  # Adam 옵티마이저, 학습률 0.001
    metrics=['accuracy']  # 평가 지표로 정확도 사용
)

history2 = Functional_model2.fit(
    x_train_poly, y_train,  # 훈련 데이터와 타겟
    epochs=300,  # 최대 300 에포크 학습
    batch_size=8,  # 배치 크기 8로 설정
    validation_split=0.2,  # 훈련 데이터의 20%를 검증용으로 분할
    callbacks=[early_stopping, reduce_lr],  # 조기 종료와 학습률 감소 콜백 적용
    verbose=0  # 학습 과정 출력
)
loss2, accuracy2 = Functional_model2.evaluate(x_test_poly, y_test, verbose=0)
# 테스트 데이터로 모델 성능 평가

# 4.3 고급 모델 작성(Subclassing API) - 다항식 특성 사용
class AdvancedModel(tf.keras.Model):
    def __init__(self, input_dim):
        super(AdvancedModel, self).__init__()  # 부모 클래스 초기화
        # 모든 레이어를 클래스 속성으로 정의
        self.dense1 = Dense(128, activation='relu', kernel_regularizer=l2(0.001))  # 첫 번째 은닉층
        self.bn1 = BatchNormalization()  # 첫 번째 배치 정규화
        self.dropout1 = Dropout(0.4)  # 첫 번째 드롭아웃 (40%)
        self.dense2 = Dense(64, activation='relu', kernel_regularizer=l2(0.001))  # 두 번째 은닉층
        self.bn2 = BatchNormalization()  # 두 번째 배치 정규화
        self.dropout2 = Dropout(0.3)  # 두 번째 드롭아웃 (30%)
        self.dense3 = Dense(32, activation='relu', kernel_regularizer=l2(0.001))  # 세 번째 은닉층
        self.bn3 = BatchNormalization()  # 세 번째 배치 정규화
        self.dropout3 = Dropout(0.2)  # 세 번째 드롭아웃 (20%)
        self.dense4 = Dense(16, activation='relu')  # 네 번째 은닉층
        self.output_layer = Dense(1, activation='sigmoid')  # 출력층

    def call(self, inputs, training=False):
        # 순전파 과정 정의
        x = self.dense1(inputs)  # 첫 번째 은닉층 통과
        x = self.bn1(x, training=training)  # 배치 정규화 (훈련 모드에 따라)
        x = self.dropout1(x, training=training)  # 드롭아웃 (훈련 모드에만 적용)
        x = self.dense2(x)  # 두 번째 은닉층 통과
        x = self.bn2(x, training=training)  # 배치 정규화
        x = self.dropout2(x, training=training)  # 드롭아웃
        x = self.dense3(x)  # 세 번째 은닉층 통과
        x = self.bn3(x, training=training)  # 배치 정규화
        x = self.dropout3(x, training=training)  # 드롭아웃
        x = self.dense4(x)  # 네 번째 은닉층 통과
        return self.output_layer(x)  # 출력층을 통해 최종 결과 반환
    
Subclass_model3 = AdvancedModel(x_train_poly.shape[1])  # 서브클래싱 모델 인스턴스 생성
Subclass_model3.compile(
    loss='binary_crossentropy',  # 이진 분류용 손실 함수
    optimizer=Adam(learning_rate=0.001),  # Adam 옵티마이저, 학습률 0.001
    metrics=['accuracy']  # 평가 지표로 정확도 사용
)

history3 = Subclass_model3.fit(
    x_train_poly, y_train,  # 훈련 데이터와 타겟
    epochs=300,  # 최대 300 에포크 학습
    batch_size=8,  # 배치 크기 8로 설정
    validation_split=0.2,  # 훈련 데이터의 20%를 검증용으로 분할
    callbacks=[early_stopping, reduce_lr],  # 조기 종료와 학습률 감소 콜백 적용
    verbose=0  # 학습 과정 출력
)
loss3, accuracy3 = Subclass_model3.evaluate(x_test_poly, y_test, verbose=0)
# 테스트 데이터로 모델 성능 평가

# 4.4 앙상블 모델 추가
def create_ensemble_model():
    """다양한 구조의 모델들로 앙상블 생성"""
    models = []  # 앙상블에 포함할 모델들을 저장할 리스트
    
    # 모델 1: 광범위한 네트워크 - 큰 레이어로 시작해서 점진적으로 줄임
    model1 = Sequential([
        Dense(256, input_dim=x_train_poly.shape[1], activation='relu', kernel_regularizer=l2(0.001)),
        # 256개 뉴런의 첫 번째 은닉층, L2 정규화 적용
        Dropout(0.5),  # 50% 드롭아웃으로 강한 정규화
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),  # 128개 뉴런의 두 번째 은닉층
        Dropout(0.3),  # 30% 드롭아웃
        Dense(64, activation='relu'),  # 64개 뉴런의 세 번째 은닉층
        Dense(1, activation='sigmoid')  # 최종 출력층 (확률 출력)
    ])
    
    # 모델 2: 깊은 네트워크 - 배치 정규화를 포함한 더 깊은 구조
    model2 = Sequential([
        Dense(64, input_dim=x_train_poly.shape[1], activation='relu', kernel_regularizer=l2(0.001)),
        # 64개 뉴런의 첫 번째 은닉층
        BatchNormalization(),  # 배치 정규화로 안정적인 학습
        Dropout(0.3),  # 30% 드롭아웃
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),  # 동일한 크기의 두 번째 은닉층
        BatchNormalization(),  # 배치 정규화
        Dropout(0.3),  # 30% 드롭아웃
        Dense(32, activation='relu'),  # 32개 뉴런의 세 번째 은닉층
        Dense(16, activation='relu'),  # 16개 뉴런의 네 번째 은닉층
        Dense(1, activation='sigmoid')  # 최종 출력층
    ])
    
    # 모델 3: 작지만 정확한 네트워크 - 단순하지만 효과적인 구조
    model3 = Sequential([
        Dense(32, input_dim=x_train_poly.shape[1], activation='relu'),  # 32개 뉴런의 첫 번째 은닉층
        Dropout(0.2),  # 낮은 드롭아웃률(20%)로 약간의 정규화
        Dense(16, activation='relu'),  # 16개 뉴런의 두 번째 은닉층
        Dense(1, activation='sigmoid')  # 최종 출력층
    ])
    
    models.extend([model1, model2, model3])  # 세 모델을 리스트에 추가
    
    # 각 모델 컴파일 및 훈련
    for i, model in enumerate(models):
        model.compile(
            loss='binary_crossentropy',  # 이진 분류용 손실 함수
            optimizer=Adam(learning_rate=0.001),  # Adam 옵티마이저
            metrics=['accuracy']  # 평가 지표로 정확도 사용
        )
        print(f"Training ensemble model {i+1}/3...")  # 현재 훈련 중인 모델 번호 출력
        model.fit(
            x_train_poly, y_train,  # 다항식 특성 데이터로 훈련
            epochs=200,  # 200 에포크 훈련
            batch_size=8,  # 배치 크기 8
            validation_split=0.2,  # 검증 데이터 20% 분할
            callbacks=[early_stopping, reduce_lr],  # 조기 종료와 학습률 감소 콜백
            verbose=0  # 훈련 과정을 자세히 출력하지 않음
        )
    
    return models  # 훈련된 모델들 반환

ensemble_models = create_ensemble_model()  # 앙상블 모델 생성 및 훈련

# 앙상블 예측 함수
def ensemble_predict(models, X):
    """
    여러 모델의 예측을 평균내어 앙상블 예측 수행
    - models: 훈련된 모델들의 리스트
    - X: 예측할 입력 데이터
    """
    predictions = []  # 각 모델의 예측 결과를 저장할 리스트
    for model in models:
        pred = model.predict(X, verbose=0)  # 각 모델로 예측 (출력 최소화)
        predictions.append(pred)  # 예측 결과를 리스트에 추가
    
    # 평균 투표: 모든 모델의 예측을 평균내어 최종 예측
    ensemble_pred = np.mean(predictions, axis=0)  # axis=0으로 모델별 평균 계산
    return ensemble_pred  # 앙상블 예측 결과 반환

ensemble_predictions = ensemble_predict(ensemble_models, x_test_poly)  # 테스트 데이터로 앙상블 예측
ensemble_accuracy = np.mean((ensemble_predictions > 0.5) == y_test.values.reshape(-1, 1))
# 0.5 기준으로 분류하여 정확도 계산

print(f"\n=== 앙상블 모델 성능 ===")  # 앙상블 성능 결과 출력
print(f"Ensemble Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
# 정확도를 소수점 4자리와 백분율로 출력

# 5. 각 모델에 대한 평가 지표
print(f'Sequential Model - Loss: {loss1}, Accuracy: {accuracy1}')
# 시퀀셜 모델의 손실과 정확도 출력
print(f'Function API Model - Loss: {loss2}, Accuracy: {accuracy2}')
# 함수형 API 모델의 손실과 정확도 출력
print(f'Subclassing API Model - Loss: {loss3}, Accuracy: {accuracy3}')
# 서브클래싱 API 모델의 손실과 정확도 출력

# 6. Loss와 Accuracy 시각화 (2x2 레이아웃으로 4개 모델 모두 시각화)
plt.figure(figsize=(16, 12))  # 전체 그래프 크기 설정 (가로 16, 세로 12)

# Sequential Model 시각화
plt.subplot(2, 2, 1)  # 2행 2열 중 첫 번째 서브플롯
plt.plot(history1.history['loss'], label='Training Loss', color='#FF6B6B', linewidth=2.5)  # 훈련 손실 그래프 (코랄 핑크)
plt.plot(history1.history['val_loss'], label='Validation Loss', color='#FF8E53', linestyle='--', linewidth=2.5)  # 검증 손실 그래프 (오렌지)
plt.plot(history1.history['accuracy'], label='Training Accuracy', color='#4ECDC4', linewidth=2.5)  # 훈련 정확도 그래프 (터키쉬 블루)
plt.plot(history1.history['val_accuracy'], label='Validation Accuracy', color='#45B7D1', linestyle='--', linewidth=2.5)  # 검증 정확도 그래프 (스카이 블루)
plt.title('Sequential Model', fontsize=14, fontweight='bold')  # 그래프 제목
plt.xlabel('Epochs')  # x축 라벨 (에포크)
plt.ylabel('Loss/Accuracy')  # y축 라벨 (손실/정확도)
plt.legend()  # 범례 표시
plt.grid(True, alpha=0.3)  # 격자 표시

# Functional API Model 시각화
plt.subplot(2, 2, 2)  # 2행 2열 중 두 번째 서브플롯
plt.plot(history2.history['loss'], label='Training Loss', color='#FF6B6B', linewidth=2.5)  # 훈련 손실 그래프 (코랄 핑크)
plt.plot(history2.history['val_loss'], label='Validation Loss', color='#FF8E53', linestyle='--', linewidth=2.5)  # 검증 손실 그래프 (오렌지)
plt.plot(history2.history['accuracy'], label='Training Accuracy', color='#4ECDC4', linewidth=2.5)  # 훈련 정확도 그래프 (터키쉬 블루)
plt.plot(history2.history['val_accuracy'], label='Validation Accuracy', color='#45B7D1', linestyle='--', linewidth=2.5)  # 검증 정확도 그래프 (스카이 블루)
plt.title('Functional API Model', fontsize=14, fontweight='bold')  # 그래프 제목
plt.xlabel('Epochs')  # x축 라벨
plt.ylabel('Loss/Accuracy')  # y축 라벨
plt.legend()  # 범례 표시
plt.grid(True, alpha=0.3)  # 격자 표시

# Subclassing API Model 시각화
plt.subplot(2, 2, 3)  # 2행 2열 중 세 번째 서브플롯
plt.plot(history3.history['loss'], label='Training Loss', color='#FF6B6B', linewidth=2.5)  # 훈련 손실 그래프 (코랄 핑크)
plt.plot(history3.history['val_loss'], label='Validation Loss', color='#FF8E53', linestyle='--', linewidth=2.5)  # 검증 손실 그래프 (오렌지)
plt.plot(history3.history['accuracy'], label='Training Accuracy', color='#4ECDC4', linewidth=2.5)  # 훈련 정확도 그래프 (터키쉬 블루)
plt.plot(history3.history['val_accuracy'], label='Validation Accuracy', color='#45B7D1', linestyle='--', linewidth=2.5)  # 검증 정확도 그래프 (스카이 블루)
plt.title('Subclassing API Model', fontsize=14, fontweight='bold')  # 그래프 제목
plt.xlabel('Epochs')  # x축 라벨
plt.ylabel('Loss/Accuracy')  # y축 라벨
plt.legend()  # 범례 표시
plt.grid(True, alpha=0.3)  # 격자 표시

# 앙상블 모델 성능 비교 시각화
plt.subplot(2, 2, 4)  # 2행 2열 중 네 번째 서브플롯
models_names = ['Sequential', 'Functional', 'Subclassing', 'Ensemble']  # 모델 이름들
accuracies = [accuracy1, accuracy2, accuracy3, ensemble_accuracy]  # 각 모델의 정확도
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']  # 각 모델별 색상 (코랄핑크, 터키쉬블루, 스카이블루, 민트그린)

bars = plt.bar(models_names, accuracies, color=colors, alpha=0.8, width=0.6)  # 막대 그래프 생성
plt.title('Test Accuracy each Models', fontsize=14, fontweight='bold')  # 그래프 제목
plt.xlabel('Models')  # x축 라벨
plt.ylabel('Test Accuracy')  # y축 라벨
plt.ylim(0, 1)  # y축 범위 설정 (0~1)
plt.grid(True, alpha=0.3, axis='y')  # y축에만 격자 표시

# 막대 위에 정확도 수치 표시
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()  # 서브플롯 간의 간격을 자동으로 조정
plt.show()  # 그래프 화면에 출력

# 7. 사용자 입력을 통한 예측 (가장 성능이 좋은 Subclassing 모델 사용)
print("\n=== 당뇨병 발병 예측 ===")  # 사용자 입력 섹션 시작 메시지
try:
    # 사용자로부터 건강 관련 정보 입력받기
    pregnancies = int(input("임신 횟수를 입력하세요: "))  # 임신 횟수
    glucose = float(input("포도당 수치를 입력하세요 (70-200): "))  # 포도당 부하 검사 수치
    blood_pressure = float(input("혈압을 입력하세요 (40-120): "))  # 혈압(mm Hg)
    skin_thickness = float(input("피하지방 두께를 입력하세요 (10-50): "))  # 팔 삼두근 뒤쪽의 피하지방 측정값(mm)
    insulin = float(input("인슐린 수치를 입력하세요 (15-850): "))  # 혈청 인슐린(mu U/ml)
    bmi = float(input("BMI를 입력하세요 (18-50): "))  # 체질량지수
    diabetes_pedigree = float(input("당뇨 내력 가중치를 입력하세요 (0.1-2.5): "))  # 당뇨 내력 가중치 값
    age = int(input("나이를 입력하세요: "))  # 나이
    
    # 입력값을 배열로 변환하고 정규화 적용
    user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])  # 2차원 배열로 변환
    user_input_scaled = scaler.transform(user_input)  # 표준화 적용 (훈련 시와 동일한 스케일)
    user_input_poly = poly.transform(user_input_scaled)  # 다항식 특성 적용
    
    # 예측 수행 - 서브클래싱 모델로 당뇨병 발병 확률 예측
    prediction = Subclass_model3.predict(user_input_poly)  # 모델로 예측 수행
    probability = prediction[0][0]  # 예측 결과 중 첫 번째 값 (확률) 추출
    
    # 결과 출력
    print(f"\n당뇨병 발병 확률: {probability:.2%}")  # 확률을 백분율로 출력 (소수점 2자리)
    if probability > 0.5:  # 확률이 50%보다 높으면
        print("예측 결과: 당뇨병 발병 위험이 높습니다")  # 당뇨병 발병 위험 높음 메시지
    else:  # 확률이 50% 이하면
        print("예측 결과: 당뇨병 발병 위험이 낮습니다")  # 당뇨병 발병 위험 낮음 메시지
        
except ValueError:  # 잘못된 형식의 입력일 때
    print("올바른 숫자를 입력해주세요.")  # 에러 메시지 출력
except Exception as e:  # 기타 예외 발생 시
    print(f"오류가 발생했습니다: {e}")  # 구체적인 에러 메시지 출력