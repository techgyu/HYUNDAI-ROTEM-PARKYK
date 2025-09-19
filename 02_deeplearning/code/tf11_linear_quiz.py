import tensorflow as tf # AI 모델 만드는 메인 도구
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input # 뉴럴네트워크  층들
from tensorflow.keras import optimizers # 모델과 최적화기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler # 성능 측정 도구
from tensorflow.keras.layers import Input
from tensorflow.keras import Model

# 문제1) 아버지 키로 아들 키 예측하는 회귀분석 모델
# https://cafe.daum.net/flowlife/S2Ul/25 
# data를 이용해 아버지 키로 아들의 키를 예측하는 회귀분석 모델을 작성하시오.
#  - train / test 분리
#  - Sequential api와 function api 를 사용해 모델을 만들어 보시오.
#  - train과 test의 mse를 시각화 하시오
#  - 새로운 아버지 키에 대한 자료로 아들의 키를 예측하시오.

# 1. 데이터 로드
url = "https://github.com/data-8/materials-fa17/raw/master/lec/galton.csv"

data = pd.read_csv(url)
# print(data.head())
# print(f"데이터 크기: {data.shape}")
# print(f"컬럼: {data.columns}")

# 2. 데이터 전처리
# 남성 자녀(아들)만 필터링
male_children = data[data['gender'] == 'male']
# print(f"전체 데이터: {len(data)}, 남성 자녀: {len(male_children)}")

X = male_children['father'].values.reshape(-1, 1)     # 아버지 키 (입력)
y = male_children['childHeight'].values.reshape(-1, 1) # 아들 키 (출력)

# print(f"입력 데이터 형태: {X.shape}")
# print(f"출력 데이터 형태: {y.shape}")

# 3. Train/Test 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# print(f"훈련 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")

# 4. 데이터 시각화
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, alpha=0.6, label='Train Data')
plt.scatter(X_test, y_test, alpha=0.6, label='Test Data', color='green')
plt.xlabel('Father Height (inches)')
plt.ylabel('Son Height (inches)')
plt.title('Galton Data: Father vs Son Height')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show() 

# AI 모델 2가지 생성
# Sequential API와 Functional API 모델 생성

# 5. Sequential API 모델
print("\n🤖 Sequential API 모델 생성")
model_seq = Sequential([
    Dense(16, activation='relu'),  # 층1: 16개 뉴런
    Dense(8, activation='relu'),   # 층2: 8개 뉴런  
    Dense(1, activation='linear')  # 층3: 1개 뉴런 (키 예측)
])
model_seq.compile(
    optimizer=optimizers.Adam(learning_rate=0.01),
    loss='mse',
    metrics=['mae']
)

# 6. Functional API 모델
print("\n🔧 Functional API 모델 생성")


inputs = Input(shape=(1,))
x = Dense(16, activation='relu')(inputs)
x = Dense(8, activation='relu')(x)
outputs = Dense(1, activation='linear')(x)

model_func = Model(inputs=inputs, outputs=outputs)
model_func.compile(
    optimizer=optimizers.Adam(learning_rate=0.01),
    loss='mse',
    metrics=['mae']
)

# 7. 두 모델 학습
print("\n🔥 Sequential 모델 학습...")
# validation_split=0.2: 훈련 데이터의 20%를 검증용으로 사용
history_seq = model_seq.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)

print("🔥 Functional 모델 학습...")
history_func = model_func.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)

# 8. Train과 Test MSE 시각화
plt.figure(figsize=(15, 5))

# Sequential 모델 결과
plt.subplot(1, 3, 1)
plt.plot(history_seq.history['loss'], label='Train Loss')
plt.plot(history_seq.history['val_loss'], label='Val Loss')
plt.title('Sequential API - Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True, alpha=0.3)

# Functional 모델 결과
# loss 그래프: 학습이 진행될수록 오차가 줄어드는 모습
plt.subplot(1, 3, 2)
plt.plot(history_func.history['loss'], label='Train Loss')
plt.plot(history_func.history['val_loss'], label='Val Loss')
plt.title('Functional API - Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True, alpha=0.3)

# 예측 결과 비교
plt.subplot(1, 3, 3)
y_pred_seq = model_seq.predict(X_test)
y_pred_func = model_func.predict(X_test)

plt.scatter(y_test, y_pred_seq, alpha=0.6, label='Sequential', s=30)
plt.scatter(y_test, y_pred_func, alpha=0.6, label='Functional', s=30)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('real height')
plt.ylabel('predicted height')
plt.title('Predicted vs Real')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 9. 모델 성능 비교
seq_r2 = r2_score(y_test, y_pred_seq)
func_r2 = r2_score(y_test, y_pred_func)

print(f"\n📊 모델 성능 비교:")
print(f"Sequential API R² 스코어: {seq_r2:.4f}")
print(f"Functional API R² 스코어: {func_r2:.4f}")

# 10. 새로운 아버지 키로 아들 키 예측
print("\n🎯 새로운 아버지 키로 아들 키 예측:")
new_father_heights = np.array([70, 72, 68, 75]).reshape(-1, 1)

pred_seq = model_seq.predict(new_father_heights)
pred_func = model_func.predict(new_father_heights)

for i, father_height in enumerate(new_father_heights.flatten()):
    print(f"아버지 키 {father_height} inches:")
    print(f"  Sequential 예측: {pred_seq[i][0]:.2f} inches")
    print(f"  Functional 예측: {pred_func[i][0]:.2f} inches")

print("\n✅ 갤튼 데이터 분석 완료!")

# 문제2) 자전거 공유 시스템 분석 - 다중선형회귀분석
# https://raw.githubusercontent.com/pykwon/python/refs/heads/master/data/train.csv
# 자전거 공유 시스템 분석용 데이터 train.csv를 이용하여 대여횟수에 영향을 주는 변수들을 골라 다중선형회귀분석 모델을 작성하시오.
# 모델 학습시에 발생하는 loss를 시각화하고 설명력을 출력하시오.
# 새로운 데이터를 input 함수를 사용해 키보드로 입력하여 대여횟수 예측결과를 콘솔로 출력하시오.

# 1. 데이터 로드
print("🚴‍♂️ 자전거 공유 시스템 데이터 분석")
url = "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/data/train.csv"

bike_data = pd.read_csv(url)
# print(f"데이터 크기: {bike_data.shape}")
# print(f"컬럼: {bike_data.columns}")
# print(bike_data.head())

# 2. 데이터 탐색 및 전처리
print("\n📊 데이터 기본 정보:")
print(bike_data.describe())

# 상관관계 분석
# 상관관계: 어떤 변수가 대여횟수에 가장 큰 영향을 주는지 분석
# 특성 선택: 온도, 체감온도, 습도, 풍속, 계절, 날씨, 근무일

# 숫자 컬럼만 선택해서 상관관계 계산 (날짜/시간 컬럼 제외)
numeric_columns = bike_data.select_dtypes(include=[np.number])
correlation = numeric_columns.corr()

print("\n🔍 대여횟수(count)와 다른 변수들의 상관관계:")
if 'count' in correlation.columns:
    count_corr = correlation['count'].sort_values(ascending=False)
    print(count_corr)
else:
    print("⚠️ 'count' 컬럼을 찾을 수 없습니다. 사용 가능한 숫자 컬럼들:")
    print(numeric_columns.columns.tolist())
    # 기본 특성 사용
    available_features = [col for col in ['temp', 'atemp', 'humidity', 'windspeed', 'season', 'weather', 'workingday'] 
                         if col in numeric_columns.columns]
    print(f"사용 가능한 특성: {available_features}")

# 대여횟수에 영향을 주는 주요 변수 선택 (상관계수 절댓값 기준)
# 실제 데이터에 존재하는 컬럼만 선택
all_possible_features = ['temp', 'atemp', 'humidity', 'windspeed', 'season', 'weather', 'workingday', 'casual', 'registered']
important_features = [col for col in all_possible_features if col in bike_data.columns]

# count 컬럼이 없으면 대체 타겟 찾기
target_col = 'count'
if 'count' not in bike_data.columns:
    possible_targets = ['cnt', 'total', 'demand']
    for col in possible_targets:
        if col in bike_data.columns:
            target_col = col
            break
    if target_col == 'count':  # 여전히 찾지 못했다면
        # 숫자 컬럼 중 마지막 컬럼을 타겟으로 사용
        numeric_cols = bike_data.select_dtypes(include=[np.number]).columns.tolist()
        target_col = numeric_cols[-1] if numeric_cols else 'count'

print(f"\n🎯 선택된 특성: {important_features}")
print(f"🎯 타겟 변수: {target_col}")

# 특성과 타겟 분리
X = bike_data[important_features]
y = bike_data[target_col]

print(f"입력 특성 형태: {X.shape}")
print(f"타겟 형태: {y.shape}")

# 3. 데이터 정규화
# 정규화: 모든 변수를 같은 스케일로 맞추기
# 이유: 온도(0-41)와 습도(0-100)의 단위가 다르니까
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test 분리
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"훈련 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")

# 4. Sequential API 모델 생성

print("\n🤖 Sequential API 모델 생성...")
# 다중 입력: 7개 변수(온도, 습도, 계절 등)를 동시에 고려
# 복잡한 모델: 여러 층으로 복잡한 패턴 학습
model_seq = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')  # 회귀이므로 linear
])

model_seq.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# 5. 모델 학습 (history 저장)
print("🔥 모델 학습 시작...")
history = model_seq.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# 6. Loss 시각화
# 2개 그래프: Loss(오차) 변화, MAE(평균절대오차) 변화
# 과적합 확인: 훈련/검증 손실이 같이 떨어지는지 확인

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE During Training')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 7. 모델 평가
train_pred = model_seq.predict(X_train)
test_pred = model_seq.predict(X_test)

train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

print(f"\n📈 모델 성능:")
print(f"훈련 데이터 R² 스코어: {train_r2:.4f}")
print(f"테스트 데이터 R² 스코어: {test_r2:.4f}")
print(f"훈련 데이터 설명력: {train_r2*100:.2f}%")
print(f"테스트 데이터 설명력: {test_r2*100:.2f}%")

# 8. 새로운 데이터 예측 (키보드 입력)
print("\n🎮 새로운 데이터로 대여횟수 예측하기!")
print("다음 정보를 입력해주세요:")

# 실제 사용된 특성 개수만큼 입력받기
print(f"📝 입력할 특성: {important_features}")
print(f"📊 총 {len(important_features)}개 특성")

new_values = []
feature_descriptions = {
    'temp': '온도 (0-41)',
    'atemp': '체감온도 (0-50)', 
    'humidity': '습도 (0-100)',
    'windspeed': '풍속 (0-67)',
    'season': '계절 (1:봄, 2:여름, 3:가을, 4:겨울)',
    'weather': '날씨 (1:맑음, 2:흐림, 3:비/눈)',
    'workingday': '근무일 여부 (0:아니오, 1:예)',
    'casual': '비회원 대여수 (예: 50)',
    'registered': '회원 대여수 (예: 200)',
    'holiday': '휴일 여부 (0:아니오, 1:예)'
}

for feature in important_features:
    description = feature_descriptions.get(feature, f'{feature} 값')
    if feature in ['season', 'weather', 'workingday', 'holiday']:
        value = int(input(f"{description}: "))
    else:
        value = float(input(f"{description}: "))
    new_values.append(value)

# 새 데이터 전처리 (정확한 특성 개수로)
new_data = np.array([new_values])
print(f"🔍 입력된 데이터 형태: {new_data.shape}")
print(f"🔍 예상되는 특성 개수: {len(important_features)}")

new_data_scaled = scaler.transform(new_data)

# 예측
prediction = model_seq.predict(new_data_scaled)

print(f"\n🎯 예측 결과:")
print(f"예상 자전거 대여횟수: {prediction[0][0]:.0f} 대")
    
print("\n✅ 자전거 공유 시스템 분석 완료!")