# -----------------------------------------------------------------------------------------------
# 문제4) testdata/HR_comma_sep.csv 파일을 이용하여 salary를 예측하는 분류 모델을 작성한다.

# * 변수 종류 *
# satisfaction_level : 직무 만족도
# last_evaluation : 마지막 평가점수
# number_project : 진행 프로젝트 수
# average_monthly_hours : 월평균 근무시간
# time_spend_company : 근속년수
# work_accident : 사건사고 여부(0: 없음, 1: 있음)
# left : 이직 여부(0: 잔류, 1: 이직)
# promotion_last_5years: 최근 5년간 승진여부(0: 승진 x, 1: 승진)
# sales : 부서
# salary : 임금 수준 (low, medium, high)

# 조건 : Randomforest 클래스로 중요 변수를 찾고, Keras 지원 딥러닝 모델을 사용하시오.
# Randomforest 모델과 Keras 지원 모델을 작성한 후 분류 정확도를 비교하시오.
# -----------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
print("TensorFlow 버전:", tf.__version__)

# -----------------------------------------------------------------------------------------------

# 1. 데이터 로딩
data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/HR_comma_sep.csv')
print(data)
print(data.columns)
print(data.dtypes)
print(data['sales'].unique())
print(data['salary'].unique())

# -------------------------------------[데이터 전 처리]-------------------------------------

# 2. 데이터 탐색 및 전처리
df = data.copy()
le_salary = LabelEncoder()
le_sales = LabelEncoder()
df['salary'] = np.array(le_salary.fit_transform(df['salary'])) + 1  # low=1, medium=2, high=3
df['sales'] = np.array(le_sales.fit_transform(df['sales'])) + 1     # 각 부서별 1,2,3...
print(df)

# 3. Randomforest로 중요 변수 찾기
x = df.drop('salary', axis=1)  # 'salary'를 예측 대상으로 설정
y = df['salary']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f'RandomForest 분류 정확도: {acc:.4f}')
importances = rf.feature_importances_
feature_names = x.columns
importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
importance_df = importance_df.sort_values(by='importance', ascending=False)
print("\n[RandomForest 변수 중요도]")
print(importance_df)


# 3. 중요 변수 선택
x = x[['average_montly_hours', 'last_evaluation', 'satisfaction_level']]

# 4. 고급 특성 엔지니어링
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif

# 1) 로버스트 스케일링 (이상치에 강함)
robust_scaler = RobustScaler()
x_robust = robust_scaler.fit_transform(x)

# 2) 다항 특성 생성 (차수 증가)
poly = PolynomialFeatures(degree=3, include_bias=False, interaction_only=True)
x_poly = poly.fit_transform(x_robust)

# 3) 특성 선택 (가장 중요한 특성들만 선택)
selector = SelectKBest(score_func=f_classif, k=15)  # 상위 15개 특성만 선택
x_selected = selector.fit_transform(x_poly, df['salary'])

print(f"최종 특성 개수: {x_selected.shape[1]}")

# 4) Min-Max 정규화 추가
minmax_scaler = MinMaxScaler()
x_final = minmax_scaler.fit_transform(x_selected)

# -------------------------------------[모델 작성]-------------------------------------

# 1. Sequential/Functional/Subclassing 모델을 위한 데이터 준비
y_cat = to_categorical(df['salary'] - 1)  # 0부터 시작하는 인덱스로 변환
x_train, x_test, y_train, y_test = train_test_split(x_final, y_cat, test_size=0.3, random_state=42)
print(x_train.shape, y_train.shape)

# 향상된 콜백 정의
early_stop = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=8, factor=0.3, min_lr=1e-7, verbose=1)

# Sequential 모델 학습 및 기록 (더 깊은 네트워크)
model = models.Sequential()
model.add(layers.Input(shape=(x_train.shape[1],)))
model.add(layers.Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(layers.Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(layers.Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(layers.Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(layers.Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(layers.Dense(y_cat.shape[1], activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history1 = model.fit(
    x_train, y_train,
    epochs=300,
    batch_size=64,
    validation_split=0.2,
    verbose=0,
    callbacks=[early_stop, reduce_lr]
)
loss, acc = model.evaluate(x_test, y_test, verbose=0)



# 2. Functional API 모델 작성 (더 깊은 네트워크)
inputs = Input(shape=(x_train.shape[1],))
x_f = Dense(256, activation='relu')(inputs)
x_f = BatchNormalization()(x_f)
x_f = Dropout(0.4)(x_f)
x_f = Dense(128, activation='relu')(x_f)
x_f = BatchNormalization()(x_f)
x_f = Dropout(0.4)(x_f)
x_f = Dense(64, activation='relu')(x_f)
x_f = BatchNormalization()(x_f)
x_f = Dropout(0.3)(x_f)
x_f = Dense(32, activation='relu')(x_f)
x_f = BatchNormalization()(x_f)
x_f = Dropout(0.3)(x_f)
x_f = Dense(16, activation='relu')(x_f)
x_f = BatchNormalization()(x_f)
x_f = Dropout(0.2)(x_f)
outputs = Dense(y_cat.shape[1], activation='softmax')(x_f)
functional_model = Model(inputs=inputs, outputs=outputs)

# Functional API 모델 학습 및 기록
functional_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history2 = functional_model.fit(
    x_train, y_train,
    epochs=300,
    batch_size=64,
    validation_split=0.2,
    verbose=0,
    callbacks=[early_stop, reduce_lr]
)
func_loss, func_acc = functional_model.evaluate(x_test, y_test, verbose=0)



# 3. Subclassing API 모델 작성 (더 깊은 네트워크)
class MyModel(Model):
    def __init__(self):
        super().__init__()
        self.d1 = Dense(256, activation='relu')
        self.bn1 = BatchNormalization()
        self.dp1 = Dropout(0.4)
        self.d2 = Dense(128, activation='relu')
        self.bn2 = BatchNormalization()
        self.dp2 = Dropout(0.4)
        self.d3 = Dense(64, activation='relu')
        self.bn3 = BatchNormalization()
        self.dp3 = Dropout(0.3)
        self.d4 = Dense(32, activation='relu')
        self.bn4 = BatchNormalization()
        self.dp4 = Dropout(0.3)
        self.d5 = Dense(16, activation='relu')
        self.bn5 = BatchNormalization()
        self.dp5 = Dropout(0.2)
        self.out = Dense(3, activation='softmax')  # 3개 클래스 (low, medium, high)

    def call(self, x):
        x = self.d1(x)
        x = self.bn1(x)
        x = self.dp1(x)
        x = self.d2(x)
        x = self.bn2(x)
        x = self.dp2(x)
        x = self.d3(x)
        x = self.bn3(x)
        x = self.dp3(x)
        x = self.d4(x)
        x = self.bn4(x)
        x = self.dp4(x)
        x = self.d5(x)
        x = self.bn5(x)
        x = self.dp5(x)
        return self.out(x)

# Subclassing API 모델 학습 및 기록
subclass_model = MyModel()
subclass_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history3 = subclass_model.fit(
    x_train, y_train,
    epochs=300,
    batch_size=64,
    validation_split=0.2,
    verbose=0,
    callbacks=[early_stop, reduce_lr]
)
sub_loss, sub_acc = subclass_model.evaluate(x_test, y_test, verbose=0)

print(f'Sequential 딥러닝 모델 분류 정확도: {acc:.4f}')
print(f"Functional API 딥러닝 모델 분류 정확도: {func_acc:.4f}")
print(f"Subclassing 딥러닝 모델 분류 정확도: {sub_acc:.4f}")

# -------------------------------------[시각화]-------------------------------------
plt.figure(figsize=(15, 5))  # 전체 그래프 크기 설정 (가로 15, 세로 5)

# 1. Sequential Model 시각화
plt.subplot(1, 3, 1)  # 1행 3열 중 첫 번째 서브플롯
plt.plot(history1.history['loss'], label='Training Loss')
plt.plot(history1.history['val_loss'], label='Validation Loss')
plt.plot(history1.history['accuracy'], label='Training Accuracy')
plt.plot(history1.history['val_accuracy'], label='Validation Accuracy')
plt.title('Sequential Model')
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.grid(True)

# 2. Functional API Model 시각화
plt.subplot(1, 3, 2)  # 1행 3열 중 두 번째 서브플롯
plt.plot(history2.history['loss'], label='Training Loss')
plt.plot(history2.history['val_loss'], label='Validation Loss')
plt.plot(history2.history['accuracy'], label='Training Accuracy')
plt.plot(history2.history['val_accuracy'], label='Validation Accuracy')
plt.title('Functional API Model')
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.grid(True)

# 3. Subclassing API Model 시각화
plt.subplot(1, 3, 3)  # 1행 3열 중 세 번째 서브플롯
plt.plot(history3.history['loss'], label='Training Loss')
plt.plot(history3.history['val_loss'], label='Validation Loss')
plt.plot(history3.history['accuracy'], label='Training Accuracy')
plt.plot(history3.history['val_accuracy'], label='Validation Accuracy')
plt.title('Subclassing API Model')
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()  # 서브플롯 간의 간격을 자동으로 조정
plt.show()  # 그래프 화면에 출력
plt.close()  # 그래프 창 닫기