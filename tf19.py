# vaidation split 방식과 KFold 방식의 차이

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

data = np.loadtxt('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/pima-indians-diabetes.data.csv', delimiter=',', dtype=np.float32)
x = data[:, :-1] # 마지막 컬럼 제외
y = data[:, -1] # 마지막 컬럼 (타겟)

print(x[:3])
print(y[:3])

def build_model():
    model = Sequential([
        Input(shape=(8,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model_val = build_model()
# validation split 방식
history_val = model_val.fit(x, y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
val_acc = history_val.history['val_accuracy'][-1]   

# k-fold 방식
kf = KFold(n_splits=5, shuffle=True, random_state=42)
print(kf)
print(f"\n전체 데이터 크기: {len(x)}")
print(f"각 폴드당 검증 데이터 크기: 약 {len(x)//5}")
print(f"각 폴드당 훈련 데이터 크기: 약 {len(x) - len(x)//5}")

kfold_accuracies = []
for fold_num, (train_idx, val_idx) in enumerate(kf.split(x), 1):
    print(f"\n=== Fold {fold_num} ===")
    print(f"훈련 인덱스: {train_idx[:10]}...{train_idx[-10:]} (총 {len(train_idx)}개)")
    print(f"검증 인덱스: {val_idx[:10]}...{val_idx[-10:]} (총 {len(val_idx)}개)")
    
    x_train, x_val = x[train_idx], x[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    print(f"훈련 데이터 형태: {x_train.shape}, 검증 데이터 형태: {x_val.shape}")
    print(f"훈련 타겟 분포: 0={np.sum(y_train==0)}, 1={np.sum(y_train==1)}")
    print(f"검증 타겟 분포: 0={np.sum(y_val==0)}, 1={np.sum(y_val==1)}")

    model_kf = build_model()
    model_kf.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0)

    y_pred = model_kf.predict(x_val, verbose=0)
    y_pred_label = (y_pred > 0.5).astype(int)   
    acc = accuracy_score(y_val, y_pred_label)
    kfold_accuracies.append(acc)
    print(f"Fold {fold_num} 정확도: {acc:.4f}")

# 비교 출력
print(f'Validation Split Accuracy: {val_acc:.4f}')
print(f'K-Fold Cross-Validation Accuracies: {np.round(kfold_accuracies, 4)}')
print(f'K-Fold Cross-Validation Mean Accuracy: {np.mean(kfold_accuracies):.4f}')