from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

x_data = np.array([[1,2], [2,3], [3,4], [4,3], [3,2], [2,1]], dtype=np.float32)
y_data = np.array([[0], [0], [0], [1], [1], [1]], dtype=np.float32)  # XOR 문제

# 모델 정의 방법1: Sequential API
model = Sequential([
    Input(shape=(2,)),
    Dense(units=10, activation='relu'),
    Dense(units=1, activation='sigmoid')  # 이진 분류에서는
    # 출력 유닛이 1개
])
model.add(Dense(units=1, activation='sigmoid'))  # 이진 분류에서는 출력 유닛이 1개
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
model.summary()
model.fit(x_data, y_data, epochs=50, batch_size=1, verbose=0)

m_eval = model.evaluate(x_data, y_data, verbose=1)
print(f'평가 결과 : 손실={m_eval[0]:.4f}, 정확도={m_eval[1]:.4f}')


new_data = np.array([[1, 2.5], [10.5, 7.1]], dtype=np.float32)
pred = model.predict(new_data, verbose=0)
print('예측 확률 : ', pred.ravel())
print('예측 결과 : ', [1 if i>= 0.5 else 0 for i in pred])
print('예측 결과 : ', (pred >= 0.5).astype(int).ravel())
print('예측 결과 : ', np.where(pred >= 0.5, 1, 0).ravel())

