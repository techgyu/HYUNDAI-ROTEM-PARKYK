# 모델 정의 방법2: Functional API
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import numpy as np

x_data = np.array([[1,2], [2,3], [3,4], [4,3], [3,2], [2,1]], dtype=np.float32)
y_data = np.array([[0], [0], [0], [1], [1], [1]], dtype=np.float32)  # XOR 문제

input_layer = Input(shape=(2,))
output_layer = Dense(1, activation='sigmoid')(input_layer)

model2 = Model(inputs=input_layer, outputs=output_layer)
model2.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
model2.summary()

model2.fit(x_data, y_data, epochs=50, batch_size=1, verbose=0)
m_eval = model2.evaluate(x_data, y_data, verbose=0)

new_data = np.array([[1, 2.5], [10.5, 7.1]], dtype=np.float32)
pred2 = model2.predict(new_data, verbose=0)
print('예측 확률 : ', pred2.ravel())
print('예측 결과 : ', [1 if i>= 0.5 else 0 for i in pred2])
print('예측 결과 : ', (pred2 >= 0.5).astype(int).ravel())
print('예측 결과 : ', np.where(pred2 >= 0.5, 1, 0).ravel())
