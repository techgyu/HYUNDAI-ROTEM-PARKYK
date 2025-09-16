# 모델 정의 방법3: Subclassing API
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import numpy as np

x_data = np.array([[1,2], [2,3], [3,4], [4,3], [3,2], [2,1]], dtype=np.float32)
y_data = np.array([[0], [0], [0], [1], [1], [1]], dtype=np.float32)  # XOR 문제

class MyBinaryClass(Model):
    def __init__(self):
        super().__init__(name='MyBinaryClass')
        self.dense = Dense(units=1, activation='sigmoid', name='dense_sigmoid')

    def build(self, input_shape):
        # 첫번째 순 방향 학습(feed forward)시점에 자동 호출되어 가중치 생성
        super().build(input_shape)

    # fit(), evaluate(), predict() 메서드가 호출될 때 자동 호출
    def call(self, inputs, training=False):
        print('>>> call() 실행됨, training=', training)
        return self.dense(inputs)
        
model3 = MyBinaryClass()
model3.build(input_shape=(None, 2))  # 입력 데이터의 크기 지정
model3.summary()

model3.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

model3.fit(x_data, y_data, epochs=50, batch_size=1, verbose=0)
m_eval = model3.evaluate(x_data, y_data, verbose=0)

new_data = np.array([[1, 2.5], [10.5, 7.1]], dtype=np.float32)
pred3 = model3.predict(new_data, verbose=0)
print('예측 결과 : ', np.where(pred3 >= 0.5, 1, 0).ravel())
