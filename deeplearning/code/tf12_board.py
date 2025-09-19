# 다중선형회귀모델 + 텐서보드
import tensorboard
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Input   # 케라스의 가장 끝 레이어는 덴스다
from tensorflow.keras import optimizers 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard

# 작년의 5명의 3회 모의고사&본 고사 점수 데이터로 학습후
# 새 데이터로 점수 예측
x_data = np.array([[70, 85, 80], [71, 89, 78], [50, 80, 60], [66, 30, 60], [50, 25, 10]])
y_data = np.array([[73, 82, 72, 57, 34]]).T

print('1) Sequential api ---')
model = Sequential()
model.add(Input(shape=(3,)))
model.add(Dense(units = 8, activation='relu', name ='a'))
model.add(Dense(units = 4, activation='relu', name ='b'))
model.add(Dense(units = 1, activation='linear', name ='c'))
print(model.summary())

opti = optimizers.Adam(learning_rate=0.1)
model.compile(optimizer=opti, loss='mse', metrics=['mse'])
history = model.fit(x_data, y_data, batch_size=1, epochs=50, verbose=2)

plt.plot(history.history['loss'], label='loss')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
# plt.show()
plt.savefig('tf12_loss.png')

loss_metrics = model.evaluate(x=x_data, y=y_data)
print('loss_metrics : ', loss_metrics)
print('결정계수:', r2_score(y_data, model.predict(x_data)))

# -------------------------------------------------------------------------------

tf.keras.backend.clear_session() # 메모리 해제

# 다중선형회귀모델 + 텐서보드
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Input   # 케라스의 가장 끝 레이어는 덴스다
from tensorflow.keras import optimizers 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
import shutil, os, datetime as dt
from sklearn.metrics import r2_score

np.random.seed(42)
tf.random.set_seed(42)

# 작년의 5명의 3회 모의고사&본 고사 점수 데이터로 학습후
# 새 데이터로 점수 예측
x_data = np.array([[70, 85, 80], [71, 89, 78], [50, 80, 60], [66, 30, 60], [50, 25, 10]])
y_data = np.array([[73, 82, 72, 57, 34]]).T
inputs = Input(shape=(3,))
h1 = Dense(units = 8, activation='relu', name ='a')(inputs)
h2 = Dense(units = 4, activation='relu', name ='b')(h1)
outputs = Dense(units = 1, activation='linear', name ='c')(h2)

model = Model(inputs=inputs, outputs=outputs, name='linear_model')

# TensorBoard ------------------------------------------------------------
BASE = "logs" # 기본 로그 저장 디렉토리명
shutil.rmtree(BASE, ignore_errors=True) # 해당 디렉토리 삭제
RUN = os.path.join(BASE, 'test')
os.makedirs(RUN, exist_ok=True) # 디렉토리 생성

tb = TensorBoard(log_dir=RUN, histogram_freq=1, write_graph=True)
# ------------------------------------------------------------------------

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit(x_data, y_data, batch_size=1, epochs=50, verbose=2, callbacks=[tb])


loss_metrics = model.evaluate(x=x_data, y=y_data)
print('loss_metrics : ', loss_metrics)

print('결정계수:', r2_score(y_data, model.predict(x_data)))

import subprocess
subprocess.run(['tensorboard', '--logdir', RUN])

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='', show_shape=True, show_layer_names=True)