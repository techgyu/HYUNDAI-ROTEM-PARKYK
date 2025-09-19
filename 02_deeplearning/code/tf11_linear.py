# 선형 회귀 모델 - 모델 생성 3가지 보여줌 
# https://cafe.daum.net/flowlife/S2Ul/10  

import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Input   # 케라스의 가장 끝 레이어는 덴스다
from tensorflow.keras import optimizers 
import numpy as np 

# 공부시간에 따른 성적 데이터 사용 
x_data = np.array([1,2,3,4,5],dtype=np.float32).reshape(-1,1)  # (5,1)
# x_data=[[1],[2],[3],[4],[5]] 이렇게 써도 됨
y_data=np.array([11,32,53,64,70],dtype=np.float32).reshape(-1,1)   

# 모델 생성 방법 1 - Sequential API 사용 
# 모델 구성이 순차적[단순한] 경우에 사용
model=Sequential() # 계층 구조임(linear layer stack) 
model.add(Input((1,)))
model.add(Dense(units=16,activation='relu'))  # 중간층은 ReLU 히든은 렐루를써라, 그래야 mse의 손실값이 낮아짐
model.add(Dense(units=1,activation='linear'))  # 선형회귀니까 활성화 함수는 linear

print(model.summary()) # 구조 확인 가능 

opti=optimizers.SGD(learning_rate=0.001)  # 확률적 경사하강법  
model.compile(optimizer=opti,loss='mse',metrics=['mse'])  # 잔차에 대해서 평균을 취한값임. 
# 추측값 예측값에 대한 정확성을 ..할때 mse 는 0에 가까울수록 좋음  

# 학습진행할때 loss값 확인하고싶으면 히스토리
history=model.fit(x_data,y_data,epochs=100,batch_size=1,verbose=0) # 뉴럴네트워크는 로지스틱리그레션에서 많이 땄다
loss_metrics=model.evaluate(x=x_data,y=y_data,verbose=0)  # 학습/검증 데이터를 나누지 않았으니까. 학습 후 평가
print('loss_metrics:',loss_metrics)  # loss: [mse, mse]

# 성능확인해보기 
from sklearn.metrics import r2_score 
y_pred=model.predict(x_data)  # 예측값 
print('설명력:',r2_score(y_data,y_pred))
print('실제값:',y_data.ravel())
print('예측:',y_pred.ravel())

# 새 데이터(n,1)로 예측 
new_data=np.array([1.5,2.2,5.8],dtype=np.float32).reshape(-1,1)
new_pred=model.predict(new_data).ravel()
print('새 데이터 예측:',new_pred)

# 시각화 
import matplotlib.pyplot as plt 
plt.plot(x_data.ravel(),y_pred.ravel(),'b',label='pred')  # 예측값
plt.plot(x_data.ravel(),y_data.ravel(),'ko',label='true')  # 실제값
plt.legend()
plt.show()

# 모델 생성 방법 2 - functional API 사용  
# 유연한구조, 입력데이터로 여러 층을 공유, 다양한 종류의 입출력 가능 
# multi-input model , multi-output model , 공유층 활용 모델, 비순차적 데이터 처리... 
from tensorflow.keras import Model

# inputs=Input(shape=(1,)) # 입력층 
# outputs=Dense(units=1,activation='linear')(inputs) # 출력층 

# model2=Model(inputs,outputs)  

# 히든 레이어 적용
inputs=Input(shape=(1,)) # 입력층 
outputs1=Dense(units=16,activation='relu')(inputs) 
outputs2=Dense(units=1,activation='linear')(outputs1) # 이전층을 적어줌
model2=Model(inputs,outputs2) # 최종 인풋,아웃풋
# 이전층을 다음층 함수의 입력으로 사용하기 위해 변수에 할당 

opti2=optimizers.SGD(learning_rate=0.001)  # 확률적 경사하강법  
model2.compile(optimizer=opti2,loss='mse',metrics=['mse'])  # 잔차에 대해서 평균을 취한값임. 
# 추측값 예측값에 대한 정확성을 ..할때 mse 는 0에 가까울수록 좋음  

# 학습진행할때 loss값 확인하고싶으면 히스토리
history2=model2.fit(x_data,y_data,epochs=100,batch_size=1,verbose=0) # 뉴럴네트워크는 로지스틱리그레션에서 많이 땄다
loss_metrics=model2.evaluate(x=x_data,y=y_data,verbose=0)  # 학습/검증 데이터를 나누지 않았으니까. 학습 후 평가
print('loss_metrics:',loss_metrics)  # loss: [mse, mse]

# 성능확인해보기 
from sklearn.metrics import r2_score 
y_pred2=model2.predict(x_data)  # 예측값 
print('설명력:',r2_score(y_data,y_pred2))
print('실제값:',y_data.ravel())
print('예측:',y_pred2.ravel())

# https://cafe.daum.net/flowlife/S2Ul/68 
# 초반엔 샘플수 적게 시작하라. 

# 모델 생성 방법 3 - sub classing API 사용, 고난이도 작업에서 활용성 높음, 동적인 구조에 적합 
class MyModel(Model):
    def __init__(self):
        super().__init__() # 부모의 init호출
        self.d1=Dense(16,activation='relu')
        self.d2=Dense(1,activation='linear')
        
# x는 input 매개변수. functional api 와 유사하나 Input 객체를 사용하지 않음
# 계산작업등을 할 수 있다
# 이 call 메서드는, model.fit(),evaluate(),predict() 하면 자동 호출됨
    def call(self,x):  
        x=self.d1(x)
        return self.d2(x) #  이전층 담음

model3=MyModel() 

opti3=optimizers.SGD(learning_rate=0.001)  # 확률적 경사하강법  
model3.compile(optimizer=opti3,loss='mse',metrics=['mse'])  # 잔차에 대해서 평균을 취한값임. 
# 추측값 예측값에 대한 정확성을 ..할때 mse 는 0에 가까울수록 좋음  

# 학습진행할때 loss값 확인하고싶으면 히스토리
history3=model3.fit(x_data,y_data,epochs=100,batch_size=1,verbose=0) # 뉴럴네트워크는 로지스틱리그레션에서 많이 땄다
loss_metrics3=model3.evaluate(x=x_data,y=y_data,verbose=0)  # 학습/검증 데이터를 나누지 않았으니까. 학습 후 평가
print('loss_metrics:',loss_metrics3)  # loss: [mse, mse]

# 성능확인해보기 
from sklearn.metrics import r2_score 
y_pred3=model3.predict(x_data)  # 예측값 
print('설명력:',r2_score(y_data,y_pred3))
print('실제값:',y_data.ravel())
print('예측:',y_pred3.ravel()) 

# 모델 생성 방법 3-1) - sub classing API 사용, 레이어 클래스 노출하기
from tensorflow.keras.layers import Layer
# 사용자 정의 층 작성용 
# 기존 Dense 레이어와 동일한 기능을 하는 나만의 레이어를 직접 만드는 것입니다!
# 케라스의 layers 패키지에 정의된 레이어 대신
# , 새로운 연산을 하는 레이어 혹은 편의를위해 여러 레이어를 하나로 묶은 레이어를 구현할 때 사용 가능
class MyLinear(Layer):
    def __init__(self,units=1,**kwargs): # 아규먼트 받아야하는데 dict형태로 받아야함 **kwargs
        super(MyLinear,self).__init__(**kwargs) 
        
        #여기에 여러 레이어를 하나로 묶은 레이어를 구현
        self.units=units  # 출력 뉴런 수 
        
    # 내부적으로 call 호출, 모델의 가중치(w)관련 내용 기술
    def build(self,input_shape):  
        print('build',input_shape)
        self.w=self.add_weight(shape=(input_shape[-1],self.units)
            ,initializer='random_normal',trainable=True) # 입력 크기 모를때는 -1 
        # ,trainable=True 백 프로퍼게이션 지원하려면 이거 선언해야함
        self.b=self.add_weight(shape=(self.units,)
            ,initializer='zeros',trainable=True)
    
# 구분	하이퍼파라미터 |	백프로퍼게이션
# 주체	👨‍💻 사람이 설정	|🤖 컴퓨터가 자동 실행
# 시점	학습 전	|학습 중
# 대상	모델 구조, 학습 설정	|가중치(w), 편향(b)
# 목적	모델 성능 조정	|오차 최소화
# 변경	수동으로 튜닝	|자동으로 업데이트
        
# 정의된 값들을 이용해 해당층의 로직을 정의
    def call(self,inputs): 
        return tf.matmul(inputs,self.w)+self.b  # 선형회귀식

class MyMlp(Model):
    def __init__(self,**kwargs):
        super(MyMlp,self).__init__(**kwargs)
        self.linear1=MyLinear(2)  # 히든
        self.linear2=MyLinear(1)   # 아웃풋

    def call(self,inputs):
        x=self.linear1(inputs)
        x=tf.nn.relu(x)  # 활성화 함수
        return self.linear2(x) # 리니어1을 완성하고 리니어2로 넘김

model4=MyMlp()
opti4=optimizers.SGD(learning_rate=0.001)  # 확률적 경사하강법  
model4.compile(optimizer=opti4,loss='mse',metrics=['mse'])  # 잔차에 대해서 평균을 취한값임. 
# 추측값 예측값에 대한 정확성을 ..할때 mse 는 0에 가까울수록 좋음  

# 학습진행할때 loss값 확인하고싶으면 히스토리
history4=model4.fit(x_data,y_data,epochs=100,batch_size=1,verbose=0) # 뉴럴네트워크는 로지스틱리그레션에서 많이 땄다
loss_metrics4=model4.evaluate(x=x_data,y=y_data,verbose=0)  # 학습/검증 데이터를 나누지 않았으니까. 학습 후 평가
print('loss_metrics:',loss_metrics4)  # loss_metrics: [mse, mse]

# 성능확인해보기 
from sklearn.metrics import r2_score 
y_pred4=model4.predict(x_data)  # 예측값 
print('설명력:',r2_score(y_data,y_pred4))
print('실제값:',y_data.ravel())
print('예측:',y_pred4.ravel()) 