from sklearn import datasets 
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap 

import pickle 

from sklearn.linear_model import LogisticRegression # 다중클래스지원 -anal_5/boost2_iris.py> 클래스==종속변수,label,y,class 

iris=datasets.load_iris() #데이터프레임아니고 배열이다
#print('\n :\n',iris['data'])  
print('\n 상관관계 :\n',np.corrcoef(iris.data[:,2],iris.data[:,3]))  
#  상관관계 :
#  [[1.         0.96286543]
#  [0.96286543 1.        ]] 
x=iris.data[:,[2,3]] # 모든 행의 2,3열 뽑음 petal.length , petal.width 만 참여 2차원 matrix
y=iris.target # 1차원 벡터

print('\n x[:,3] :\n',x[:3])
print('\n y[:,3], set(y) :\n',y[:3], set(y)) 
#  x[:,3] :
#  [[1.4 0.2]
#  [1.4 0.2]
#  [1.3 0.2]]

#  y[:,3], set(y) :
#  [0 0 0] {np.int64(0), np.int64(1), np.int64(2)} 

# 나눠서 작업하기
# train test split 7:3 
# 다중클래스를 지원하도록 일반화돼있다 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0) #shuffle true가 기본값임
print('\n x_train.shape,x_test.shape,y_train.shape,y_test.shape :\n',x_train.shape,x_test.shape,y_train.shape,y_test.shape)
#  x_train.shape,x_test.shape,y_train.shape,y_test.shape :
#  (105, 2) (45, 2) (105,) (45,) 

# 분류 모델 생성 
# C속성: L2규제 - 모델에 패널티 적용. 
# 숫자값을 조정해가며 분류 정확도를 확인한다. -> 1.0 10.0 100.0 ... 값이 작을수록 더 강한 정규화 규제를 지원한다 (튜닝 파라미터의 일종..)
# c값을 늘렸다 줄였다 함 

#region 여기 바꿔줌
# 0901
#model=LogisticRegression(C=0.1,random_state=0 , verbose=0 ) 
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier # xgboost 보다 성능 우수하나 테이터양이 적으면 과적합 발생
import lightgbm as lgb 

#model=DecisionTreeClassifier(criterion='entropy',max_depth=5,min_samples_leaf=5,random_state=0) # 선형에서 박스형식으로 바뀌어서 나온다.
#model=LGBMClassifier(n_estimators=500, random_state=42,verbose=-1) 
from sklearn.linear_model import LogisticRegression
from sklearn import svm, metrics 
model=svm.LinearSVC() 
print('\n model :\n',model)
#  model :
#  LogisticRegression(C=0.1, random_state=0)
#endregion 여기 바꿔줌

# 학습 진행하기
model.fit(x_train,y_train) # supervised learning 문제 x_train 주고 답지 y_train.. 사람이 개입함  
# 분류 예측  - 모델 성능 파악용
y_pred=model.predict(x_test)
print('\n y 예측값 :\n',y_pred)
print('\n y 실제값 :\n',y_test)
#  y 예측값 :
#  [2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0
#  2 1 1 2 0 2 0 0]
#  y 실제값 :
#  [2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0
#  1 1 1 2 0 2 0 0]

# 정확도 확인
print(f'\n 총갯수{len(y_test)},오류수:{(y_test!=y_pred).sum()}') # 총갯수45,오류수:1

#분류 정확도 총 3가지
print('\n 1 분류 정확도 확인 :\n',accuracy_score(y_test,y_pred)) # 실제, 예측
con_mat=pd.crosstab(y_test,y_pred,rownames=['예측값'],colnames=['관측값'])
print('\n 2 분류 정확도 확인 :\n',(con_mat[0][0]+con_mat[1][1]+con_mat[2][2])/len(y_test)) 

print(' 3 분류 정확도 확인 :') # 실제, 예측
print('\n test :\n',model.score(x_test,y_test))
print('\n train :\n',model.score(x_train,y_train)) # score 두개값차이가크면 과적합 의심
#  3 분류 정확도 확인 :
#  test :
#  0.9777777777777777
#  train :
#  0.9428571428571428

# 모델 저장 및 부르기
import pickle
#저장
pickle.dump(model,open('logimodel.sav','wb')) 
del model # 모델지움 

#읽기
readf_model=pickle.load(open('logimodel.sav','rb'))

# 새로운 값으로 예측: petal.length, petal.width 만 참여
print('\n x_test[:3] :\n',x_test[:3]) # 2 1 0으로 예측
#  x_test[:3] :
#  [[5.1 2.4]
#  [4.  1. ]
#  [1.4 0.2]]

new_data=np.array([[5.1,1.1],[1.1,1.1],[6.1,7.1]])
# 참고: 만약 표준화한 데이터로 모델을 생성했다면 .. 스케일맞추고 하는 거 
# sc.fit(new_data); new_data=sc.transform(new_data)
new_pred=readf_model.predict(new_data) # 내부적으로 softmax가 출력한 값을 argmax로 처리함 (아규먼트들 중에서 제일 큰값을 반환)
print('\n 예측 결과 :\n', new_pred)
#  예측 결과 :
#원래값
#print('\n 원래값 :\n', readf_model.predict_proba(new_data)) # 소프트맥스가 출력한값
#  원래값 :
#  [[3.05859126e-02 4.80004878e-01 4.89409209e-01] 4.89409209e-01 class2 확률이 제일 큼
#  [8.86247468e-01 1.10956949e-01 2.79558303e-03] 8.86247468e-01
#  [1.40841977e-05 4.79092000e-03 9.95194996e-01]]  9.95194996e-01

#  [2 0 2] 제일 큰 범주값을 뱉는거임 .첫번째에서 제일큰값이 2번째에서 제일큰게 0번째 3번째에서 제일큰게 2번째  

# predict_proba → 소프트맥스 확률 전체 출력
# predict → 소프트맥스 확률 중 가장 큰 클래스만 뽑아 최종 라벨로 반환

# X_train으로 스케일링 기준을 정한다 =
# "이 데이터의 평균과 표준편차를 기억해!"
# X_test는 그 기준으로만 변환한다 =
# "새로운 데이터를 봐도, 훈련 때 기준 그대로 바꿔!"

#region 스케일링

""" 
# 
print('----------------원본으로해보고 너무 단위가 들쭉날쭉이면 그때 처리해서 사용----------------')
# 독립변수에만 하는 과정임
# 스케일링 (데이터 표준화 - 최적화 과정에서 안정성,수렴 속도 향상, 오버플로우/언더플로우 방지 효과가 있다)
print(x_train[:3])
sc=StandardScaler()
sc.fit(x_train)
sc.fit(x_test) 
x_train=sc.transform(x_train) 
x_test=sc.transform(x_test) 
print('\n x_train[:3] :\n',x_train[:3])

# 스케일링 원복 
inver_x_train=sc.inverse_transform(x_train) 
print('\n x_train[:3] 원복 :\n',inver_x_train[:3])
#  x_train[:3] :
#  [[-0.05624622 -0.18650096]
#  [ 1.14902997  0.93250481]
#  [ 1.26955759  1.91163486]]

#  x_train[:3] 원복 :
#  [[3.5 1. ]
#  [5.5 1.8]
#  [5.7 2.5]] 
"""
#endregion 스케일링


#region 시각화
import matplotlib.pyplot as plt
#시각화
#한국어 패키지
plt.rc('font',family='malgun gothic') 
#음수깨지는거 처리
plt.rcParams['axes.unicode_minus']=False
def plot_decisionFunc(X, y, classifier, test_idx=None, resulution=0.02, title=''):
    # test_idx : test 샘플의 인덱스
    # resulution : 등고선 오차 간격
    markers = ('s','x','o','^','v')   # 마커(점) 모양 5개 정의함
    colors = ('r', 'b', 'lightgray', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])  # 색상팔레트를 이용
    # print(cmap.colors[0], cmap.colors[1])
    
    # surface(결정 경계) 만들기
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # 좌표 범위 지정
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # 격자 좌표 생성
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, resulution), \
                         np.arange(x2_min, x2_max, resulution))
    
    # xx, yy를 1차원배열로 만든 후 전치한다. 이어 분류기로 클래스 예측값 Z얻기
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)  # 원래 배열(격자 모양)로 복원

    # 배경을 클래스별 색으로 채운 등고선 그리기
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    X_test = X[test_idx, :]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], color=cmap(idx), \
                    marker=markers[idx], label=cl)
    if test_idx:
        X_test = X[test_idx, :]
        plt.scatter(x=X[:, 0], y=X[:, 1], color=[], \
                    marker='o', linewidths=1, s=80, label='test')
    plt.xlabel('꽃잎길이')
    plt.ylabel('꽃잎너비')
    plt.legend()
    plt.title(title)
    plt.show()

# train과 test 모두를 한 화면에 보여주기 위한 작업 진행
# train과 test 자료 수직 결합(위 아래로 이어 붙임 - 큰행렬 X 작성)
x_combined_std = np.vstack((x_train, x_test))   # feature
# 좌우로 이어 붙여 하나의 큰 레이블 벡터 y 만들기
y_combined = np.hstack((y_train, y_test))    # label
plot_decisionFunc(X=x_combined_std, y=y_combined, classifier=readf_model, \
                  test_idx = range(100, 150), title='scikit-learn 제공')
