# 1) 
# kaggle.com이 제공하는 'Red Wine quality' 분류 ( 0 - 10)

# Input variables (based on physicochemical tests):
#  1 - fixed acidity
#  2 - volatile acidity
#  3 - citric acid
#  4 - residual sugar
#  5 - chlorides
#  6 - free sulfur dioxide
#  7 - total sulfur dioxide
#  8 - density
#  9 - pH
#  10 - sulphates
#  11 - alcohol
#  Output variable (based on sensory data):
#  12 - quality (score between 0 and 10)
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

df = pd.read_csv('winequality-red.csv')
print(df.head(2))
print(df.shape, df.info())
print(df.describe())

model = RandomForestClassifier(criterion='entropy', n_estimators=500, random_state=1)

x = df.drop('quality', axis=1)
y = df['quality']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(criterion='entropy', n_estimators=500, random_state=1)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print('예측값 : ', y_pred)       
print('실제값 : ', np.array(y_test))

print('총 갯수:%d, 오류수:%d' % (len(y_test), (y_test != y_pred).sum()))

print('분류 정확도 : %.5f' % accuracy_score(y_test, y_pred))
print()

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

importances = model.feature_importances_
feature_names = x.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.grid(axis='x', linestyle='--')
plt.show()

pickle.dump(model, open('rf_model.sav', 'wb'))
del model 

read_model = pickle.load(open('rf_model.sav', 'rb'))

print('불러온 모델 정확도 : ', read_model.score(x_test, y_test))









