
# 상관관계 문제)

# https://github.com/pykwon/python 에 있는 Advertising.csv 파일을 읽어 tv,radio,newspaper 간의 상관관계를 파악하시오. 

# 그리고 이들의 관계를 heatmap 그래프로 표현하시오. 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("./01_big_data_machine_learning/data/Advertising.csv")
print(df)

df = df[['tv', 'radio', 'newspaper']]
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()