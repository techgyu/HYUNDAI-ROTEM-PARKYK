# 가상의 데이터로 쇼핑몰 고객 세분화(집단화)
# DBSCAN 군집화 - 표준화 추천
# 고객수, 연간 지출액, 방문수 ...
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

np.random.seed(0)
n_customers = 200 # 고객 수
annual_spending = np.random.normal(50000, 15000, n_customers) # 연간 지출액
monthly_visits = np.random.normal(5, 2, n_customers) # 월 방문 횟수
print(annual_spending)
print(monthly_visits)
# numpy.clip() : 수치 안정화, 범위 고정 등에 사용
annual_spending = np.clip(annual_spending, 0, None) # 지출액 0 이상으로 제한
monthly_visits = np.clip(monthly_visits, 0, None) # 방문횟수 0 이상으로 제한
print(annual_spending[:5])
print(monthly_visits[:5])

data = pd.DataFrame({
    'annual_spending': annual_spending,
    'monthly_visits': monthly_visits
})
print(data.head(2))

# 표준화
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
print(data_scaled[:2])

dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(data_scaled)
data['cluster'] = clusters
print(data.head(3))

# 시각화
for cluster_id in np.unique(clusters):
    cluster_data = data[data['cluster'] == cluster_id]
    plt.scatter(cluster_data['annual_spending'], cluster_data['monthly_visits'], label=f'Cluster {cluster_id}')
    plt.xlabel('annual_spending')
    plt.ylabel('monthly_visits')
    plt.legend()
    plt.show()
    plt.close()

print(data['cluster'].value_counts())