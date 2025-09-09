# 계층적 군집
# 10명의 학생의 시험 점수, 키, 몸무게 데이터를 이용해 계층적 군집화 수행

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

students = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']
# 시험 점수
scores = np.array([76, 95, 65, 85, 60, 92, 55, 88, 83, 72]).reshape(-1, 1)
print('점수:', scores.ravel())

# 계층적 군집
linked = linkage(scores, method='ward')

plt.figure(figsize=(10, 6))
dendrogram(linked, labels=students)
plt.axhline(y=25, color='r', linestyle='--', label='cut at height=25')
plt.xlabel('Students')
plt.ylabel('Distance')
plt.legend()
plt.grid(True)
# plt.show()
plt.close()

# 군집 3개로 나누기
clusters = fcluster(linked, 3, criterion='maxclust')
print('Clusters:', clusters)
for student, cluster in zip(students, clusters):
    print(f'{student}: Cluster {cluster}')

# 군집별로 점수와 이름 정리
cluster_info = {}
for student, cluster, score in zip(students, clusters, scores.ravel()):
    if cluster not in cluster_info:
        cluster_info[cluster] = {'students': [], 'scores': []}
    cluster_info[cluster]['students'].append(student)
    cluster_info[cluster]['scores'].append(score)

for cluster_id, info in sorted(cluster_info.items()):
    avg_score = np.mean(info['scores'])
    student_list = ', '.join(info['students'])
    print(f'Cluster {cluster_id} : 평균 점수 {avg_score:.2f}, 학생들: {student_list}')

# 군집 시각화
x_positions = np.arange(len(students))
y_scores = scores.ravel()
colors = {1: 'red', 2: 'blue', 3: 'green'}
plt.figure(figsize=(10, 6))

for i, (x, y, cluster) in enumerate(zip(x_positions, y_scores, clusters)):
    plt.scatter(x, y, color=colors[cluster], s=100)
    plt.text(x, y + 1, students[i], fontsize=10, ha='center')
plt.xticks(x_positions, students)
plt.xlabel('Students')
plt.ylabel('Scores')
plt.grid()
plt.show()