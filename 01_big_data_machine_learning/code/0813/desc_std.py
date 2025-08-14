# 표준편차, 분산은 중요하다.
# 2개 반의 시험 성적이 다를 때, 그 차이를 수치적으로 나타내기 위해 사용된다.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')  # 한글 폰트 설정

np.random.seed(42)  # 시드 넘버 고정

target_mean = 60  # 목표 평균
std_dev_small = 10  # 표준편차 최소 값
std_dev_large = 20  # 표준편차 최대 값

class1_raw = np.random.normal(loc=target_mean, scale=std_dev_small, size=100)  # 1반 raw 데이터
class2_raw = np.random.normal(loc=target_mean, scale=std_dev_large, size=100)  # 2반 raw 데이터

class1_adj = class1_raw - np.mean(class1_raw) + target_mean  # 1반 평균 보정
class2_adj = class2_raw - np.mean(class2_raw) + target_mean  # 2반 평균 보정

class1 = np.clip(np.round(class1_adj), 10, 100).astype(int)  # 1반 정수화 및 범위 제한
class2 = np.clip(np.round(class2_adj), 10, 100).astype(int)  # 2반 정수화 및 범위 제한

print("데이터 1차 가공 결과")
print("class1: \n", class1)
print("class2: \n", class2)

mean1, mean2 = np.mean(class1), np.mean(class2)  # 평균
std1, std2 = np.std(class1), np.std(class2)  # 표준편차
var1, var2 = np.var(class1), np.var(class2)  # 분산

print("1반 성적: ", class1)
print("평균 = {:.2f}, 표준편차 = {:.2f}, 분산 = {:.2f}".format(mean1, std1, var1))
print("2반 성적: ", class2)
print("평균 = {:.2f}, 표준편차 = {:.2f}, 분산 = {:.2f}".format(mean2, std2, var2))

df = pd.DataFrame({
    'Class': ['1반'] * 100 + ['2반'] * 100,
    'Score': np.concatenate([class1, class2])
})  # 데이터프레임 생성

print(df)

df.to_csv('./01_big_data_machine_learning/data/desc_std1.csv', index=False, encoding='utf-8')  # CSV 저장

x1 = np.random.normal(1, 0.05, size=100)  # 1반 x좌표
x2 = np.random.normal(2, 0.05, size=100)  # 2반 x좌표
plt.scatter(x1, class1, label=f'1반 (평균={mean1:.2f}), σ={std1:.2f}')  # 1반 산포도
plt.scatter(x2, class2, label=f'2반 (평균={mean2:.2f}), σ={std2:.2f}')  # 2반 산포도
plt.hlines(target_mean, 0.5, 2.5, colors='red', linestyles='dashed', label=f'공통평균={target_mean}')  # 평균선
plt.title('동일 평균, 다른 성적 분포를 가진 두 반 비교')  # 그래프 제목
plt.xticks([1, 2], ['1반', '2반'])  # x축 라벨
plt.ylabel('시험 점수')  # y축 라벨
plt.legend()  # 범례
plt.grid(True)  # 격자
plt.tight_layout()  # 레이아웃 조정
plt.show()  # 그래프 출력

plt.figure(figsize=(8, 5))  # 박스플롯 크기 설정
plt.boxplot([class1, class2], label=['1반', '2반'])  # 박스플롯
plt.title('성적 분포를 가진 두 반 비교')  # 그래프 제목
plt.ylabel('시험 점수')  # y축 라벨
plt.grid(True)  # 격자
plt.tight_layout()  # 레이아웃 조정
plt.show()  # 그래프 출력
plt.close()  # 플롯 종료

