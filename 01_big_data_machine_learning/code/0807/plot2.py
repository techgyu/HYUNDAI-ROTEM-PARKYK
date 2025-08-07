import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = np.arange(10)

# # figure 구성 방법
# # 1) matplotlib 스타일의 인터페이스
# plt.figure()
# plt.subplot(2, 1, 1) # row, column, index
# plt.plot(x, np.sin(x))

# plt.subplot(2, 1, 2)
# plt.plot(x, np.cos(x))

# plt.show()

# # 2) 객체 지향 인터페이스
# fig, ax = plt.subplots(nrows = 2, ncols = 1) # 행, 열
# ax[0].plot(x, np.sin(x))
# ax[1].plot(x, np.cos(x))
# plt.show()

fig = plt.figure(figsize=(10, 5))  # 가로 10, 세로 5인치
ax1 = fig.add_subplot(1, 2, 1)  # 1행 2열의 첫 번째 서브플롯
ax2 = fig.add_subplot(1, 2, 2)  # 1행 2열의 두 번째 서브플롯
ax1.hist(np.random.randn(10), bins=10, alpha=0.9)
ax2.plot(np.random.randn(10))
# plt.show()

# bar
data = [50, 80, 100, 79, 90]
plt.bar(range(len(data)), data)
# plt.show()

# 오차 막대
loss = np.random.rand(len(data))
plt.barh(range(len(data)), data, xerr=loss, alpha=0.7)
# plt.show()

# pie
plt.pie(data, explode=(0, 0.1, 0, 0, 0), colors=['yellow', 'red', 'blue'])
# plt.show()

# boxplot : 사분위 등에 의한 데이터 분포, 확인에 대단히 효과적
plt.boxplot(data)
# plt.show()

# bouble chart
n = 30
np.random.seed(0)
x = np.random.rand(n)
y = np.random.rand(n)
color = np.random.rand(n)
scale = np.pi * (15 * np.random.rand(n)) ** 2
plt.scatter(x, y, c = color, s=scale)
plt.show()

# 시계열 데이터
fdata = pd.DataFrame(np.random.randn(1000, 4), index=pd.date_range('1/1/2000', periods=1000), columns=list('ABCD'))
fdata = fdata - fdata.cumsum()
print(fdata.head(3))
plt.plot(fdata['A'], label='A')
plt.show
# 판단스 자체 내장 시각화 도구
fdata.plot()
fdata.plot(kind='bar')
fdata.plot(kind='box')
plt.xlabel('Time')
plt.ylabel('data')
plt.show()

