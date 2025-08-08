# matploblib은 floating 모듈
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import imread

plt.rc('font', family='Malgun Gothic')  # 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

x = ['서울', '인천', '수원'] # 리스트 / 튜플 가능
y = [5, 3, 7] # 리스트 / 튜플 가능

# plt.xlim([-1, 3]) # x축 범위 설정
# plt.ylim([0, 10]) # y축 범위 설정
# plt.plot(x, y)
# plt.yticks(list(range(0, 11, 1))) # y축 눈금 설정
# # plt.show()
# # jupyter notebook에서 실행 시, %matplotlib inline
# print('ok')

# data = np.arange(1, 11, 2) # 1부터 10까지의 홀수
# print(data)
# plt.plot(data)
# x = [0, 1, 2, 3, 4]
# for a, b in zip(x, data):
#     plt.text(a, b, str(b))  # 각 점 위에 값 표시
# plt.show()


# plt.plot(data)
# plt.plot(data, data, 'r')
# for a, b in zip(data, data):
#     plt.text(a, b, str(b))  # 각 점 위에 값 표시
# plt.show()


# sin 곡선
# x = np.arange(10)
# y = np.sin(x)
# print(x, y)
# plt.plot(x, y, 'bo')  # 'bo'는 파란색 원형 마커
# plt.plot(x, y, 'r+')  # 'r+'는 빨간색 플러스 마커
# plt.plot(x, y, 'go--', linewidth=2, markersize=12)  # 'go--'는 초록색 점선
# - (solid line), -- (dashed line), : (dotted line)
# c='b'  # 'b'는 파란색, color
# lw=2 # linewidth
# marker='o'  # 'o'는 원형 마커
# ms=12  # markersize
# plt.show()

# 홀드 명령: 하나의 영역에 여러 그래프를 그릴 때 사용
# plt.plot(x, y, 'bo')  # 첫 번째 그래프
# plt.hold(True)  # 홀드 명령
# x = np.arange(0, 3 * np.pi, 0.1)
# y_sin = np.sin(x)
# y_cos = np.cos(x)

# plt.figure(figsize=(10, 5))  # 그래프 크기 설정
# plt.plot(x, y_sin, 'r') # 직선
# plt.scatter(x, y_cos) # 점선
# plt.plot(x, y_cos)
# plt.xlabel('x축')
# plt.ylabel('y축')
# plt.title('제목')
# plt.legend(['sine', 'cosine'])  # 범례 추가
# plt.show()


# 서브 플롯 : figure 안에 여러 개의 그래프를 그릴 때 사용
# plt.subplot(2, 1, 1) # 2행 1열의 첫 번째 서브플롯
# plt.plot(x, y_sin, 'r') # 직선
# plt.title('사인')

# plt.subplot(2, 1, 2) # 2행 1열의 두 번째 서브플롯
# plt.plot(x, y_cos) # 직선
# plt.title('코사인')

# plt.show()

# print()
# name = ['a', 'b', 'c', 'd', 'e']
# kor_scores = [80, 50, 70, 70, 90]
# eng_scores = [60, 70, 80, 70, 60]
# plt.plot(name, kor_scores, 'ro-')
# plt.plot(name, eng_scores, 'gs-')
# plt.ylim([0, 100])  # y축 범위 설정
# plt.legend(['국어', '영어'], loc='best')  # 1, 2, 3, 4
# plt.grid(True) # 그리드 추가
# # 그래프 파일 저장 방법 -1
# # plt.savefig('plot1.png')  # 그래프를 파일로 저장
# # 그래프 파일 저장 방법 -2
# fig = plt.gcf() # 현재 Figure 객체 가져오기
# plt.show()  # 그래프 출력
# fig.savefig('plot1.png')  # 그래프를 파일로 저장

# img = imread('plot1.png')  # 이미지 파일 읽기
# plt.imshow(img)  # 이미지 표시
# plt.show()  # 이미지 출력

