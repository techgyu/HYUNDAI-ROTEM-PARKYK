# 미분계수 연습
# 평균변화율과 순간 변화율을 코드로 작성
# 서울에서 부산까지 자동차로 이동하는 경우 총거리 400km, 총 소요시간 4시간
# 중심 차분 공식으로 순간 속력을 근사
# 중심 차분: 양 옆 두 지점의 차이를 이용해 중간 지점의 순간 변화율을 추정

import numpy as np
import matplotlib.pyplot as plt
t = np.array([0, 1, 2, 3, 4], dtype=float) # 시간(단위: 시간)
s = np.array([0.0, 80.0, 180.0, 300.0, 400.0], dtype=float) # 거리(단위: km)

plt.plot(t, s)
plt.xlabel('t')
plt.ylabel('s')
plt.grid()
# plt.show()
plt.close()

# 전체 주행 거리
s_tot = s[-1]
s_half = s_tot / 2.0 # 중간 지점 = 200km

# 평균 변화율
# 평균 속도 = 전체거리 변화량 / 전체시간 변화량
t_tot = t[-1] - t[0] # 총 소요 시간

# 전체 평균 속도
v_avg = (s[-1] - s[0]) / t_tot

# 보간 함수 사용
t_mid = np.interp(s_half, s, t) # 200km에 해당하는 시간
print('t_mid =', t_mid) # 2.17: 실제 주행 곡선은 속도의 변화로 인해 더 늦게 200km에 도달

# 시간 간격의 중앙값을 구함 -> 평균적인 샘플 간격 계산
dt_mid = np.median(np.diff(t))
h = dt_mid * 0.5 # 중심 차분에 사용할 작은 간격 = 0.5h
s_plus = np.interp(t_mid + h, t, s) # t_mid + h에 해당하는 거리
s_minus = np.interp(t_mid - h, t, s) # t_mid - h에 해당하는 거리

# 중심 차분으로 순간 속도 추정
v_kph = (s_plus - s_minus) / (2.0 * h) # 단위: km/h

print(f'총 이동거리 = {s_tot:.1f}km, 총 소요시간 = {t_tot:.1f}h')
print(f'평균변화율(평균속력) = {v_avg:.1f}km/h')
print(f'중간지점을 지나는 시간 = {t_mid:.2f}h')
print(f's_plus = {s_plus:.1f}km, s_minus = {s_minus:.1f}km')
print(f'순간변화율(순간속도) = {v_kph:.1f}km/h') # 순간 속도 = 순간 변화율 = 접선의 기울기 = 미분계수





