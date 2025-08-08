# 아웃라이어(이상치) 처리와 박스플롯(Boxplot)

박스플롯은 데이터의 분포와 이상치(outlier)를 시각적으로 쉽게 파악할 수 있는 그래프입니다.  
관련 링크: https://cafe.daum.net/flowlife/SBU0/5

---

## Boxplot 기준 이상치(outlier)의 정의

이상치는 다음과 같이 정의합니다.

- $Q_1$: 1사분위수 (하위 25%)
- $Q_3$: 3사분위수 (상위 75%)
- $IQR$: 사분위 범위 ($IQR = Q_3 - Q_1$)

**이상치 경계:**
- 하한선 (lower bound): $Q_1 - 1.5 \times IQR$
- 상한선 (upper bound): $Q_3 + 1.5 \times IQR$

이 두 경계 바깥에 있는 값 → 이상치