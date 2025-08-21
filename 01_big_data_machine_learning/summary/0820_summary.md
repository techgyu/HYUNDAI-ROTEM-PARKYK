# 1. 세 개 이상의 모집단에 대한 가설검정 – 분산분석(ANOVA)

1.1 **분산분석(ANOVA)란?**
- 분산이 발생한 과정을 분석하여, 요인에 의한 분산과 집단 내 분산으로 나누고  
  요인에 의한 분산이 의미 있는 크기를 가지는지 검정하는 방법
- 평균을 직접 비교하지 않고, 집단 내 분산과 집단 간 분산을 이용하여  
  집단의 평균이 서로 다른지 확인
- f-value = 그룹 간 분산(Between variance) / 그룹 내 분산(Within variance)

1.2 **집단 평균 비교의 오류**
- 세 집단 이상의 평균비교에서 두 집단 평균 비교를 반복하면  
  제1종 오류(잘못된 유의성 판정)가 증가
- 이를 해결하기 위해 Fisher가 개발한 분산분석(ANOVA)을 사용

1.3 **분산분석 참고 자료**
- f-value의 의미: [Daum Cafe 그림](https://cafe.daum.net/flowlife/SBU0/28)
- 면접 대비: [Daum Cafe](https://cafe.daum.net/flowlife/SBU0/43)
- 추가 설명: [Hazel 블로그](https://hazel01.tistory.com/15)
- 위키백과: [분산분석](https://ko.wikipedia.org/wiki/%EB%B6%84%EC%82%B0_%EB%B6%84%EC%84%9D)

---

# 2. 세 집단 이상 평균 비교 시 절차

2.1 **정규성 + 등분산성 가정 충족 여부**
- 정규성 만족 + 등분산성 만족 → 일원분산분석(One-way ANOVA)
- 정규성 만족 + 등분산성 불만족 → Welch’s ANOVA
- 정규성 불만족 → 비모수 검정(Kruskal-Wallis test)

2.2 **사후검정(Post-hoc test)**
- ANOVA는 "차이가 있다"까지만 알려줌  
  → 구체적으로 어떤 집단쌍이 차이가 나는지 알고 싶으면 사후검정 필요
- 정규성 만족 + 등분산성 만족 → Tukey’s HSD test
- 등분산성 불만족 → Games-Howell test
- 비모수일 때 → Dunn’s test, Conover’s test 등

---

# 3. ANOVA 결과표 주요 용어

3.1 **SSR, MSR, SSE, MSE, F, PR(>F)의 의미**
- sum_sq: 각 요인별 제곱합(분산의 크기)
- df: 자유도
- F: F-통계량 (집단 간 분산 / 집단 내 분산)
- PR(>F): p-value (F값이 우연히 나올 확률)

예시:
```
                 sum_sq    df         F    PR(>F)
C(method)     28.907967   2.0  0.062312  0.939639
Residual   17397.207418  75.0       NaN       NaN
```
- C(method): 교육방법에 의한 분산(집단 간 분산)
- Residual: 오차(집단 내 분산)
- df: 자유도 (집단 수 - 1, 전체 표본 수 - 집단 수)
- F: 집단 간 분산 / 집단 내 분산
- PR(>F): p-value (0.939639 → 0.05보다 크면 귀무가설 채택, 평균 차이 없음)