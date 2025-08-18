# 1. 통계 분석 요약 (0814)

본 문서는 ANOVA, 가설검정 구조, 카이제곱 검정(CV vs p-value 포함) 핵심 정리.

---

## 2. 가설 검정 기본 구조
2.1 용어  
- 귀무가설(H₀): 차이·효과·관계 없음  
- 대립가설(H₁): 차이·효과·관계 있음  
2.2 판단  
- 유의수준 α (기본 0.05)  
- p-value < α → H₀ 기각 / p-value ≥ α → H₀ 기각 못함  
2.3 예시  
- H₀: μ_개 = μ_고양이 / H₁: μ_개 ≠ μ_고양이 (t-test 또는 ANOVA)

---

## 3. ANOVA(분산분석)
3.1 목적: 여러 집단 평균 차이 검정 (F = 집단간분산 / 집단내분산)  
3.2 종류  
- 3.2.1 일원 ANOVA: 한 요인  
- 3.2.2 이원 ANOVA: 두 요인 + 상호작용  
- 3.2.3 반복측정 ANOVA: 동일 개체 반복 측정  
- 3.2.4 이원 반복측정 ANOVA: 2요인 + 반복  
3.3 가정: 정규성, 등분산성, 독립성(반복측정은 구형성)  
3.4 사후검정: 전체 유의 → Tukey 등으로 어떤 집단 차이인지 탐색  

---

## 4. 카이제곱(χ²) 검정 개념
4.1 목적: 범주형 관측도수 vs 기대도수 차이 평가  
4.2 종류  
- 4.2.1 적합도(Goodness of Fit)  
- 4.2.2 독립성(Independence)  
- 4.2.3 동질성(Homogeneity)  
4.3 절차  
1) 교차표 생성  
2) 기대도수 E = (행합×열합)/전체합  
3) χ² = Σ (O−E)² / E  
4) 자유도 df = (행−1)(열−1)  
5) p-value 또는 임계값으로 판단  
4.4 해석  
- χ² 크거나 p-value ≤ α → H₀ 기각(관계/차이 있음)  
- 그렇지 않음 → H₀ 기각 못함(관계 증거 부족)  
4.5 예시 가설  
- H₀: 벼락치기 공부와 합격 여부 무관  
- H₁: 관련 있음  

---

## 5. 코드 예시 (독립성 검정)
```python
import pandas as pd, scipy.stats as stats
data = pd.read_csv("./01_big_data_machine_learning/data/pass_cross.csv")
ctab = pd.crosstab(index=data['공부안함'], columns=data['불합격'], margins=True)
chi2, p, dof, expected = stats.chi2_contingency(ctab)
print(chi2, p, dof)
```
예시 결과: χ²=3.0, df=1, p=0.5578 → p>0.05 → H₀ 기각 못함.

---

## 6. p-value 개념
6.1 정의: H₀ 참 가정 하에 관측된 통계량 이상 나올 확률  
6.2 특징: 작을수록(≤α) H₀ 기각 근거 강함  
6.3 계산: 라이브러리 자동 (scipy.stats.*)

---

## 7. 임계값(CV) vs p-value 비교
7.1 두 방식은 동일 결론 도출 (형식만 다름)  
7.2 정의  
- CV: df, α로 분포표에서 찾은 경계  
- p-value: P(Χ²(df) ≥ 관측 χ²)  
7.3 결정 규칙  
| 방식 | 기각 조건 | 기각 못함 |
|------|-----------|-----------|
| CV   | χ² ≥ CV   | χ² < CV   |
| p    | p ≤ α     | p > α     |
7.4 p-value 선호 이유: 표 불필요, α 변경 유연  
7.5 계산 개요  
1) O 정리  2) E 계산  3) χ² 합산  4) df 산출  5) 오른쪽 꼬리 확률  
7.6 직관: (O−E) 괴리↑ → χ²↑ → p↓ → H₀ 기각 근거↑  
7.7 예시: χ²=14.2, df=5 → p≈0.014 <0.05 → H₀ 기각  
7.8 한 줄 요약: CV 비교와 p-value 비교는 논리 동일, p-value가 실무 표준.

---

## 8. 전처리·실무 팁
8.1 결측 제거: df = df.dropna(subset=['col1','col2'])  
8.2 범주 매핑: df['직급코드'] = df['직급'].replace({'이사':1,'부장':2,...})  
8.3 구간화: pd.cut / pd.qcut  
8.4 기대도수 체크: (expected < 5).sum()==0 권장  
8.5 독립성 ≠ 인과관계  

---

## 9. 추가 코드 스니펫
9.1 적합도(주사위)  
```python
import scipy.stats as stats
obs = [4,6,17,16,8,9]; exp=[10]*6
print(stats.chisquare(obs, exp))
```
9.2 선호도(음료)  
```python
import pandas as pd, scipy.stats as stats
url="https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/drinkdata.csv"
df = pd.read_csv(url)
print(stats.chisquare(df['관측도수']))
```
9.3 범주형 교차표 시각화(예시)  
```python
import seaborn as sns, matplotlib.pyplot as plt
sns.heatmap(ctab.iloc[:-1,:-1], annot=True, fmt='d', cmap='Blues')
plt.show()
```

---

## 10. 해석 요약 표
| 검정 | 질문 | 통계량 | 기각 기준(p) |
|------|------|--------|--------------|
| ANOVA | 평균 차이? | F | p ≤ α |
| χ² 적합도 | 분포 일치? | χ² | p ≤ α |
| χ² 독립성/동질성 | 관계/분포 동일? | χ² | p ≤ α |

---

## 11. 주의 사항
11.1 p ≥ α → H₀가 ‘참’ 입증 아님 (근거 부족)  
11.2 기대도수 너무 작으면 (특히 <5) χ² 근사 약화 → Fisher 등 고려  
11.3 통계적 유의성 ≠ 실질 효과 (효과크기 별도)  

---

## 12. 핵심 용어
- 자유도(df): 독립 정보 수  
- 기대도수(expected): H₀ 하 이론적 빈도  
- 사후검정(post-hoc): ANOVA 유의 후 세부 비교  

---

## 13. 참고 자료
- SciPy: chisquare, chi2_contingency  
- Wikipedia: ANOVA, Chi-squared test

- [Chi-squared test - Wikipedia](https://en.wikipedia.org/wiki/Chi-squared_test)
- [Scipy.stats.chi2_contingency - SciPy Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html)