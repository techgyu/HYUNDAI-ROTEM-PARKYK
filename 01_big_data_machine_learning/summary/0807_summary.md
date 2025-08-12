# 08/07 수업 + 코딩 주요 개념 요약
# 시각화 자료 활용 가이드

---

## 1. 정적(Static) 시각화
- 한 번만 보여주고, 상호작용이 필요 없는 경우 이미지를 `.png`, `.jpg`, `.svg` 등으로 저장해서 불러와 사용하는 것이 효율적이다.
- 예시:
    - 보고서에 삽입하는 매출 추이 그래프
    - 논문에 들어가는 산점도, 히스토그램
    - 프레젠테이션에 넣는 파이 차트
    - 신문/잡지 등 인쇄 매체용 인포그래픽
- 장점:
    - 빠른 로딩
    - 환경에 상관없이 동일한 결과 제공
    - 배포가 간편함
    - 파일로 저장해두면 재사용이 쉬움

---

## 2. 동적(Interactive) 시각화
- 사용자의 상호작용(확대, 축소, 필터, 마우스 오버 등)이 필요하거나, 데이터가 자주 바뀌는 경우에는 데이터만 넘겨주고 라이브러리(예: matplotlib, plotly, seaborn, d3.js, Highcharts 등)를 이용해 실시간으로 처리하도록 한다.
- 예시:
    - 웹 대시보드(Plotly Dash, Tableau, Power BI 등)
    - 실시간 주가/환율 차트
    - 사용자가 원하는 조건으로 필터링하는 데이터 분석 툴
    - 지도 기반 데이터 시각화(예: 코로나 확진자 분포, 물류 이동 경로 등)
- 장점:
    - 사용자 맞춤형 분석
    - 다양한 시나리오 대응
    - 최신 데이터 반영 가능
    - 데이터 탐색 및 인사이트 도출에 유리

---

## 3. 시각화 활용 팁
- 정적/동적 시각화 선택은 목적과 사용 환경(웹, 문서, 발표 등)에 따라 결정한다.
- 정적 이미지는 용량이 크지 않게 저장하고, 해상도와 포맷을 상황에 맞게 선택한다.
- 동적 시각화는 데이터 보안, 성능, 라이브러리 호환성 등을 고려해야 한다.
- 시각화의 목적(비교, 분포, 추세, 관계 등)에 따라 적합한 그래프 종류(막대, 선, 산점도, 히트맵 등)를 선택한다.

---

## 4. 시각화 예시 코드

### 4-1. 정적 시각화 (matplotlib)
```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [10, 20, 15, 25, 30]
plt.plot(x, y)
plt.title('매출 추이')
plt.xlabel('월')
plt.ylabel('매출')
plt.savefig('sales_trend.png')  # 이미지로 저장
plt.close()
```

### 4-2. 동적 시각화 (plotly)
```python
import plotly.express as px

df = px.data.gapminder().query("year == 2007")
fig = px.scatter(df, x="gdpPercap", y="lifeExp", size="pop", color="continent",
                 hover_name="country", log_x=True, size_max=60)
fig.show()
```

---

## 5. matplotlib (정적 시각화) 주요 함수

1. **plt.plot(x, y)** : 선 그래프 그리기  
    - x, y 값의 추이(변화)를 선으로 연결해 시각화  
    - 예시: 시간에 따른 매출 변화  
    - 옵션: `color`, `linestyle`, `marker` 등으로 선 색상, 스타일, 마커 지정 가능
2. **plt.bar(x, y)** : 막대 그래프 그리기  
    - 범주형 데이터의 크기 비교에 사용  
    - 예시: 월별 판매량, 제품별 매출  
    - 옵션: `color`, `width`, `align` 등
3. **plt.scatter(x, y)** : 산점도 그리기  
    - 두 변수의 관계(분포, 상관관계) 시각화  
    - 예시: 키와 몸무게, GDP와 기대수명  
    - 옵션: `color`, `s`(점 크기), `alpha`(투명도) 등
4. **plt.hist(x)** : 히스토그램 그리기  
    - 데이터의 분포(빈도, 구간별 개수) 시각화  
    - 예시: 시험 점수 분포, 연령대 분포  
    - 옵션: `bins`(구간 개수), `color`, `alpha`
5. **plt.title(), plt.xlabel(), plt.ylabel()** : 그래프 제목, x/y축 라벨 설정  
    - `plt.title('제목')`, `plt.xlabel('x축 이름')`, `plt.ylabel('y축 이름')`
6. **plt.legend()** : 범례 표시  
    - 여러 데이터 시리즈를 한 그래프에 그릴 때 각 시리즈의 이름 표시  
    - 예시: `plt.plot(x, y, label='매출')` 후 `plt.legend()`
7. **plt.savefig('파일명.png')** : 그래프를 이미지 파일로 저장  
    - 다양한 포맷 지원(`.png`, `.jpg`, `.svg` 등)  
    - DPI, 투명도 등 옵션 지정 가능
8. **plt.show()** : 그래프 화면에 출력  
    - 주피터 노트북/스크립트 모두에서 사용
9. **plt.close()** : 현재 그래프 닫기  
    - 여러 그래프를 연속으로 그릴 때 이전 그래프를 지우고 새로 그림

---

## 6. plotly (동적 시각화) 주요 함수

1. **plotly.express.scatter()** : 동적 산점도  
    - 마우스 오버, 확대/축소, 범례 클릭 등 상호작용 지원  
    - 예시:  
      ```python
      import plotly.express as px
      fig = px.scatter(df, x='gdpPercap', y='lifeExp', color='continent')
      fig.show()
      ```
2. **plotly.express.bar()** : 동적 막대 그래프  
    - 막대 클릭, 범례 필터 등 동적 기능  
    - 예시:  
      ```python
      fig = px.bar(df, x='country', y='pop')
      fig.show()
      ```
3. **plotly.express.line()** : 동적 선 그래프  
    - 시계열 데이터, 추이 분석에 적합  
    - 예시:  
      ```python
      fig = px.line(df, x='year', y='value', color='category')
      fig.show()
      ```
4. **fig.show()** : plotly 그래프를 브라우저에 출력  
    - 노트북/웹/로컬 환경 모두 지원

---

## 7. pandas 데이터 처리 주요 함수

1. **pd.read_csv()** : CSV 파일 읽기  
    - 다양한 옵션(`sep`, `encoding`, `header`, `index_col` 등)으로 데이터 불러오기  
    - 예시:  
      ```python
      df = pd.read_csv('data.csv', encoding='utf-8')
      ```
2. **DataFrame.head(), DataFrame.tail()** : 데이터 앞/뒤 일부 확인  
    - `df.head(5)`, `df.tail(3)` 등으로 데이터 미리보기
3. **DataFrame.describe()** : 주요 통계 요약  
    - 평균, 표준편차, 최소/최대, 사분위수 등 자동 계산  
    - 수치형 데이터 전체 요약
4. **DataFrame.groupby()** : 그룹별 집계  
    - 특정 컬럼 기준으로 데이터 묶고, 합계/평균 등 집계  
    - 예시:  
      ```python
      df.groupby('category')['value'].mean()
      ```
5. **DataFrame.pivot_table()** : 피벗 테이블 생성  
    - 여러 기준으로 데이터 요약, 집계  
    - 예시:  
      ```python
      df.pivot_table(index='A', columns='B', values='C', aggfunc='sum')
      ```
6. **DataFrame.plot()** : pandas 내장 시각화 (matplotlib 기반)  
    - `df.plot(kind='bar')`, `df.plot(kind='line')` 등  
    - 빠르게 데이터 시각화 가능

---

## 8. 기타 시각화/분석 관련 함수

1. **plt.subplot()** : 여러 그래프를 한 화면에 배치  
    - 예시:  
      ```python
      plt.subplot(2, 1, 1)  # 2행 1열 중 첫 번째
      plt.plot(x1, y1)
      plt.subplot(2, 1, 2)  # 두 번째
      plt.plot(x2, y2)
      ```
2. **plt.grid()** : 격자 표시  
    - 그래프 해석을 쉽게 해줌  
    - 예시: `plt.grid(True)`
3. **plt.xticks(), plt.yticks()** : 축 눈금 설정  
    - 눈금 위치, 라벨 지정  
    - 예시:  
      ```python
      plt.xticks([0, 1, 2], ['A', 'B', 'C'])
      ```
4. **plt.annotate()** : 그래프에 텍스트 표시  
    - 특정 위치에 설명, 값 등 표시  
    - 예시:  
      ```python
      plt.annotate('최고점', xy=(x, y), xytext=(x+1, y+1),
                   arrowprops=dict(facecolor='black', shrink=0.05))
      ```
5. **plt.style.use()** : 스타일 테마 적용  
    - 다양한 스타일(`'ggplot'`, `'seaborn'`, `'dark_background'` 등)으로 그래프 분위기 변경