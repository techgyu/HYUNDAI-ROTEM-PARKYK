# 08/07 수업 + 코딩 주요 개념 요약
# 시각화 자료 활용 가이드

- **정적(Static) 시각화**  
  - 한 번만 보여주고, 상호작용이 필요 없는 경우에는 이미지를 `.png`, `.jpg`, `.svg` 등으로 저장해서 불러와 사용하는 것이 효율적이다.
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

- **동적(Interactive) 시각화**  
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

- **추가 팁**
  - 정적/동적 시각화 선택은 목적과 사용 환경(웹, 문서, 발표 등)에 따라 결정한다.
  - 정적 이미지는 용량이 크지 않게 저장하고, 해상도와 포맷을 상황에 맞게 선택한다.
  - 동적 시각화는 데이터 보안, 성능, 라이브러리 호환성 등을 고려해야 한다.
  - 시각화의 목적(비교, 분포, 추세, 관계 등)에 따라 적합한 그래프 종류(막대, 선, 산점도, 히트맵 등)를 선택한다.

---

## 시각화 예시 코드

### 정적 시각화 (matplotlib)
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

### 동적 시각화 (plotly)
```python
import plotly.express as px

df = px.data.gapminder().query("year == 2007")
fig = px.scatter(df, x="gdpPercap", y="lifeExp", size="pop", color="continent",
                 hover_name="country", log_x=True, size_max=60)
fig.show()  #
```
