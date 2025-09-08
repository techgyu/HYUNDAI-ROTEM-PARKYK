# EDA (Exploratory Data Analysis) 완벽 가이드

## 📊 EDA란 무엇인가?

**탐색적 데이터 분석(EDA)**은 분석을 목적으로 데이터의 성격 및 구조를 이해하는 과정입니다. 분석 모델을 통해 최적의 결과를 얻기 위한 가장 쉬운 방법은 질문을 잘 만들어 둔 후 탐색과정을 통해 결국 데이터를 표현하는 적절한 모형, 시각화, 다음 과정을 위한 데이터를 생성하는 것입니다.

### 🎯 EDA의 목적

- **데이터 품질 향상**: 기계학습 알고리즘의 성능은 데이터의 품질과 정보량에 달려있음
- **모델링 방향성 설정**: 데이터에 적합한 분석 기법을 결정
- **인사이트 발견**: 데이터의 숨겨진 구조와 관계, 이상값, 패턴을 찾아냄
- **가설 설정**: 데이터와 대화하며 문제 해결의 실마리를 제공

---

## 🔍 EDA의 6가지 핵심 영역

### 1. **Data Understanding (데이터 이해)**
- 데이터를 처음 접할 때 가장 먼저 해야 할 작업
- 변수의 의미, 단위, 타입, 결측치 여부 등을 파악
- **목적**: 모델링 이전에 데이터를 "해석"하는 첫 단계

```python
# 기본 정보 확인
df.info()           # 데이터 타입, 결측치 개수
df.describe()       # 기초 통계량
df.head()          # 상위 몇 개 행
df.shape           # 데이터 크기
```

### 2. **Data Cleaning (데이터 정제)**
- 이상치 제거, 결측값 처리, 중복 제거, 포맷 통일 등
- 데이터 품질을 개선해 모델 학습의 정확도 향상
- **목적**: Garbage in, garbage out 방지

```python
# 결측치 확인
df.isnull().sum()
df.isnull().sum() / len(df) * 100  # 결측치 비율

# 중복 확인
df.duplicated().sum()

# 이상치 확인
df.describe()  # 통계량으로 이상값 확인
```

### 3. **Pattern Discovery (패턴 발견)**
- 변수 간의 관계, 트렌드, 상관관계 등을 탐색
- 비즈니스 인사이트 도출 가능 (예: 고객 세분화)
- **목적**: 예측 모델의 방향성 설정

```python
# 상관관계 분석
df.corr()

# 그룹별 패턴 분석
df.groupby('category').mean()

# 트렌드 분석
df.groupby('date').sum().plot()
```

### 4. **Data Visualization (데이터 시각화)**
- 데이터를 그래픽으로 표현해 직관적으로 이해
- 히스토그램, 박스플롯, 산점도, 히트맵 등 활용
- **목적**: 데이터 속 숨은 구조나 이상값 시각적 탐색

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 분포 확인
plt.hist(df['column'])
sns.distplot(df['column'])

# 상관관계 시각화
sns.heatmap(df.corr(), annot=True)

# 범주별 분포
sns.boxplot(x='category', y='value', data=df)
```

### 5. **Model Selection (모델 선택)**
- EDA 결과를 바탕으로 어떤 모델이 적합한지 판단
- 예: 정규분포 → 선형 회귀 가능 / 비선형 → 트리 기반
- **목적**: 모델링의 사전 설계에 결정적 역할

### 6. **Quality Control (품질 관리)**
- 전체 분석 과정 중 데이터와 결과의 일관성 유지
- 분석 오류, 편향 데이터, 변형된 스케일 문제 확인
- **목적**: 신뢰할 수 있는 결과 도출을 위한 검증 단계

---

## 🛠️ 실습 예제

### 자동 EDA: pandas-profiling 사용

```bash
pip install -U pandas-profiling
```

```python
import pandas as pd
import pandas_profiling

# 데이터 로드
data = pd.read_csv('datasets_spam.csv', encoding='latin1')

# 자동 EDA 리포트 생성
pr_report = data.profile_report()
pr_report.to_file('./pr_report.html')

print(pr_report)
```

**장점**: 한 번의 명령으로 Overview, Variables, Correlations, Missing values 등 모든 정보를 포함한 HTML 리포트 생성

---

### 수동 EDA: Titanic 데이터 예제

#### 📋 1단계: 라이브러리 및 데이터 로드

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 타이타닉 데이터 로드
titanic = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/titanic_data.csv")

print(titanic.head())
print(titanic.info())
```

#### 🔧 2단계: 데이터 타입 수정

```python
# 범주형 변수를 올바른 타입으로 변경
titanic['Survived'] = titanic['Survived'].astype(object)
titanic['Pclass'] = titanic['Pclass'].astype(object)
```

**💡 팁**: 숫자로 된 범주형 변수(0,1,2 등)는 int64로 잘못 인식될 수 있으므로 수동으로 object 타입으로 변경

#### 📊 3단계: 결측치 분석

```python
# 결측치 확인
print(titanic.isnull().sum())

# 결측치 비율 계산
missing_df = titanic.isnull().sum().reset_index()
missing_df.columns = ['column', 'count']
missing_df['ratio'] = missing_df['count'] / titanic.shape[0]
print(missing_df.loc[missing_df['ratio'] != 0])
```

**결과 해석**:
- Age: 177개 (19.9% 결측)
- Cabin: 687개 (77.1% 결측) 
- Embarked: 2개 (0.2% 결측)

#### 📈 4단계: 종속변수 분석

```python
# 생존 여부 분포 확인
titanic['Survived'].value_counts().plot(kind='bar')
plt.title('Survival Distribution')
plt.show()
```

#### 🏷️ 5단계: 명목형 변수 탐색

```python
# 명목형 변수 추출
category_feature = [col for col in titanic.columns if titanic[col].dtypes == "object"]
category_feature = list(set(category_feature) - set(['PassengerId','Survived']))

print(category_feature)
# ['Cabin', 'Embarked', 'Ticket', 'Sex', 'Name', 'Pclass']

# 각 명목형 변수의 분포 시각화
for col in category_feature:
    titanic[col].value_counts().plot(kind='bar')
    plt.title(col)
    plt.show()
```

#### 📊 6단계: 이변수 관계 분석

```python
# 성별과 생존율 관계
sex_df = titanic.groupby(['Sex','Survived'])['Survived'].count().unstack('Survived')
sex_df.plot(kind='bar', figsize=(10,6))
plt.title('Survival by Gender')
plt.show()
```

#### 🔢 7단계: 수치형 변수 탐색

```python
# 수치형 변수 추출
numerical_feature = list(set(titanic.columns) - set(category_feature) - set(['PassengerId','Survived']))
numerical_feature = np.sort(numerical_feature)

# 각 수치형 변수의 분포 확인
for col in numerical_feature:
    sns.distplot(titanic.loc[titanic[col].notnull(), col])
    plt.title(col)
    plt.show()
```

#### 🔗 8단계: 다변수 관계 분석

```python
# Pairplot으로 다변수 관계 한 번에 확인
sns.pairplot(titanic[list(numerical_feature) + ['Survived']], 
             hue='Survived', 
             x_vars=numerical_feature, 
             y_vars=numerical_feature)
plt.show()
```

#### 📦 9단계: 수치형-명목형 변수 관계

```python
# 성별에 따른 수치형 변수들의 분포 (생존 여부별)
for col in numerical_feature:
    plt.figure(figsize=(12,6))
    sns.boxplot(x='Sex', y=col, hue='Survived', data=titanic.dropna())
    plt.title("Sex - {}".format(col))
    plt.show()
```

---

## 🎯 EDA에서 발견할 수 있는 인사이트

### 타이타닉 데이터 예시

1. **성별과 생존율**: 
   - 여성의 생존율 > 남성의 생존율
   - "여성과 어린이 먼저" 원칙 확인

2. **객실 등급과 생존율**:
   - 1등급 > 2등급 > 3등급 순으로 생존율 높음
   - 사회적 계층과 생존 기회의 상관관계

3. **나이 분포**:
   - 어린이(0-10세)의 상대적으로 높은 생존율
   - 중년층 vs 노년층의 생존 패턴 차이

4. **결측치 패턴**:
   - Cabin 정보 부족 → 하위 등급 승객 추정 가능

---

## ✅ EDA 체크리스트

### 📋 기본 탐색
- [ ] 데이터 크기 확인 (`df.shape`)
- [ ] 변수 타입 확인 (`df.info()`)
- [ ] 기초 통계량 확인 (`df.describe()`)
- [ ] 결측치 확인 (`df.isnull().sum()`)
- [ ] 중복값 확인 (`df.duplicated().sum()`)

### 📊 단변수 분석
- [ ] 수치형 변수: 히스토그램, 박스플롯
- [ ] 범주형 변수: 빈도표, 막대그래프
- [ ] 이상값 탐지
- [ ] 분포의 정규성 확인

### 🔗 다변수 분석
- [ ] 상관관계 매트릭스
- [ ] 산점도 매트릭스 (pairplot)
- [ ] 그룹별 통계량 비교
- [ ] 교차표 분석

### 🎨 시각화
- [ ] 적절한 차트 타입 선택
- [ ] 색상과 범례 활용
- [ ] 제목과 라벨 명시
- [ ] 스케일 조정

---

## 🚀 EDA 모범 사례

### ✅ Do's (해야 할 것)

1. **질문 기반 접근**
   - "이 데이터로 무엇을 알고 싶은가?"
   - "어떤 가설을 검증하고 싶은가?"

2. **단계별 접근**
   - 개별 변수 → 변수 간 관계 → 종합 분석

3. **도메인 지식 활용**
   - 비즈니스 맥락 고려
   - 전문가 의견 반영

4. **반복적 분석**
   - 새로운 발견 시 재분석
   - 가설 수정 및 재검증

### ❌ Don'ts (하지 말아야 할 것)

1. **무작정 그래프만 그리기**
   - 목적 없는 시각화는 의미 없음

2. **이상값 무조건 제거**
   - 이상값도 중요한 정보일 수 있음

3. **상관관계 = 인과관계로 해석**
   - 상관관계와 인과관계는 다름

4. **결측치 무조건 제거**
   - 결측치 패턴 자체가 정보가 될 수 있음

---

## 📚 참고 자료

- [EDA AI Lab](https://eda-ai-lab.tistory.com/13)
- [Python 탐색적 자료분석 가이드](https://3months.tistory.com/325)
- [Kaggle SMS Spam Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset)

---

## 💡 결론

EDA는 단순한 데이터 요약이 아닌, **"이해 → 정제 → 탐색 → 시각화 → 모델 선택 → 품질 관리"** 전 과정을 아우르는 데이터 과학의 핵심 활동입니다.

> **"EDA는 데이터와의 첫 만남이자, 모든 분석의 출발점입니다!"**

데이터 속에 숨어있는 스토리를 찾아내는 탐정이 되어 보세요! 🕵️‍♂️
