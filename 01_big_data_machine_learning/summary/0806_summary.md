# 08/06 수업 + 코딩 주요 개념 요약

---

## 1. 시큐어 코딩 가이드라인
- **표준 가이드라인**을 따라야 한다.
- **입력값 검증**: 사용자 입력, 외부 데이터 등 모든 입력값에 대해 유효성 검사를 수행한다.
- **인증 및 인가**: 사용자 인증(로그인)과 권한 확인(인가)을 반드시 구현한다.
- **데이터 암호화**: 중요 정보(비밀번호, 개인정보 등)는 저장 및 전송 시 암호화한다.
- **에러 및 예외 처리**: 에러 메시지에 민감한 정보가 노출되지 않도록 주의하고, 예외 상황을 안전하게 처리한다.
- **자원 관리**: 파일, 네트워크, DB 연결 등 외부 자원을 사용한 후 반드시 반환(닫기)한다.
- **보안 취약점 예방**: SQL Injection, XSS, CSRF 등 주요 보안 취약점에 대한 방어 코드를 작성한다.
- **최신 보안 패치 적용**: 사용하는 라이브러리, 프레임워크, DB 등에 대해 최신 보안 패치를 적용한다.

---

## 2. UI / UX
- **사용자 경험(UX)**은 중요하다.
- **직관적인 인터페이스**와 **일관된 디자인**을 제공한다.
- **접근성(Accessibility)**과 **반응성(Responsiveness)**을 고려한다.
- **사용자 피드백**을 적극 반영하여 개선한다.

---

## 3. 정규 표현식
- **문자열에서 특정 패턴을 찾거나 치환**할 때 사용한다.
- 예시: 이메일, 전화번호, 우편번호 등 형식 검증
- 복잡한 패턴 작성 시 **가독성과 성능**을 고려한다.
- `re` 모듈의 주요 함수:
    - `re.match()` : 문자열의 처음부터 정규식과 매치되는지 검사
    - `re.search()` : 문자열 전체에서 정규식과 매치되는 부분 검색
    - `re.findall()` : 정규식과 매치되는 모든 부분을 리스트로 반환
    - `re.sub()` : 정규식과 매치되는 부분을 다른 문자열로 치환

#### 실전 예시
```python
import re

# 이메일 검증
email = "test@example.com"
if re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email):
    print("올바른 이메일 형식입니다.")

# 전화번호 추출
text = "문의: 010-1234-5678, 02-123-4567"
phones = re.findall(r"\d{2,3}-\d{3,4}-\d{4}", text)
print(phones)  # ['010-1234-5678', '02-123-4567']

# 문자열 치환
sentence = "Hello, World!"
result = re.sub(r"World", "Python", sentence)
print(result)  # Hello, Python!
```
- 정규 표현식은 **데이터 유효성 검사, 데이터 전처리, 로그 분석** 등 다양한 분야에서 활용된다.

---

## 4. 청크(데이터 분석)
- **대량의 데이터를 한 번에 메모리에 올리지 않고**, 일정 크기씩 나누어(청크 단위로) 처리하는 방법
- pandas의 `read_csv` 등에서 `chunksize` 파라미터로 사용
- **메모리 효율적 데이터 처리** 및 **스트리밍 분석**에 유용

#### 예시
```python
for chunk in pd.read_csv('bigdata.csv', chunksize=10000):
    # chunk 단위로 데이터 처리
    pass
```

---

## 5. pandas 주요 객체/함수

### Series
- **1차원 데이터(리스트, 배열 등)와 인덱스를 결합한 자료구조**
- 생성: `pd.Series(data, index=None, dtype=None, name=None, ...)`
    - `data`: 리스트, 배열, 딕셔너리 등 1차원 데이터
    - `index`: 인덱스 라벨(생략 시 0부터 자동 부여)
    - `dtype`: 데이터 타입 지정
    - `name`: Series 이름
- 주요 속성/메서드:
    - `.values`: 데이터 값(numpy 배열)
    - `.index`: 인덱스 객체
    - `.dtype`: 데이터 타입
    - `.head(n)`: 앞에서 n개 데이터 반환(기본 5개)
    - `.tail(n)`: 뒤에서 n개 데이터 반환(기본 5개)

### DataFrame
- **2차원 표 형태의 데이터(행/열)**
- 생성: `pd.DataFrame(data, index=None, columns=None, dtype=None, ...)`
    - `data`: 2차원 배열, 딕셔너리, 리스트 등
    - `index`: 행 인덱스
    - `columns`: 열 이름
    - `dtype`: 데이터 타입
- 주요 속성/메서드:
    - `.values`: 데이터 값(numpy 2차원 배열)
    - `.columns`: 열 이름(Index 객체)
    - `.index`: 행 인덱스(Index 객체)
    - `.info()`: 데이터프레임 구조, 타입, 결측치 등 정보 출력
    - `.describe()`: 주요 통계 요약(평균, 표준편차 등)

### merge
- **두 DataFrame을 특정 컬럼(또는 인덱스) 기준으로 병합**
- 사용법: `pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None, ...)`
    - `left`, `right`: 병합할 DataFrame
    - `how`: 'inner', 'outer', 'left', 'right' (조인 방식)
    - `on`: 병합 기준 컬럼명(리스트 가능)
    - `left_on`, `right_on`: 각각 왼쪽/오른쪽 DataFrame의 병합 기준 컬럼
- SQL의 JOIN과 유사하게 동작

### concat
- **여러 DataFrame/Series를 행 또는 열 방향으로 이어붙임**
- 사용법: `pd.concat(objs, axis=0, join='outer', ignore_index=False, ...)`
    - `objs`: 합칠 객체 리스트
    - `axis`: 0(행방향, 기본), 1(열방향)
    - `join`: 'outer'(기본, 전체 합침), 'inner'(공통 부분만 합침)
    - `ignore_index`: 인덱스 무시하고 새로 부여

### groupby
- **특정 컬럼(또는 인덱스) 기준으로 그룹화**
- 사용법: `df.groupby(by, axis=0, as_index=True, ...)`
    - `by`: 그룹화 기준 컬럼명/리스트/함수
    - `as_index`: 그룹 컬럼을 인덱스로 쓸지 여부
- 그룹별 집계, 변환, 필터링 등에 사용

### pivot
- **행/열/값 기준으로 데이터 재구조화**
- 사용법: `df.pivot(index=None, columns=None, values=None)`
    - `index`: 행 인덱스가 될 컬럼명
    - `columns`: 열 인덱스가 될 컬럼명
    - `values`: 값이 될 컬럼명
- 단일 값만 가능(중복 불가)

### pivot_table
- **그룹화 + 집계 + 피벗(결측치 처리, 여러 집계함수 지원)**
- 사용법: `df.pivot_table(values=None, index=None, columns=None, aggfunc='mean', fill_value=None, margins=False, ...)`
    - `values`: 집계할 컬럼명
    - `index`: 행 인덱스
    - `columns`: 열 인덱스
    - `aggfunc`: 집계 함수(기본 'mean', 'sum', 'count', np.median 등)
    - `fill_value`: 결측치 대체값
    - `margins`: 전체(합계) 행/열 추가

### 기술통계 함수
- `mean()`: 평균
- `var()`: 분산
- `std()`: 표준편차
- `max()`: 최대값
- `min()`: 최소값
- `sum()`: 합계
- `median()`: 중앙값
- `idxmax()`: 최대값의 인덱스
- `idxmin()`: 최소값의 인덱스
- `corr()`: 상관계수
- `cov()`: 공분산
- `describe()`: 주요 통계 요약(평균, 표준편차, 사분위수 등)
- 대부분 axis, skipna 등 매개변수 지원

### 결측치 처리
- `isnull()`: 결측치 여부(True/False)
- `notnull()`: 결측치가 아닌지 여부
- `dropna(axis=0, how='any', subset=None)`: 결측치가 있는 행/열 삭제
    - `axis`: 0(행), 1(열)
    - `how`: 'any'(하나라도 결측이면), 'all'(전부 결측이면)
    - `subset`: 특정 컬럼만 적용
- `fillna(value, method=None, axis=None, ...)`: 결측치 채우기
    - `value`: 채울 값
    - `method`: 'ffill'(앞값으로), 'bfill'(뒷값으로)

### 파일 입출력
- `read_csv(filepath, sep=',', header='infer', names=None, index_col=None, ...)`: CSV 파일 읽기
- `to_csv(filepath, sep=',', index=True, header=True, ...)`: CSV 파일 저장
- `read_excel()`, `to_excel()`, `read_html()`, `read_table()`, `read_fwf()` 등 다양한 포맷 지원
    - `filepath`: 파일 경로 또는 URL
    - `sep`: 구분자
    - `header`: 헤더 행 지정
    - `names`: 컬럼명 지정
    - `index_col`: 인덱스 컬럼 지정

### 청크 처리
- **대용량 파일을 나눠서 읽기**: `pd.read_csv(filepath, chunksize=10000)`
- 반환값은 이터레이터(반복문으로 chunk 단위 처리)

---

## 6. 데이터 전처리/분석

- **cut**: 연속형 데이터를 구간별로 나눔  
    - 사용법: `pd.cut(x, bins, labels=None, right=True, ...)`
    - 주요 매개변수:
        - `x`: 나눌 데이터
        - `bins`: 구간 경계값 리스트
        - `labels`: 구간별 라벨
        - `right`: 우측 경계 포함 여부
- **qcut**: 분위수(동일 개수) 기준으로 구간 나눔  
    - 사용법: `pd.qcut(x, q, labels=None, ...)`
    - 주요 매개변수:
        - `x`: 나눌 데이터
        - `q`: 구간 개수 또는 분위수 리스트
        - `labels`: 구간별 라벨
- **groupby**: 위 참조
- **agg**: 그룹별 여러 집계 함수 적용  
    - 사용법: `grouped.agg(['mean', 'sum', ...])`
- **apply**: 그룹별 사용자 정의 함수 적용  
    - 사용법: `grouped.apply(func)`
- **샘플링**:  
    - `sample(n=None, frac=None, random_state=None)`
    - 주요 매개변수:
        - `n`: 샘플 개수
        - `frac`: 전체에서 샘플링할 비율
        - `random_state`: 시드값(재현성)
- **describe**: 주요 통계 요약  
    - 사용법: `df.describe()`
- **info**: 데이터프레임 구조/타입/결측치 등 정보  
    - 사용법: `df.info()`

---

## 7. 웹 데이터 수집/전처리

- **BeautifulSoup**: HTML/XML 파싱 라이브러리  
    - 사용법: `BeautifulSoup(html, 'html.parser')`
    - 주요 메서드: `.find()`, `.find_all()`, `.select()`, `.get_text()`
- **requests**: HTTP 요청 라이브러리  
    - 사용법: `requests.get(url)`, `res.text`, `res.content`
- **csv**: CSV 파일 읽기/쓰기 표준 라이브러리  
    - 사용법: `csv.reader()`, `csv.writer()`
- **re**: 정규표현식 라이브러리  
    - 사용법: `re.match()`, `re.search()`, `re.findall()`, `re.sub()`

---