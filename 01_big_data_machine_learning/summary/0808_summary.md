# 08/08 수업 + 코딩 주요 개념 요약

---

## 1. 아웃라이어(이상치) 처리와 박스플롯(Boxplot)

1. **박스플롯(Boxplot) 개념**
    - 데이터의 분포와 이상치(outlier)를 시각적으로 쉽게 파악할 수 있는 그래프.
    - 박스: Q1 ~ Q3 구간(중앙 50% 데이터)
    - 중앙선: 중앙값(median)
    - 수염(whisker): 이상치 경계 내의 데이터 범위
    - 점: 이상치(outlier)

2. **이상치(outlier) 기준**
    - Q1 (1사분위수): 하위 25% 값
    - Q3 (3사분위수): 상위 75% 값
    - IQR (Interquartile Range): Q3 - Q1
    - 하한선: Q1 - 1.5 * IQR
    - 상한선: Q3 + 1.5 * IQR
    - 이 경계 바깥의 값은 이상치로 간주

---

## 2. 데이터 표기 및 실무 팁

1. **파이썬 튜플 표기**
    - (4): 정수 4로 인식
    - (4, ): 1개짜리 튜플로 인식 (콤마 필수)

2. **외부 자원 작업 시 예외 처리**
    - DB, 네트워크, 파일 등은 반드시 try-except로 예외 처리

---

## 3. pandas 데이터 연산 및 결합

1. **Series 연산**
    - 서로 다른 인덱스를 가진 Series끼리 연산 시, 인덱스가 일치하는 값만 계산되고 나머지는 NaN
    - `.add()`, `.multiply()` 등 메서드는 `fill_value`로 결측값 대체 가능
    - 예시:
      ```python
      s1.add(s2, fill_value=0)
      s1.multiply(s2, fill_value=1)
      ```

2. **DataFrame 연산**
    - DataFrame 간 연산도 인덱스(행/열)가 일치하는 값끼리 계산
    - `.add()`, `.multiply()` 등에서 `fill_value`로 결측값 처리 가능

3. **DataFrame과 Series 연산 (브로드캐스팅)**
    - DataFrame에서 특정 행 또는 열(Series)을 빼거나 더할 때 자동으로 각 행/열에 맞춰 연산

4. **결측치 처리**
    - 결측값 확인: `isnull()`, `notnull()`
    - 결측값 제거: `dropna(axis=0/1, how='any'/'all', subset=[...])`
    - 결측값 대체: `fillna(값)`

---

## 4. pandas 기술통계 및 요약

1. **기술통계 함수**
    - `mean()`, `var()`, `std()`, `max()`, `min()`, `sum()`, `median()`, `idxmax()`, `idxmin()`, `corr()`, `cov()`, `describe()` 등

2. **피벗, 그룹화, 집계**
    - `groupby()`: 특정 컬럼 기준으로 그룹화
    - `agg()`: 그룹별 여러 집계 함수 적용
    - `apply()`: 그룹별 사용자 정의 함수 적용
    - `pivot_table()`: 다중 기준 집계 및 요약, 결측치 처리, 다양한 집계 함수 지원

3. **구간화(cut, qcut)**
    - `pd.cut()`: 연속형 데이터를 지정한 구간으로 나눔, 구간별 라벨 지정 가능
    - `pd.qcut()`: 분위수 기준으로 구간 나눔(동일 개수로 분할)

---

## 5. 데이터 병합 및 연결

1. **merge**
    - SQL의 JOIN처럼 여러 DataFrame을 키 기준으로 병합
    - `how` 옵션: 'inner', 'outer', 'left', 'right'
    - `on`, `left_on`, `right_on` 등 다양한 병합 기준 지정 가능

2. **concat**
    - 여러 DataFrame/Series를 행 또는 열 방향으로 단순 연결
    - `axis=0`(행), `axis=1`(열), `ignore_index=True` 등 옵션

---

## 6. 샘플링 및 데이터 요약

1. **샘플링**
    - `sample(n=...)`: n개 무작위 추출
    - `sample(frac=...)`: 전체에서 비율로 무작위 추출
    - `random_state`: 시드값으로 재현성 확보

2. **기타**
    - `head()`, `tail()`: 처음/마지막 n개 데이터 확인
    - `info()`: 데이터프레임 구조, 타입, 결측치 등 정보
    - `describe()`: 주요 통계 요약

---

## 7. 파일 입출력

1. **CSV, Excel, 텍스트 파일 읽기/쓰기**
    - `read_csv()`, `to_csv()`, `read_excel()`, `to_excel()`, `read_table()`, `read_fwf()` 등 다양한 포맷 지원
    - `chunksize` 옵션으로 대용량 파일을 청크 단위로 읽어 메모리 효율적으로 처리

---

## 8. 실전 예시 및 실습 문제

1. **타이타닉, tips, human 등 실제 데이터셋을 활용한 실습**
    - `cut()`으로 나이대 구간화
    - `pivot_table()`로 생존율 집계
    - `groupby()`로 그룹별 통계
    - 문자열 전처리: `.str.strip()`으로 공백 제거, 결측치 처리, 컬럼명 정리