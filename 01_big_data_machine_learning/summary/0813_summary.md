# 서버에서 긁어오는 데이터의 문제점은 항상 데이터의 소스와 구조가 바뀌기 때문에 문제가 있다.

# 정형 데이터 vs 비정형 데이터


# DIKW 피라미드 요약

- **DATA(데이터)**:  
  단순한 값이나 사실(예: 180/120). 이 자체로는 의미를 알 수 없음.

- **Information(정보)**:  
  데이터에 의미를 부여한 것(예: 180/120이 혈압 수치임을 알게 됨).

- **Knowledge(지식)**:  
  정보를 경험이나 맥락과 결합해 이해하는 것(예: 180/120은 높은 혈압이라는 사실을 경험적으로 알게 됨).

- **Wisdom(지혜)**:  
  지식을 바탕으로 미래의 행동이나 결정을 내리는 것(예: 180/120 혈압을 가진 사람은 운동, 식단조절 등 조치가 필요함을 알게 됨).


# 빅데이터 분석 5단계
- 분석 기획
- 데이터 준비
- 데이터 분석
- 시스템 구현
- 평가 및 전개

# 오늘은 p-value를 알고 집에 간다

# 피쳐 독립변수

# 레이블 종속변수

원인 변수 반응 변수

예측 변수 결과 변수

# 모집단일 떄는 M 표본 일때, 분산일 때, 표준 편차일 때, 개수일 때 사용하는 기호가 정해져 있다.

# 각 표본은 동일한 확률 분포를 갖는다

## pd.cut() 함수 타입 힌트 분석
```python
def cut(
    x: Series | Index | ArrayLike,
    bins: int | Series | Index[int] | Index[float] | Sequence[int] | Sequence[float] | IntervalIndex,
    right: bool = True,
    labels: Sequence[Hashable] | bool | None = None,
    retbins: bool = False,
    precision: int = 3,
    include_lowest: bool = False,
    duplicates: str = "raise",
    ordered: bool = True,
) -> Series | tuple[Series, ndarray]
```

### 매개변수 설명
- **`x`**: 잘라낼 데이터 (Series, Index, ArrayLike)
- **`bins`**: 구간 설정 (int=구간개수, Sequence=구간경계, 기타)
- **`right`**: 구간의 오른쪽 포함 여부 (True=(a,b], False=[a,b))
- **`labels`**: 구간 라벨 (문자열 리스트, False=정수코드, None=기본표기)
- **`retbins`**: 구간 경계값도 함께 반환할지 여부
- **`precision`**: 구간 표시 시 소수점 자릿수
- **`include_lowest`**: 최소값을 첫 구간에 포함시킬지 여부
- **`duplicates`**: 중복 구간 경계 처리 방법 ("raise", "drop")
- **`ordered`**: 결과 카테고리의 순서 여부

### 타입 힌트 읽기 기호
- **`|`**: "또는" (Union 타입)
- **`[T]`**: 제네릭 타입 (T 타입의 요소를 가진 컨테이너)
- **`= 값`**: 기본값
- **`-> 타입`**: 반환 타입
- **`Sequence`**: list, tuple 등 순서가 있는 컬렉션
- **`ArrayLike`**: numpy array, list 등 배열과 유사한 객체

### 주의사항
- `bins`에 `numpy.ndarray` 사용 시 타입 경고 발생
- 해결: `bins = list(np.arange(156, 195, 5))`로 변환


# 중심 극한 정리 이해할 것
- 30개 이상의 데이터를 수집해야 하는 이유 -> 중심 극한 정리에 의해 정규 분포를 따르기 위해서는 n의 개수가 30개 이상은 되어야 한다.

# 표준 정규 분포(z 분포, t 분포, f 분포)
표준정규분포(z-분포), t-분포, 카이제곱분포, F-분
포 등 수학적으로 증명된 분포(표)가 이미 존재하고 있으며 우리가 얻어낸 통계량을 적절한 분포(표)에
대조하여 관측치의 크고 작음을 판별할 수 있다.
- 종이 뾰족하면 t 납작하면 z 분포
- 카이 제곱과 아노바는 왼쪽으로 치우진 형태를 갖는다.

# 왜도(감마 1)와 첨도(감사 2)
- 왜도는 좌 우로 치우는 것
- 첨도는 완만하냐 뾰죡하냐(데이터 분포를 확인)

# 귀무가설, 대립 가설

# 양측 검정
좌측에 0.25
우측에 0.25

고기가 150g인데 145g으로 나왔을 경우
-> 