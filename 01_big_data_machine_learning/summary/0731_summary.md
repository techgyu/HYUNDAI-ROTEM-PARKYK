# 07/31 Numpy와 Python을 활용한 배열 및 통계 기초 요약

## 1. Python으로 직접 구현한 통계 함수

- 합계, 평균, 분산, 표준편차를 직접 함수로 구현할 수 있다.
- 예시:
  ```python
  def grades_sum(grades):
      return sum(grades)
  def grades_ave(grades):
      return grades_sum(grades) / len(grades)
  def grades_varience(grades):
      ave = grades_ave(grades)
      return sum((x - ave) ** 2 for x in grades) / len(grades)
  def grades_std(grades):
      return grades_varience(grades) ** 0.5
  ```
- 결과:  
  Sum: 6, Average: 1.5, Variance: 6.25, Standard Deviation: 2.5

## 2. Numpy를 활용한 통계 연산

- Numpy의 내장 함수로 합계, 평균, 분산, 표준편차를 간단하게 계산할 수 있다.
  - `np.sum(grades)`, `np.mean(grades)`, `np.var(grades)`, `np.std(grades)`
- 결과는 직접 구현과 동일하며, 대용량 데이터에서 훨씬 빠르다.

## 3. Numpy 배열 생성과 기초 연산

- 다양한 방식으로 배열 생성:  
  - `np.array`, `np.zeros`, `np.ones`, `np.full`, `np.eye`, `np.arange`, `np.random.rand`, `np.random.randn`
- 배열의 데이터 타입, shape, ndim, size 등 다양한 속성 확인 가능

## 4. 배열 인덱싱과 슬라이싱

- 1차원/2차원 배열에서 인덱싱, 슬라이싱, 복사, 뷰(view) 개념
- 행/열 단위로 슬라이싱, 불리언 인덱싱 등 다양한 추출 방법
  - 예시:
    ```python
    a = np.array([[1,2,3],[4,5,6],[7,8,9]])
    r1 = a[1, :]      # [4 5 6]
    r2 = a[1:2, :]    # [[4 5 6]]
    c1 = a[:, 1]      # [2 5 8]
    c2 = a[:, 1:2]    # [[2],[5],[8]]
    bool_idx = a >= 5 # [[False ...],[False True True],[True True True]]
    a[bool_idx]       # [5 6 7 8 9]
    ```

## 5. 배열 연산 및 브로드캐스팅

- 배열 간 덧셈, 뺄셈, 곱셈, 나눗셈 등 요소별 연산
  - `x + y`, `np.add(x, y)`, `np.subtract(x, y)`, `np.multiply(x, y)`, `np.divide(x, y)`
- 행렬 곱셈(내적): `np.dot(x, y)`, `x.dot(y)`
- 브로드캐스팅을 통해 shape이 맞지 않아도 연산 가능

## 6. Numpy의 연산 속도

- 파이썬 내장 sum과 numpy sum의 속도 비교: numpy가 훨씬 빠름
- 대용량 데이터 처리에 numpy가 필수적임
