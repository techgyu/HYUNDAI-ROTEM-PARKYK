# 로그변환, 편차가 큰 데이터를 로그변환하면 분포를 개선하고, 큰 범위 차이를 줄이며,
# 모델이 보다 안정적으로 학습할 수 있도록 만들어주는 장점이 있다.

import numpy as np
np.set_printoptions(suppress=True, precision=4)  # 소수점 이하 자리수를 줄여서 출력
from typing import Optional

# 로그 변환을 통해 데이터의 분포를 개선하고, 범위 차이를 줄이는 함수
def test():
    values = np.array([3.45, 34.5, 0.345, 0.01, 0.1, 10, 100, 1000])
    print("원본 자료 로그 변환 결과:")
    print("log(이진 로그):\n", np.log2(values)) # 이진 로그
    print("log(상용 로그): \n", np.log10(values)) # 상용 로그
    print("log(자연 로그): \n", np.log(values)) # 자연 로그

    # 자연 로그는 왜 ln이라고 부르나요?
    # 자연 로그(natural logarithm)는 밑이 e(약 2.718)를 사용하는 로그입니다.
    # ln은 라틴어 "logarithmus naturalis"의 약자로, 자연 로그를 의미합니다.
    # 수학에서는 log_e(x) 대신 ln(x)로 표기하는 것이 관례입니다.
    print("ln: \n", np.log(values)) # 자연 로그


    # 로그 값의 최소, 최대를 구하는 코드
    print("로그 값의 최소:", np.min(np.log(values)))
    print("로그 값의 최대:", np.max(np.log(values)))

    # log_values 변수 추가
    log_values = np.log(values)

    # 위에서 구한 최소, 최대를 0 ~ 1 사이로 정규화
    # → 정규화(Normalization)는 데이터의 최소값을 0, 최대값을 1로 맞추는 작업입니다.
    #   각 값을 (값 - 최소) / (최대 - 최소)로 계산하면 모든 데이터가 0~1 범위로 변환됩니다.
    normalized = (log_values - np.min(log_values)) / (np.max(log_values) - np.min(log_values))
    print("로그 변환 후 정규화 결과:\n", normalized)


    # 표준화(Standardization)는 데이터의 평균을 0, 표준편차를 1로 맞추는 작업입니다.
    #   각 값을 (값 - 평균) / (표준편차)로 계산하면 평균이 0, 분산이 1인 데이터로 변환됩니다.
    standardized = (log_values - np.mean(log_values)) / np.std(log_values)
    print("로그 변환 후 표준화 결과:\n", standardized)

def log_inverse():
    offset = 1
    # offset은 로그 변환 시 0 이하의 값이 입력되는 것을 방지하기 위해 더해주는 값입니다.
    # 로그 함수는 0 이하의 값에서 정의되지 않으므로, 데이터가 0 또는 음수일 때 offset을 더해 양수로 만들어줍니다.
    print("최초 투입 값: \n", 10 + offset)  # 로그 변환을 위해 10에 offset을 더함
    # np.log() 함수는 자연 로그를 계산합니다.
    # 즉, np.log(x)는 e를 밑으로 하는 로그 값을 반환합니다.
    log_values = np.log(10 + offset) # 로그 변환된 값
    print("로그 변환 결과: \n", log_values)
    original = np.exp(log_values) - offset  # 역변환을 위해 로그 값을 지수 함수로 변환
    # np.exp 함수는 자연상수 e(약 2.718)를 밑으로 하는 지수 함수를 계산합니다.
    # 즉, np.exp(x)는 e^x(=e의 x제곱) 값을 반환합니다.
    # 로그 변환의 역변환(원래 값 복원)에 사용되며,
    # 예를 들어 y = log(x)라면, x = exp(y)로 원래 값을 구할 수 있습니다.
    print("역변환 결과(10이 나오면 정상, 미세 오차 있을 수 있음): \n", original)


# 로그 변환을 위한 클래스 정의
class LogTrans:
    def __init__(self, offset:float=1.0):
        self.offset = offset

    # 로그 변환 수행 메소드
    def transform(self, x:np.ndarray, offset:Optional[float]=None):
        # f(x) = log(x + offset)
        # offset 인자가 주어지면 해당 값을 사용, 아니면 self.offset 사용
        use_offset = self.offset if offset is None else offset
        return np.log(x + use_offset)

    # 로그 역변환 수행 메소드
    def transform_inverse(self, x:np.ndarray, offset:Optional[float]=None):
        # f(x) = exp(x) - offset
        # offset 인자가 주어지면 해당 값을 사용, 아니면 self.offset 사용
        use_offset = self.offset if offset is None else offset
        # np.exp() 함수는 exponential(지수 함수 e)의 약어로, e^x 값을 반환합니다.
        return np.exp(x) - use_offset

def test_logTrans():
    # LogTrans 클래스의 인스턴스 생성
    data = np.array([0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], dtype=np.float64)
    # 로그 변환
    log_trans = LogTrans()
    transformed = log_trans.transform(data, offset=1.0)
    print("로그 변환 결과: \n", transformed)

    # 역변환
    inverse_transformed = log_trans.transform_inverse(transformed, offset=1.0)
    print("역변환 결과(원래 값 복원): \n", inverse_transformed)

if __name__ == "__main__": # 현재 모듈이 메인 프로그램으로 실행될 때만 test() 함수를 호출
    test()
    log_inverse()  # 로그 역변환 함수 호출
    test_logTrans()  # LogTrans 클래스 테스트