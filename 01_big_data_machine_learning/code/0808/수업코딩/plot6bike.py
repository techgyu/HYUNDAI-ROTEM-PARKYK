# 자전거 공유 시스템(워싱턴 D.C.) 관련 파일로 시각화
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno # 결측치 시각화 라이브러리
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False


# -------------- 로그, 출력 설정 --------------

# 로그 선택 문(True, False)
LOG_TRAIN_SUMMARY = False  # train 데이터 요약 정보 출력 여부
LOG_TRAIN_SPECIFIC = False  # train 데이터 세부 데이터 부분 로그 출력 여부
LOG_TRAIN_CHECK_NULL = False # train 데이터 결측치 확인 여부
LOG_3_TRAIN_SPECIFIC = False # 연월연시 데이터 생성 후 train 데이터 세부 데이터 부분 로그 출력 여부

# 그래프 출력 여부(True, False)
GRAPH_MATRIX = False # 결측치 시각화 여부
GRAPH_BAR = False # 막대 그래프 시각화 여부
GRAPH_DATETIME = False  # 연월연시 데이터로 시각화 여부
GRAPH_BOXPLOT = False # Boxplot 시각화 여부 - 대여량 - 계절별, 시간별 근무일 여부에 따른 대여량

# -------------- 로그, 출력 설정 --------------


plt.style.use('ggplot') # 시각화 스타일 설정

# 1. 데이터 수집 후 가공(EDA) - train.csv
train = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/data/train.csv', 
                    parse_dates=['datetime']) # datetime 컬럼을 datetime 형식(판다스의 날짜/시간 자료형)으로 변환, 원래 str 형식으로 저장

# 1에서 수집한 train 파일에 대한 내용 확인용 코드
if LOG_TRAIN_SUMMARY:
    print("\n데이터 형태:\n", train.shape) # 데이터의 행과 열의 개수
    print("\n컬럼명:\n", train.columns) # 컬럼의 이름
    print("\n정보:") # 데이터의 정보
    train.info()
    print("\n기술 통계:\n", train.describe()) # 각 컬럼의 기술 통계

if LOG_TRAIN_SPECIFIC:
    print("\n상위 10개 데이터:\n", train.head(10))
    print("\n하위 10개 데이터:\n", train.tail(10))

if LOG_TRAIN_CHECK_NULL:
    print("\n결측치 정보:\n", train.isnull().sum())

# 2. 결측치 시각화(그래프)
if GRAPH_MATRIX:
    msno.matrix(train)
    plt.show()

if GRAPH_BAR:
    msno.bar(train)
    plt.show()

# 3. 연월연시 데이터 생성
    train['year'] = train['datetime'].dt.year
    train['month'] = train['datetime'].dt.month
    train['day'] = train['datetime'].dt.day
    train['hour'] = train['datetime'].dt.hour
    train['minute'] = train['datetime'].dt.minute
    train['second'] = train['datetime'].dt.second

if LOG_3_TRAIN_SPECIFIC:
    print(train.columns) # 컬럼명 확인
    print(train.head()) # 상위 5개 데이터 확인

# 4. 연월연시 데이터로 시각화
if GRAPH_DATETIME:
    figure, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4) # 1행 4열의 서브플롯 생성
    figure.set_size_inches(15, 5) # 서브플롯 크기 설정
    sns.barplot(data = train, x = 'year', y = 'count', ax = ax1) # 연도별 자전거 대여량
    sns.barplot(data = train, x = 'month', y = 'count', ax = ax2) # 월별 자전거 대여량
    sns.barplot(data = train, x = 'day', y = 'count', ax = ax3) # 일별 자전거 대여량
    sns.barplot(data = train, x = 'hour', y = 'count', ax = ax4) # 시간별 자전거 대여량
    ax1.set(ylabel='건수', title='연도별 자전거 대여량')
    ax2.set(ylabel='건수', title='월별 자전거 대여량')
    ax3.set(ylabel='건수', title='일별 자전거 대여량')
    ax4.set(ylabel='건수', title='시간별 자전거 대여량')
    plt.show()

# 5. Boxplot 시각화 - 대여량 - 계절별, 시간별 근무일 여부에 따른 대여량
# - Boxplot: x - 분류 기준, y - 값을 주면, 중앙값, 사분위수(Q1, Q3), IQR, 이상치 등을 파악하여 박스플롯을 그려줌
# - 박스의 의미: 하위 25% ~ 상위 75% 까지의 값의 범위
# - 박스 안 굵은 선: 순서대로 값을 나열했을 때 가운데 값
# - 박스 밖의 점들: 데이터 분포에서 비정상적으로 튀는 값
# - 박스 아래 / 위의 선: 
#   - 각 수염(whisker)은 박스(1사분위수 Q1 ~ 3사분위수 Q3) 바깥쪽으로 뻗어 있는 선입니다.
#   - 아래쪽 수염은 Q1 - 1.5×IQR보다 크거나 같은 값 중에서 가장 작은 값까지,
#   - 위쪽 수염은 Q3 + 1.5×IQR보다 작거나 같은 값 중에서 가장 큰 값까지를 연결합니다.
#   - 즉, 수염은 이상치가 아닌 데이터의 최소값과 최대값을 나타냅니다.
#   - 이 범위를 벗어난 값들은 이상치(Outlier)로 간주되어 점(●)으로 따로 표시됩니다.
if GRAPH_BOXPLOT:
    figure, (ax1, ax2) = plt.subplots(nrows=1, ncols=2) # 1행 2열의 서브플롯 생성
    figure.set_size_inches(15, 5) # 서브플롯 크기 설정
    sns.boxplot(data = train, x = 'season', y = 'count', ax = ax1) # 계절을 기준으로한 자전거 대여량
    sns.boxplot(data = train, x = 'workingday', y = 'count', ax = ax2) # 근무일을 기준으로 한 따른 자전거 대여량
    ax1.set(ylabel='건수', title='계절별 자전거 대여량')
    ax2.set(ylabel='건수', title='근무일 여부에 따른 자전거 대여량')
    plt.show()