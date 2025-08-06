import pandas as pd

pd.set_option('display.max_rows', None)    # 행 모두 출력
pd.set_option('display.max_columns', None) # 열 모두 출력

titanic_data = pd.read_csv('./code/0806/연습문제/titanic_data.csv')
print("Read한 titanic_data:", titanic_data.head(40))  # 처음 10개 행 출력

#   열 구성 정보
#     Survived : 0 = 사망, 1 = 생존
#     Pclass : 1 = 1등석, 2 = 2등석, 3 = 3등석
#     Sex : male = 남성, female = 여성
#     Age : 나이
#     SibSp : 타이타닉 호에 동승한 자매 / 배우자의 수
#     Parch : 타이타닉 호에 동승한 부모 / 자식의 수
#     Ticket : 티켓 번호
#     Fare : 승객 요금
#     Cabin : 방 호수
#     Embarked : 탑승지, C = 셰르부르, Q = 퀸즈타운, S = 사우샘프턴



# 1) 데이터프레임의 자료로 나이대(소년, 청년, 장년, 노년)에 대한 생존자수를 계산한다.
# - cut() 함수 사용
# - bins = [1, 20, 35, 60, 150]
# - labels = ["소년", "청년", "장년", "노년"]

# 나이대 정리
# 소년: 1 ~ 20세
# 청년: 21 ~ 35세
# 장년: 36 ~ 60세
# 노년: 61세 이상

# AgeGroup 추가
titanic_data['AgeGroup'] = pd.cut(titanic_data['Age'], bins=[1, 20, 35, 60, 150], labels=["소년", "청년", "장년", "노년"], right=False)

# 나이대별 생존자 수 계산
age_group_survival = titanic_data.groupby('AgeGroup')['Survived'].sum()
print("나이대별 생존자 수:", age_group_survival)



# 2) 성별 및 선실에 대한 자료를 이용해서 생존여부(Survived)에 대한 생존율을 피봇테이블 형태로 작성한다.
# - df.pivot_table()
    #     values=None,      # 집계할 데이터(컬럼) 이름, 생략하면 모든 수치형 컬럼
    #     index=None,       # 행 인덱스로 사용할 컬럼명(들)
    #     columns=None,     # 열 인덱스로 사용할 컬럼명(들)
    #     aggfunc='mean',   # 집계 함수(평균, 합계 등), 기본값은 'mean'
    #     fill_value=None,  # 결측치(NaN) 대신 채울 값
    #     margins=False     # 전체(합계) 행/열 추가 여부
#   )
# - index에는 성별(Sex)를 사용하고, column에는 선실(Pclass) 인덱스를 사용한다.

temp =  titanic_data.pivot_table(
        values='Survived',    # 집계할 값
        index='Sex',          # 행: 성별
        columns='Pclass',     # 열: 선실등급
        aggfunc='mean'        # 집계함수: 평균(생존율)
    )

print(temp)


# index에는 성별(Sex) 및 나이(Age)를 사용하고, column에는 선실(Pclass) 인덱스를 사용한다.
# 출력 결과 샘플2 : 위 결과물에 Age를 추가. 백분율로 표시. 소수 둘째자리까지.    예: 92.86

temp = titanic_data.pivot_table(
        values='Survived',    # 집계할 값
        index=['Sex', 'AgeGroup'],  # 행: 성별 및 나이
        columns='Pclass',     # 열: 선실등급
        aggfunc='mean',       # 집계함수: 평균(생존율)
        fill_value=0          # 결측치(NaN) 대신 0으로 채움
    )

print(temp)




# 6 - 1) human.csv 파일을 읽어 아래와 같이 처리하시오.
    #  - Group이 NA인 행은 삭제
    #  - Career, Score 칼럼을 추출하여 데이터프레임을 작성
    #  - Career, Score 칼럼의 평균계산

    #  참고 : strip() 함수를 사용하면 주어진 문자열에서 양쪽 끝에 있는 공백과 \n 기호를 삭제시켜 준다. 
    #          그래서 위의 문자열에서 \n과 오른쪽에 있는 공백이 모두 사라진 것을 확인할 수 있다. 
    #          주의할 점은 strip() 함수는 문자열의 양 끝에 있는 공백과 \n을 제거해주는 것이지 중간에 
    #          있는 것까지 제거해주지 않는다.

human_data = pd.read_csv('./code/0806/연습문제/human.csv')

# 칼럼 이름 만 공백 제거
human_data.columns = human_data.columns.str.strip()

# 전체 칼럼 순회 공백 제거
for col in human_data.columns:
    if human_data[col].dtype == 'object': #strip() 함수는 object에만 사용 가능, 아니면 오류 발생
        human_data[col] = human_data[col].str.strip()

print("Read한 human_data:\n", human_data.head(10))  # 처음 10개 행 출력
# Group이 NA인 행은 삭제
human_data = human_data.drop(human_data[human_data['Group'] == 'NA'].index)
print("Group이 NA인 행 삭제 후:\n", human_data.head(10))  # 처음 10개 행 출력

#  - Career, Score 칼럼을 추출하여 데이터프레임을 작성
dataframe = pd.DataFrame(human_data[['Group', 'Career', 'Score']])
print(dataframe)

#  - Career, Score 칼럼의 평균계산
dataframe = dataframe[['Career', 'Score']].mean()
print(dataframe)

# 6 - 2) tips.csv 파일을 읽어 아래와 같이 처리하시오.

#      - 파일 정보 확인
#      - 앞에서 3개의 행만 출력
#      - 요약 통계량 보기
#      - 흡연자, 비흡연자 수를 계산  : value_counts()
#      - 요일을 가진 칼럼의 유일한 값 출력  : unique()
#           결과 : ['Sun' 'Sat' 'Thur' 'Fri']