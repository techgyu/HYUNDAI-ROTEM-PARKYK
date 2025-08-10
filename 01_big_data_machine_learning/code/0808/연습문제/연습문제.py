import MySQLdb
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle

# 문제를 처리할 수 있는 선택지 
# - 처리 방법 1: DB에서 쿼리문으로 필요한 데이터만 골라서 처리하기 -> 복잡하지만, 대규모 처리에 적합
# - 처리 방법 2: DB 테이블을 그대로 가져와서 판다스로 처리하기 -> 직관적이나, 대규모 처리에 적합하지 않음

# 문제 7
#  a) MariaDB에 저장된 jikwon, buser, gogek 테이블을 이용하여 아래의 문제에 답하시오.
#  a -1) 사번 이름 부서명 연봉, 직급을 읽어 DataFrame을 작성
#  a -2) DataFrame의 자료를 파일로 저장
#  a -3) 부서명별 연봉의 합, 연봉의 최대/최소값을 출력
#  a -4) 부서명, 직급으로 교차 테이블(빈도표)을 작성(crosstab(부서, 직급))
#  a -5) 직원별 담당 고객자료(고객번호, 고객명, 고객전화)를 출력. 담당 고객이 없으면 "담당 고객  X"으로 표시
#  a -6) 부서명별 연봉의 평균으로 가로 막대 그래프를 작성

#  b) MariaDB에 저장된 jikwon 테이블을 이용하여 아래의 문제에 답하시오.
#  b -1) pivot_table을 사용하여 성별 연봉의 평균을 출력
#  b -2) 성별(남, 여) 연봉의 평균으로 시각화 - 세로 막대 그래프
#  b -3) 부서명, 성별로 교차 테이블을 작성 (crosstab(부서, 성별))

#  c) 키보드로 사번, 직원명을 입력받아 로그인에 성공하면 console에 아래와 같이 출력하시오.
#  c-1) 조건 :  try ~ except MySQLdb.OperationalError as e:      사용
#  c-2) c-1 충족 시, 전 직원의 [사번  직원명  부서명   직급  부서전화  성별] 출력
#  c-3) c-1 충족 시, 적 직원의 [인원수 : * 명] 출력

# 데이터 베이스 구조
# buser: buserno[부서번호], busername[부서명], buserloc[부서위치], busertel[부서전화]
# gogek: gogekno[고객번호], gogekname[고객명], gogektel[고객전화], gogekjumin[고객주민번호],  gogekdamsano[담당사원번호]
# jikwon: jikwonno[사원번호], jikwonname[사원명], busernum[부서번호], jikwonjik[직급], jikwonpay[연봉], jikwonibsail[입사일], jikwongen[성별], jikwonrating[평가]]
# sangdata: sangno[상품번호], sangname[상품명], sangprice[가격], sangstock[재고량]

# 1. 폰트 깨짐 방지 설정
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False

# 2. db configure 파일 설정
try:
    with open('./01_big_data_machine_learning/data/mymaria.dat', 'rb') as obj:
        config = pickle.load(obj)
except Exception as e:
    print("읽기 오류:", e)
    sys.exit()

# 3. 데이터베이스 연결: 외부 데이터 연결 부분에만 try except로 처리
try:
    conn = MySQLdb.connect(**config)
    # buser
    buser = conn.cursor()
    sql = "select * from buser"
    buser.execute(sql)
    # gogek
    gogek = conn.cursor()
    sql = "select * from gogek"
    gogek.execute(sql)
    # jikwon
    jikwon = conn.cursor()
    sql = "select * from jikwon"
    jikwon.execute(sql)
    # sangdata
    sangdata = conn.cursor()
    sql = "select * from sangdata"
    sangdata.execute(sql)
except Exception as e:
    print("SQL 로딩 오류:", e)
    sys.exit()

# 4. 각각의 테이블을 DataFrame으로 변환
buser_df = pd.DataFrame(buser.fetchall(), columns=[i[0] for i in buser.description])
gogek_df = pd.DataFrame(gogek.fetchall(), columns=[i[0] for i in gogek.description])
jikwon_df = pd.DataFrame(jikwon.fetchall(), columns=[i[0] for i in jikwon.description])
sangdata_df = pd.DataFrame(sangdata.fetchall(), columns=[i[0] for i in sangdata.description])

# 5. 출력 확인
print("buser DataFrame:\n", buser_df)
print("gogek DataFrame:\n", gogek_df)
print("jikwon DataFrame:\n", jikwon_df)
print("sangdata DataFrame:\n", sangdata_df)

#  a -1) 사번 이름 부서명 연봉, 직급을 읽어 DataFrame을 작성
filed_df = pd.merge(jikwon_df, buser_df, left_on='busernum', right_on='buserno', how='left')
filed_df = pd.DataFrame({
    '사번': filed_df['jikwonno'],
    '이름': filed_df['jikwonname'],
    '부서명': filed_df['busername'],
    '연봉': filed_df['jikwonpay'],
    '직급': filed_df['jikwonjik']
})

# 출력 확인
print("직원 DataFrame:\n", filed_df)

#  a -2) DataFrame의 자료를 파일로 저장
# csv 파일로 저장
filed_df.to_csv('./01_big_data_machine_learning/data/employee_data.csv', index=False, encoding='utf-8-sig')
# excel 파일로 저장
filed_df.to_excel('./01_big_data_machine_learning/data/employee_data.xlsx', index=False)
# json 파일로 저장
filed_df.to_json('./01_big_data_machine_learning/data/employee_data.json', orient='records', force_ascii=False)

#  a -3) 부서명별 연봉의 합, 연봉의 최대/최소값을 출력
# busername[부서명]과 jikwonpay[연봉]가 들어간 새로운 DataFrame을 생성
pay_by_buser_df = pd.merge(jikwon_df, buser_df, left_on='busernum', right_on='buserno', how='left')
pay_by_buser_df = pd.DataFrame({
    'busername' : pay_by_buser_df['busername'],
    'jikwonpay' : pay_by_buser_df['jikwonpay']
})
# 출력 확인
print("부서, 연봉 DataFrame:\n", pay_by_buser_df)
print("\n부서별 연봉 총액:\n", pay_by_buser_df.groupby('busername')['jikwonpay'].sum())
print("\n부서별 연봉 최고 액수:\n", pay_by_buser_df.groupby('busername')['jikwonpay'].max())
print("\n부서별 연봉 최저 액수:\n", pay_by_buser_df.groupby('busername')['jikwonpay'].min())

# a -4) 부서명, 직급으로 교차 테이블(빈도표)을 작성(crosstab(부서, 직급))
print("\n부서 별 직급 교차 표")
cross_table_busername_jikwonjik = pd.merge(buser_df, jikwon_df, left_on='buserno', right_on='busernum', how='left')
ctab = pd.crosstab(cross_table_busername_jikwonjik['busername'], cross_table_busername_jikwonjik['jikwonjik'], margins=True)
print(ctab)

# a -5) 직원별 담당 고객자료(고객번호, 고객명, 고객전화)를 출력. 담당 고객이 없으면 "담당 고객  X"으로 표시
client_per_manager = pd.merge(jikwon_df, gogek_df, left_on='jikwonno', right_on='gogekdamsano', how='left')
client_per_manager = pd.DataFrame({
    '직원명' : client_per_manager['jikwonname'],
    '고객번호' : client_per_manager['gogekno'],
    '고객명': client_per_manager['gogekname'],
    '고객전화': client_per_manager['gogektel']
}).fillna("담당 고객 X")
print(client_per_manager)

# a -6) 부서명별 연봉의 평균으로 가로 막대 그래프를 작성
mean_by_buser = pd.merge(buser_df, jikwon_df, left_on='buserno', right_on='busernum', how='left')
print(mean_by_buser)
mean_by_buser = mean_by_buser.groupby('busername')['jikwonpay'].mean()
print(mean_by_buser)

plt.figure(figsize=(13, 6))
plt.barh(list(mean_by_buser.index), list(mean_by_buser.values))
plt.title('부서명별 연봉의 평균')
plt.xlabel('부서명')
plt.ylabel('연봉')
plt.grid()
plt.show()

# b -1) pivot_table을 사용하여 성별 연봉의 평균을 출력
mean_by_sex = pd.pivot_table(jikwon_df, index='jikwongen', values='jikwonpay', aggfunc='mean')
print("\n성별 연봉 평균:\n", mean_by_sex)

# b -2) 성별(남, 여) 연봉의 평균으로 시각화 - 세로 막대 그래프
plt.figure(figsize=(13, 6))
plt.bar(list(mean_by_sex.index), list(mean_by_sex['jikwonpay'].values))
plt.title('성별 연봉 평균')
plt.xlabel('성별')
plt.ylabel('연봉')
plt.grid()
plt.show()

# b -3) 부서명, 성별로 교차 테이블을 작성 (crosstab(부서, 성별))
print("\n부서, 성별 교차 테이블")
cross_table_busername_sex = pd.merge(buser_df, jikwon_df, left_on='buserno', right_on='busernum', how='left')
cross_table_busername_sex = pd.crosstab(cross_table_busername_sex['busername'], cross_table_busername_sex['jikwongen'], margins=True)
print(cross_table_busername_sex)

#  c-1) 조건: try ~ except MySQLdb.OperationalError as e: 사용하여 로그인 구현
try:
    employee_num = input("사번: ")
    employee_name = input("직원명: ")
    # 로그인 처리용 df 생성
    login_df = pd.merge(buser_df, jikwon_df, left_on='buserno', right_on='busernum', how='left')
    # 로그인 정보 확인
    matched = login_df[(login_df['jikwonno'] == int(employee_num)) & (login_df['jikwonname'] == employee_name)]

    if not matched.empty:
        # 전 직원 정보 한글 컬럼명으로 출력
        print(
            login_df[['jikwonno', 'jikwonname', 'busername', 'jikwonjik', 'busertel', 'jikwongen']]
        .rename(columns={
            'jikwonno': '사번',
            'jikwonname': '직원명',
            'busername': '부서명',
            'jikwonjik': '직급',
            'busertel': '부서전화',
            'jikwongen': '성별'
        })
        .to_string(index=False)
        )
        print("인원수 : ", len(login_df))

except MySQLdb.OperationalError as e:
    print("MySQL OperationalError:", e)
finally:
    pass