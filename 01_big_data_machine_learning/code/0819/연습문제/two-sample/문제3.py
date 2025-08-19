# [two-sample t 검정 : 문제3]

# DB에 저장된 jikwon 테이블에서 총무부, 영업부 직원의 연봉의 평균에 차이가 존재하는지 검정하시오.

# 연봉이 없는 직원은 해당 부서의 평균연봉으로 채워준다.

import MySQLdb
import pickle
import pandas as pd
import sys
import scipy.stats as stats
import two_sample

try:
    with open('./01_big_data_machine_learning/data/mymaria.dat', 'rb') as obj:
        config = pickle.load(obj)

except Exception as e:
    print("읽기 오류:", e)
    sys.exit()

try:
    # 데이터 읽어오기 
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor()

    sql = """
    select a.jikwonpay, b.busername
    from jikwon a inner join buser b
    on a.busernum=b.buserno
    """
    cursor.execute(sql)

    # 데이터 프레임 생성
    df = pd.DataFrame(cursor.fetchall(),
                    columns=['jikwonpay', 'busername'])
    
    # 형 변환
    df['jikwonpay'] = df['jikwonpay'].astype(float)
    
    # 부서별 연봉 평균
    gad_pay = df[df['busername'] == '총무부']['jikwonpay']
    sales_pay = df[df['busername'] == '영업부']['jikwonpay']

    two_sample.two_sample(gad_pay, sales_pay)

except Exception as e:
    print("SQL 실행 오류:", e)
finally:
    conn.close()