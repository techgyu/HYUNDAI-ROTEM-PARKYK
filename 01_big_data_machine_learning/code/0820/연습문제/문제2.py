# [ANOVA 예제 2]
# DB에 저장된 buser와 jikwon 테이블을 이용하여 총무부, 영업부, 전산부, 
# 관리부 직원의 연봉의 평균에 차이가 있는지 검정하시오. 만약에 연봉이 없는 직원이 있다면 작업에서 제외한다.

# 대립가설(H1): 총무부, 영업부, 전산부, 관리부 직원의 연봉의 평균에 차이가 있다.
# 귀무가설(H0): 총무부, 영업부, 전산부, 관리부 직원의 연봉의 평균에 차이가 없다.

import MySQLdb
import pickle
import pandas as pd
import sys
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt

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

    # 결측치 제거
    df = df.dropna(subset=['jikwonpay'])
    
    # 부서별 연봉 평균
    gad_pay = df[df['busername'] == '총무부']['jikwonpay']
    sales_pay = df[df['busername'] == '영업부']['jikwonpay']
    it_pay = df[df['busername'] == '전산부']['jikwonpay']
    admin_pay = df[df['busername'] == '관리부']['jikwonpay']

    # 정규성 검정
    print(stats.shapiro(gad_pay).pvalue) # 0.026 < 0.05 정규성 불만족
    print(stats.shapiro(sales_pay).pvalue) # 0.025 < 0.05 정규성 불만족
    print(stats.shapiro(it_pay).pvalue) # 0.419 > 0.05 정규성 만족
    print(stats.shapiro(admin_pay).pvalue) # 0.907 > 0.05 정규성 만족

    # 등분산성 검정
    print(stats.levene(gad_pay, sales_pay, it_pay, admin_pay).pvalue) # 0.798 > 0.05 등분산성 만족
    print(stats.bartlett(gad_pay, sales_pay, it_pay, admin_pay).pvalue) # 0.629 > 0.05 등분산성 만족

    # 일원분산분석
    print(stats.f_oneway(gad_pay, sales_pay, it_pay, admin_pay).pvalue) # 0.745 > (유의 수준) 0.05 (귀무가설 채택)

    # ANOVA 사후 검정(Post-hoc test)
    # 분산 분석은 집단의 평균의 차이 여부만 알려줄 뿐, 
    # 각 집단 간의 평균 차이는 알려주지 않는다.
    # 각 집단 간의 평균 차이를 확인하기 위해 사후 검정 실시

    # Tukey의 사후 검정(집단별 평균 차이 유의성 확인)
    turResult = pairwise_tukeyhsd(endog=df.jikwonpay, groups=df.busername)
    print(turResult)  # 각 집단 쌍별로 평균 차이, p-value, 유의성 결과 출력

    # 사후 검정 결과를 시각화 (신뢰구간 그래프)
    turResult.plot_simultaneous(xlabel='mean', ylabel='group')
    plt.show()        # 그래프 창 표시
    plt.close()       # 그래프 창 닫기

    # 결론: 총무부, 영업부, 전산부, 관리부 직원의 연봉의 평균에 차이가 없다.
except Exception as e:
    print("SQL 실행 오류:", e)
finally:
    conn.close()

 
