# 대응표본[동일표본] T검정 [쌍체 검정]
# 서로대응인두집단의평균차이검정(pairedsamplest-test)
# 처리이전과처리이후를각각의모집단으로판단하여,동일한관찰대상으로부터처리이전과처리이후를1:1로대응시킨두집단으로부터
# 의표본을대응표본(pairedsample)이라고한다.
# 대응인두집단의평균비교는동일한관찰대상으로부터처리이전의관찰과이후의관찰을비교하여영향을미친정도를밝히는데주로사용
# 하고있다.집단간비교가아니므로등분산검정을할필요가없다. 
# 하나의 집단에 대해 독립변수를 적용하기 전과 후 종속변수의 수준을 측정하고 
# 이들의 차이가 통계적으로 유의한가를 분석한다. 
# 집단 간 분석이 아니므로 등분산 검정은 안 한다 (집단이 2개이상은 돼야 등분산 함..) 
# 예) 광고전후의 상품선호도 측정.. 

import pandas as pd
import numpy as np
import scipy.stats as stats

#region 실습1
print('-----------실습1-----------') 
# 4강의실 학생들을 대상으로 특강이 시험점수에 영향을 준다
# 가설검정
# 귀무: 특강 전후의 시험점수 평균은 차이가 없다
# 대립: 특강 전후의 시험점수 평균은 차이가 있다
np.random.seed(123)
x1=np.random.normal(75,10,30)
x2=np.random.normal(80,10,30)
# print(x1,'',np.mean(x1)) # 75.4471
# print(x2,'',np.mean(x2)) # 81.414852  
#정규성 검정해주기
#print(stats.shapiro(x1),'',stats.shapiro(x2))  
# ShapiroResult(statistic=np.float64(0.9621369983724306), pvalue=np.float64(0.35087442733137086))  
# ShapiroResult(statistic=np.float64(0.9874832681735088), pvalue=np.float64(0.9716624767423043))
# paired 테스트
print(stats.ttest_rel(x1,x2))
# TtestResult(statistic=np.float64(-1.7949604825202188), pvalue=np.float64(0.08309005008718207), df=np.int64(29))
# 결론 귀무 채택  
#endregion 실습1

print('-----------실습2-----------') 
# 복부수술전후 9명의몸무게변화 
# 귀무: 복부수술전후 몸무게 차이가 없다
# 대립: 복부수술전후 몸무게 차이가 있다
baseline = [67.2, 67.4, 71.5, 77.6, 86.0, 89.1, 59.5, 81.9, 105.5]
follow_up = [62.4, 64.6, 70.4, 62.6, 80.1, 73.2, 58.2, 71.0, 101.0]
#print(np.mean(baseline),'',np.mean(follow_up)) # 78.4  71.5
paired_sample=stats.ttest_rel(baseline,follow_up)
print(paired_sample) 
# TtestResult(statistic=np.float64(3.6681166519351103), pvalue=np.float64(0.006326650855933662), df=np.int64(8))
# 귀무 기각

# [대응표본 t 검정 : 문제4]
# 어느 학급의 교사는 매년 학기 내 치뤄지는 시험성적의 결과가 실력의 차이없이 비슷하게 
# 유지되고 있다고 말하고 있다. 이 때, 올해의 해당 학급의 중간고사 성적과 기말고사 성적은 다음과 같다. 
# 점수는 학생 번호 순으로 배열되어 있다.
#    중간 : 80, 75, 85, 50, 60, 75, 45, 70, 90, 95, 85, 80
#    기말 : 90, 70, 90, 65, 80, 85, 65, 75, 80, 90, 95, 95
# 그렇다면 이 학급의 학업능력이 변화했다고 이야기 할 수 있는가?