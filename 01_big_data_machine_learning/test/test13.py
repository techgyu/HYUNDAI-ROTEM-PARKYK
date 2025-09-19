# 델	
# [문항13] 다음 데이터는 어느 교육센터에서 실시하고 있는 파이썬 과정 중 두 명의 강사에 따른 성적에 대한 자료이다. 
# 강사에 따라 성적에 차이가 있는지 평균 차이 검정을 하시오. (배점:10)
# 강사1 : 71 58 92 78 71 68 67 88 88 60 80 70 68 82 78
# 강사2 : 50 65 75 91 67 39 81 68 97 86 66 60 65 55 58

# ① 귀무가설 : 강사에 따라 수업 성적 차이가 없다.
# ② 대립가설 : 강사에 따라 수업 성적 차이가 있다
# ③ 검정을 위한 소스 코드
# 답안 :
# ANOVA 검정을 사용
import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols #최소 제곱 법(기울과 절편을 구할 수 있다 -> 직선을 구함 -> 회귀 분석에서 중요함!)
import matplotlib.pyplot as plt
import statsmodels.api as sm

data = pd.DataFrame({
    'teacher': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    'score': [71, 58, 92, 78, 71, 68, 67, 88, 88, 60, 80, 70, 68, 82, 78, 50, 65, 75, 91, 67, 39, 81, 68, 97, 86, 66, 60, 65, 55, 58]
})

print(data)

reg = ols("data['score'] ~ C(data['teacher'])", data=data).fit()

table = sm.stats.anova_lm(reg, type=2) # anova linear regression model 생성
print(table) # p-value: 0.0.19511 > 0.05 이므로 귀무 채택

# 결론: 강사에 따라 수업 성적 차이가 없다.