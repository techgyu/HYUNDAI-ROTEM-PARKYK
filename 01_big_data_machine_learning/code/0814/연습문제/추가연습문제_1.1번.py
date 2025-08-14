import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
# 부모학력 수준이 자녀의 진학여부와 관련이 있는가를 가설설정하기
# 대립가설(HI) : 부모의 학력 수준과 자녀의 진학 여부는 서로 관련이 있다.
# 귀무가설(H0) : 부모의 학력 수준과 자녀의 진학 여부는 서로 관련이 없다.

sdata = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/cleanDescriptive.csv")
# print(data.head(3))
ctab = pd.crosstab(index=sdata['pass'],columns=sdata['level'],dropna=False)  # 교차분석을 위한 교차표 생성

# print(ctab)
ctab.columns = ['레벨1','레벨2','레벨3','NAN']
ctab.index = pd.Index(['실패', '합격', 'NAN'])
# print(ctab)

chi2, p, df, expected = scipy.stats.chi2_contingency(ctab)  # 카이제곱 검정 수행
print("카이제곱 통계량 : ", chi2)
print("p-value : ", p)
print("자유도 : ", df)
print("기대도 : ", expected)

# p-value >0.05 이므로 귀무가설을 기각할 수 없다. : 
# p-value는 우연의 확률 
# 즉, "부모학력이 자녀의 진학여부에 관련이 있다" 라는 주장이 우연이라고 나올 확률이기 때문에
# 이 확률이 0.05를 넘는다는 것은 아 존나 우연이다 라고 말해주는 것
# 우연이다->관련없다->귀무가설 기각 x

