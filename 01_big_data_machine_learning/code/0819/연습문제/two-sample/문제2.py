# [two-sample t 검정 : 문제2]  

# 아래와 같은 자료 중에서 남자와 여자를 각각 15명씩 무작위로 비복원 추출하여 혈관 내의 
# 콜레스테롤 양에 차이가 있는지를 검정하시오.

# 대립: 남자와 여자의 콜레스테롤 양에 차이가 있다.
# 귀무: 남자와 여자의 콜레스테롤 양이 차이가 없다.

import pandas as pd
import numpy as np
import scipy.stats as stats
import two_sample

male = pd.DataFrame([0.9, 2.2, 1.6, 2.8, 4.2, 3.7, 2.6, 2.9, 3.3, 1.2, 3.2, 2.7, 3.8, 4.5, 4, 2.2, 0.8, 0.5, 0.3, 5.3, 5.7, 2.3, 9.8])
female = pd.DataFrame([1.4, 2.7, 2.1, 1.8, 3.3, 3.2, 1.6, 1.9, 2.3, 2.5, 2.3, 1.4, 2.6, 3.5, 2.1, 6.6, 7.7, 8.8, 6.6, 6.4])

male = male.sample(15, replace=False)
female = female.sample(15, replace=False)

two_sample.two_sample(male, female)
