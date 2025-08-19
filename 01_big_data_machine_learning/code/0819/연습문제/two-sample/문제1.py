# [two-sample t 검정 : 문제1] 

# 다음 데이터는 동일한 상품의 포장지 색상에 따른 매출액에 대한 자료이다. 

# 포장지 색상에 따른 제품의 매출액에 차이가 존재하는지 검정하시오.

#    blue : 70 68 82 78 72 68 67 68 88 60 80

#    red : 60 65 55 58 67 59 61 68 77 66 66

import pandas as pd
import two_sample

blue = pd.Series([70, 68, 82, 78, 72, 68, 67, 68, 88, 60, 80])
red = pd.Series([60, 65, 55, 58, 67, 59, 61, 68, 77, 66, 66])

two_sample.two_sample(blue, red)