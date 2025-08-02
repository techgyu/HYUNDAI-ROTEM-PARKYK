# 기본 통계 함수를 직접 작성: 평균, 분산, 표준 편차
grades = [1, 3, -2, 4]

# 합
def grades_sum(grades):
    tot = 0
    for g in grades:
        tot += g
    return tot

print("Sum:", grades_sum(grades))  # Sum: 6

# 평균
def grades_ave(grades):
    ave = grades_sum(grades) / len(grades)
    return ave

print ("Average:", grades_ave(grades))  # Average: 1.5

# 분산
def grades_varience(grades):
    ave = grades_ave(grades)
    varience = 0
    for su in grades:
        varience += (su - ave) ** 2
    return varience / len(grades)

print("Variance:", grades_varience(grades))  # Variance: 6.25

# 표준 편차
def grades_std(grades):
    varience = grades_varience(grades)
    return (varience ** 0.5)

print("Standard Deviation:", grades_std(grades))  # Standard Deviation: 2.5

print('**' * 10)  # ********************

import numpy as np
print('합은', np.sum(grades))           # 합은 6
print('평균은', np.mean(grades))        # 평균은 1.5
print('평균은', np.average(grades))     # 평균은 1.5
print('분산은', np.var(grades))         # 분산은 6.25
print('표준 편차는', np.std(grades))    # 표준 편차는 2.5


