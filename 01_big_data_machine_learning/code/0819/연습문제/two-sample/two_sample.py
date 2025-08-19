# 함수를 이용하여 두 집단의 t-검정을 수행하는 코드입니다.
# 사용 방법: import 현재 파일 이름 해놓고
# 함수가 들어있는 파일 이름.two_sample(첫번째 인자, 두번째 인자)으로 처리

import scipy.stats as stats

def two_sample(first_value, second_value): #first_value: 첫번째 집단, second_value: 두번째 집단
    # 정규성 검정
    print("\n정규성 검정 결과:")
    first_shapiro = stats.shapiro(first_value)
    second_shapiro = stats.shapiro(second_value)
    if first_shapiro.pvalue > 0.05 and second_shapiro.pvalue > 0.05:
        notify = f"p-value: {first_shapiro.pvalue:.3f}, second.pvalue: {second_shapiro.pvalue:.3f}로 유의수준 0.05보다 크므로, 정규성 만족"
    else:
        notify = f"p-value: {first_shapiro.pvalue:.3f}, second.pvalue: {second_shapiro.pvalue:.3f}로 정규성 불만족"
    print(notify)

    # 등분산성 검정
    print('\n등분산성 검정 결과: ')
    levene_value = stats.levene(first_value, second_value)
    levene_pval = float(levene_value.pvalue)
    if levene_pval > 0.05:
        notify = f"p-value: {levene_pval:.3f}로 유의수준 0.05보다 크므로, 등분산성 만족"
    else:
        notify = f"p-value: {levene_pval:.3f}로 유의수준 0.05보다 작으므로, 등분산성 불만족"
    print(notify)

    # T-test 검정
    if(levene_pval > 0.05):
        # 등분산성 만족 시
        print("\nT-test(equal_var=True) 결과: ")
        ttest_value = stats.ttest_ind(first_value, second_value, equal_var=True)
        ttest_pval = float(ttest_value.pvalue)
        if ttest_pval > 0.05:
            notify = f"p-value: {ttest_pval:.3f} 유의수준 0.05보다 크므로, 귀무가설 채택"
        else:
            notify = f"p-value: {ttest_pval:.3f} 유의수준 0.05보다 작으므로, 귀무가설 기각"
        print(notify)
    else:
        # 등분산성 불만족 시
        print("\nT-test(equal_var=False) 결과: ")
        ttest_value = stats.ttest_ind(first_value, second_value, equal_var=False)
        ttest_pval = float(ttest_value.pvalue)
        if ttest_pval > 0.05:
            notify = f"p-value: {ttest_pval:.3f} 유의수준 0.05보다 크므로, 귀무가설 채택"
        else:
            notify = f"p-value: {ttest_pval:.3f} 유의수준 0.05보다 작으므로, 귀무가설 기각"
        print(notify)

    return first_shapiro, second_shapiro, levene_value, ttest_value