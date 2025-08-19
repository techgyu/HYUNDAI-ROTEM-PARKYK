def analysisFunc(rdata):
    # 가설검정
    # 귀무가설: 성별에따라 선호하는 커피브랜드에 차이가 없다
    # 대립가설: 성별에따라 선호하는 커피브랜드에 차이가 있다
    df=pd.DataFrame(rdata)
    if df.empty:
        return pd.DataFrame(),'데이터가 없어요',pd.DataFrame() 
    
    # 결측치 제거
    df=df.dropna(subset=['gender','co_survey'])
    # 숫자로 바꿔야함. 더미변수로 바꿔야함
    df['genNUM']=df['gender'].apply(lambda g:1 if g=='남' else '여')
    df['coNUM'] = df['co_survey'].apply(
        lambda c: 1 if c == '스타벅스' 
        else 2 if c == '커피빈' 
        else 3 if c == '이디야' 
        else 4
    )   
    #print('df!!!!!!!!!!!!!!!!',df)
    # 교차표 작성
    crossTb1=pd.crosstab(index=df['gender'],columns=df['co_survey'])
    #print('crossTb1!!!!!!!!!!!!!!!!',crossTb1)
    # 카이제곱검정, 5이상은 해야한다
    
    # 표본 부족 시 메세지 전달
    if crossTb1.size ==0 or crossTb1.shape[0] < 2 or crossTb1.shape[1] < 2:
        results='!!!!표본이 부족!!!! 카이제곱 검정 수행 불가'
        return crossTb1, results,df 
    
    # 멀쩡하면 카이제곱검정 시작
    # 유의수준 
    alpha=0.05
    st,pv,dof,expected = stats.chi2_contingency(crossTb1)
    
    # 기대빈도 최소값체크 (경고용.. 5)
    min_expected=expected.min()
    expected_note=''
    if min_expected<5:
        #웹으로출력 tag 걸어줌
        expected_note=f'<br><small> *주의: 기대빈도의 최소값이{min_expected:.2f}로 5미만이 있어 카이제곱 가정에 다소 취약합니다. </small>'
   
    if pv>=alpha:
        #귀무채택 
        results=(
            f'p값이{pv:.5f}이므로 {alpha} 이상 -> '
            f'귀무 채택, 성별에 따라 선호 브랜드에 차이가 없다'
        ) 
    else:
        #귀무기각 
        results=(
            f'p값이{pv:.5f}이므로 {alpha} 미만 -> '
            f'귀무 기각, 성별에 따라 선호 브랜드에 차이가 있다'
        ) 
    return crossTb1,results,df # 리턴값은 하나다. 튜플 생략 가능 (crossTb1,results,df)