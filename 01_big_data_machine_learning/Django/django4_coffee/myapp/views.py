from django.shortcuts import render, redirect
import os
from django.conf import settings
import matplotlib
matplotlib.use('Agg') # 서버에서 시각화 GUI 없이 저장할 때 사용
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')

from myapp.models import Survey
import pandas as pd
import numpy as np
import scipy.stats as stats

# Create your views here.
def surveyMain(request):
    return render(request, "index.html")

def surveyView(request):
    return render(request, "coffee/coffeesurvey.html")

def surveyProcess(request):
    print("surveyProcess 실행")
    insertDataFunc(request)
    
    rdata = list(Survey.objects.all().values())
    # print("rdata:\n", rdata)
    print("호출이 왜 안 되노")
    crossTbl, results, df = analysisFunc(rdata) # 데이터 분석(이원 카이 제곱 검정)

    # 차트 처리함수 호출
    static_img_path = os.path.join(settings.BASE_DIR, 'static', 'images', 'cobar.png')
    save_brand_barFunc(df, static_img_path)


    return render(request, "coffee/result.html", {
        'crossTbl': crossTbl.to_html(),
        'results': results,
        'df': df.to_html(index=False)
    })

def analysisFunc(rdata):
    # 귀무가설(H1): 성별에 따른 선호하는 커피 전문점에 차이가 없다.
    # 대립가설(H0): 성별에 따른 선호하는 커피 전문점에 차이가 있다.
    
    # 데이터 분석(이원 카이 제곱 검정)
    df = pd.DataFrame(rdata)
    if df.empty:
        return pd.DataFrame(), '데이터가 없어요', pd.DataFrame()
    
    df.dropna(subset=['gender', 'co_survey'])
    df['genNum'] = df['gender'].apply(lambda g: 1 if g == '남' else '여') # dummy 변수 작성
    df['coNum'] = df['co_survey'].apply(
        lambda c: 1 if c == '스타벅스' 
        else 2 if c == '커피빈' 
        else 3 if c == '이디야' 
        else 4
    )
    crossTbl = pd.crosstab(index=df['gender'], columns=df['co_survey']) 
    print("crossTbl:\n", crossTbl)

    # 표본 부족 시 메세지 전달
    print("표본 부족 시 메시지 전달")
    if crossTbl.size == 0 or crossTbl.shape[0] < 2 or crossTbl.shape[1] < 2:
        results = "표본 자료가 부족해 카이제곱 검정 수행 불가!!!"
        return crossTbl, results, df
    
    # 카이제곱 검정
    alpha = 0.05 # 유의 수준
    st, pv, ddof, expected = stats.chi2_contingency(crossTbl)
    print("st, pv, ddof, expected:\n", st, pv, ddof, expected)

    # 기대 빈도 최솟값 체크(경고용)
    min_expected = expected.min()

    expected_note = ""
    if min_expected < 5:
        expected_note = f"<br><small>* 주의: 기대빈도의 최솟값이 {min_expected}로 5 미만이 있어 카이제곱 가정에 다소 취약합니다.</small>"
    if pv >= alpha:
        results = (
            f"p값이 {pv}로 유의수준 {alpha} 이상 ->"
            f"성별에 따라 선호 브랜드에 차이가 없다.(귀무가설 채택)"
        )
    else:
        results = (
            f"p값이 {pv}로 유의수준 {alpha} 미만 ->"
            f"성별에 따라 선호 브랜드에 차이가 있다.(대립가설 채택)"
        )

    return crossTbl, results, df

def insertDataFunc(request):
    if request.method == 'POST':
        print("POST method:\n", request.POST.get('gender'), request.POST.get('age'), request.POST.get('co_survey'))
        Survey.objects.create(
            gender=request.POST.get('gender'),
            age=request.POST.get('age'),
            co_survey=request.POST.get('co_survey')
        )

def surveyShow(request):
    return render(request, "coffee/coffeesurvey.html")

def save_brand_barFunc(df, out_path):
    # 브랜드명(x축)
    if df is None or df.empty or 'co_survey' not in df.columns:
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass
        return False
    
    order = ['스타벅스', '커피빈', '이디야', '탐앤탐스']
    brand_counts = df['co_survey'].value_counts().reindex(order, fill_value=0)

    # color 는 무지개 색을 사용
    cmap = plt.get_cmap("rainbow")
    n = len(brand_counts)
    colors = [cmap(i/ max(n - 1, 1)) for i in range(n)]

    fig = plt.figure()
    ax = brand_counts.plot(kind = 'bar', width=0.6, color=colors, edgecolor='black')
    ax.set_xlabel('커피사')
    ax.set_ylabel('선호 건수')
    ax.set_title('커피 브랜드 선호 건수')
    ax.set_xticklabels(order, rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi = 120, bbox_inches='tight')
    plt.close(fig)  # 메모리 해제