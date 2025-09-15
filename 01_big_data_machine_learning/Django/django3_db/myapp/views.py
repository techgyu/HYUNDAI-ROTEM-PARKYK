from django.shortcuts import render
from django.db import connection
from django.utils.html import escape
import pandas as pd
import matplotlib.pyplot as plt
import json

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def indexFunc(request):
    return render(request, "index.html")

def dbshowFunc(request):
    dept = request.GET.get("dept")
    sql = """
        SELECT j.jikwonno AS 직원번호, j.jikwonname AS 직원명,
               b.busername AS 부서명, b.busertel AS 부서전화,
               j.jikwonpay AS 연봉, j.jikwonjik AS 직급
        FROM jikwon j
        INNER JOIN buser b ON j.busernum = b.buserno
    """
    params = []
    if dept:
        sql += " WHERE b.busername LIKE %s"
        params.append(f"%{dept}%")
    sql += " ORDER BY j.jikwonno"

    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        cols = [c[0] for c in cursor.description] if cursor.description else []

    df = pd.DataFrame(rows, columns=cols)
    join_html = df[["직원번호", "직원명", "부서명", "부서전화", "연봉", "직급"]].to_html(index=False) if not df.empty else '조회된 자료가 없어요'

    if not df.empty:
        stats_df = (
            df.groupby("직급")["연봉"]
            .agg(평균='mean', 표준편차=lambda x: x.std(ddof=0), 인원수="count")
            .round(2)
            .reset_index()
            .sort_values(by="평균", ascending=False)
        )
        stats_df['표준편차'] = stats_df['표준편차'].fillna(0)
        stats_html = stats_df.to_html(index=False)
    else:
        stats_html = "통계 대상 자료가 없어요"

    ctx_dict = {
        'dept': escape(dept),
        'join_html': join_html,
        'stats_html': stats_html,
    }
    return render(request, "dbshow.html", ctx_dict)

def gyutech_dbshowFunc(request):
    # 1. 직원/부서 join 및 근무년수 계산
    sql = """
        SELECT a.jikwonno AS 사번, a.jikwonname AS 직원명, b.busername AS 부서명,
               a.jikwonjik AS 직급, a.jikwonpay AS 연봉, a.jikwonibsail AS 입사일
        FROM jikwon a
        INNER JOIN buser b ON a.busernum = b.buserno
        ORDER BY b.buserno, a.jikwonname
    """
    with connection.cursor() as cursor:
        cursor.execute(sql)
        rows = cursor.fetchall()
        cols = [c[0] for c in cursor.description] if cursor.description else []
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df['근무년수'] = pd.to_datetime('today').year - pd.to_datetime(df['입사일']).dt.year

    join_html = df.to_html(index=False) if not df.empty else '조회된 자료가 없어요'

    # 2. 부서별 직급별 연봉합/평균
    jikgub_order = ['이사', '부장', '과장', '대리', '사원']
    if not df.empty:
        df['직급'] = pd.Categorical(df['직급'], categories=jikgub_order, ordered=True)
        sum_pay_by_buser = df.groupby('부서명')['연봉'].sum()
        mean_pay_by_buser = df.groupby('부서명')['연봉'].mean()
        labels = list(sum_pay_by_buser.index)
        sum_series = sum_pay_by_buser.values.tolist()
        mean_series = mean_pay_by_buser.values.tolist()
    else:
        labels, sum_series, mean_series = [], [], []

    context = {
        'labels_json': json.dumps(labels, ensure_ascii=False),
        'sum_series_json': json.dumps(sum_series, ensure_ascii=False),
        'mean_series_json': json.dumps(mean_series, ensure_ascii=False),
        'join_html': join_html
    }

    # 3. 성별, 직급별 빈도표 (출력만, context 미포함)
    sql = "SELECT jikwonjik AS 직급, jikwongen AS 성별 FROM jikwon ORDER BY jikwonno"
    with connection.cursor() as cursor:
        cursor.execute(sql)
        rows = cursor.fetchall()
        cols = [c[0] for c in cursor.description] if cursor.description else []
    df_sex = pd.DataFrame(rows, columns=cols)
    if not df_sex.empty:
        cross_table = pd.crosstab(df_sex['직급'], df_sex['성별'], margins=True)
        cross_table_html = cross_table.to_html()
    else:
        cross_table_html = "자료가 없습니다."

    context['cross_table_html'] = cross_table_html

    return render(request, "gyutech_dbshow.html", context)