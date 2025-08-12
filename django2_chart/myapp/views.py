from django.shortcuts import render  # 템플릿 렌더링 함수 임포트
import json, os  # JSON 처리와 OS 경로 관련 모듈 임포트
import pandas as pd  # 데이터 분석을 위한 pandas 임포트
import numpy as np  # 수치 계산을 위한 numpy 임포트
import requests  # HTTP 요청을 위한 requests 임포트
from django.conf import settings  # Django 설정값 임포트
from datetime import datetime  # 날짜/시간 처리를 위한 모듈 임포트

DATA_DIR = os.path.join(settings.BASE_DIR, 'data')  # 데이터 폴더 경로 설정
CSV_PATH = os.path.join(DATA_DIR, 'seattle_weather.csv')  # CSV 파일 경로 설정
CSV_URL = "https://raw.githubusercontent.com/vega/vega-datasets/master/data/seattle-weather.csv"  # 원본 CSV URL

# Create your views here.
def index(request):  # 메인 페이지 뷰 함수
    return render(request, "index.html")  # index.html 렌더링

def csvFunc():  # CSV 파일 확보 함수
    os.makedirs(DATA_DIR, exist_ok=True)  # 데이터 폴더가 없으면 생성
    if not os.path.exists(CSV_PATH):  # CSV 파일이 없으면
        response = requests.get(CSV_URL)  # 원격 CSV 파일 다운로드
        response.raise_for_status() # http 상태 코드가 200(성공)이 아니면 예외를 발생시킴
        
        with open(CSV_PATH, 'wb') as f:  # 파일을 바이너리 쓰기 모드로 열기
            f.write(response.content)  # 다운로드한 내용 저장

def show(request):  # 데이터 시각화 페이지 뷰 함수
    csvFunc() # csv 데이터 확보
    df = pd.read_csv(CSV_PATH)  # CSV 파일을 pandas DataFrame으로 읽기
    print(df.columns)  # 컬럼명 출력(디버깅용)
    print(df.info())  # 데이터프레임 정보 출력(디버깅용)

    # 일부 열만 참여
    df = df[['date', 'precipitation', 'temp_max', 'temp_min']].copy()  # 필요한 열만 추출
    df['date'] = pd.to_datetime(df['date'])  # 날짜 컬럼을 datetime 타입으로 변환
    df = df.dropna()  # 결측값 제거

    # 기술 통계 - 평균 / 표준편차 ...
    stats_df = df[['precipitation', 'temp_max', 'temp_min']].describe().round(3)  # 기술통계 계산 및 반올림
    print('stats_df:', stats_df)  # 통계 출력(디버깅용)

    # df의 상위 5행
    head_html = df.head(5).to_html(classes='table table-sm table-striped', index=False, border=0)  # 상위 5행을 HTML 테이블로 변환
    stats_html = stats_df.to_html(classes = 'table table-sm table-striped', border=0)  # 통계 테이블을 HTML로 변환

    # Echart 용 데이터(월별 평균 최고 기온)
    # 월 단위 평균 최고 기온 집계
    monthly = (
        df.set_index('date')  # 날짜를 인덱스로 설정
        .resample('ME')[['temp_max', 'temp_min']]  # 매 월 마지막 일을 기준으로 그룹화
        .mean()  # 평균 계산
        .reset_index()  # 인덱스 초기화
    )

    print('monthly :', monthly.head(2))  # 월별 데이터 일부 출력(디버깅용)

    labels = monthly['date'].dt.strftime('%Y-%m').tolist()  # 월별 라벨(YYYY-MM) 리스트 생성
    # print('labels : ', labels)

    series = monthly['temp_max'].round(2).tolist()  # 월별 평균 최고기온 리스트 생성
    print('series : ', series)  # 시리즈 출력(디버깅용)

    context_dic = {
        'head_html': head_html,  # 상위 5행 HTML
        'stats_html': stats_html,  # 통계 HTML
        'monthly_html': monthly.to_html(classes='table table-sm table-striped', index=False, border=0),  # 월별 데이터 HTML
        'labels_json': json.dumps(labels, ensure_ascii=False), # list -> json
        'series_json': json.dumps(series, ensure_ascii=False), # list -> json

    }

    return render(request,'show.html', context_dic)  # show.html 렌더링