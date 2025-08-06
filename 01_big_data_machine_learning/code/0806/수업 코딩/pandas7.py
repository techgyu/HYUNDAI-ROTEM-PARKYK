# 웹 문서 읽기
import urllib.request as req 
from bs4 import BeautifulSoup 
import urllib
import csv;
import re 
import pandas as pd
import requests 

# 위키백과 문서 읽기
# url = "https://ko.wikipedia.org/wiki/%EC%9D%B4%EC%88%9C%EC%8B%A0"
# df = pd.read_html(url)

# wiki = req.urlopen(url)
# print(wiki)

# soup = BeautifulSoup(wiki, "html.parser")
# print(soup.select("#mw-content-text > div.mw-parser-output > p"))

# 네이버 제공 코스피 정보 읽기 - DataFrame에 담아서 처리
# 웹문서 읽기
# 위키백과 문서 읽기 - 이순신 자료

# url_template="https://finance.naver.com/sise/sise_market_sum.naver?&page={}"
# csv_fname='네이버코스피.csv'

# with open(csv_fname, mode='w',encoding='utf-8',newline='') as f:
#     writer = csv.writer(f)
#     #제목표시 
#     headers='N   종목명   현재가   전일비   등락률   액면가   시가총액   상장주식수   외국인비율   거래량   PER   ROE'.split()
#     writer.writerow(headers)
    
#     for page in range(1,3):
#         print(f"페이지 {page} 처리 중...")
#         url=url_template.format(page)
#        # print(url) 
#         res=requests.get(url)
#         res.raise_for_status() #실패하면 작업중지하라
#         soup=BeautifulSoup(res.text,'html.parser') #html 안되면 lxml 하라
#       #  rows=soup.find('table',attrs={'class':'type_2'}).find('tbody').find_all('tr')
#         rows=soup.select('table.type_2 tbody tr') #클래스는 . ,  아이디는 # , 자식은 스페이스
        
#         for row in rows:
#             # print("----------------row---------------: \n", row)
#             cols=row.find_all('td')
#             if len(cols)<len(headers):
#                 # print(f"[스킵됨] 열 수 부족:{len(cols)}")
#                 continue 
#             row_data=[re.sub(r'[\n\t]+','',col.get_text()) for col in cols] #섭스티튜트 공백으로 대체하기
#             writer.writerow(row_data)
# print('csv 저장성공')

df = pd.read_csv('네이버코스피.csv', encoding='utf-8', index_col=False)
print(df.head(3))  # 처음 3개 행 출력
print(df.tail(3))  # 마지막 3개 행 출력
print(df.columns.tolist())  # 컬럼 이름 출력

print(df.info())  # DataFrame 정보 출력
#방법1
numeric_cols = ['현재가', '전일비', '등락률', '액면가', '시가총액', '상장주식수', '외국인비율', 'ROE']

print(f"숫자형 컬럼: {numeric_cols}")  # 숫자형 컬럼 이름 출력

# 전일비 전용 전처리 함수
def clean_change_direction(value):
    if pd.isna(value):
        return pd.NA
    val = str(value).strip()
    val = val.replace(',', '').replace('상승', '+').replace('하락', '-')
    val = re.sub(r'[^\d\.\-\+]', '', val) # 숫자/기호 외 문자 제거
    try:
        return float(val)
    except ValueError:
        return pd.NA
    
df['전일비'] = df['전일비'].apply(clean_change_direction)
print(df.head(3))  # 전처리 후 처음 3개 행 출력

# 일반 숫자형 컬럼 전처리
def clean_numeric_column(series):
    return(
        series.astype(str)
        .str.replace(',', '', regex=False)
        .str.replace('%', '', regex=False)
        .replace(['', '-', 'N/A', 'nan'], pd.NA)
        .apply(lambda x: pd.to_numeric(x, errors='coerce'))
    )
for col in numeric_cols:
    df[col] = clean_numeric_column(df[col])
print('숫자 칼럼 일괄 처리 후')
print(df.head(2))  # 전처리 후 처음 2개 행 출력

print(df.describe())  # 데이터 요약 통계
print(df[['종목명', '현재가', '전일비']].head()) # 특정 컬럼 출력
print('시가 총액 top 5')
# 방법 2
#  top5 = df.dropna(subset=['시가총액']).nlargest(5, '시가총액')
top5 = df.dropna(subset=['시가총액']).sort_values(by='시가총액', ascending=False).head(5)
print(top5[['종목명', '시가총액']])  # 시가 총액 상위 5개 종목 출력

#방법2
# numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
# print(f"숫자형 컬럼: {numeric_cols}")  # 숫자형 컬럼 이름 출력