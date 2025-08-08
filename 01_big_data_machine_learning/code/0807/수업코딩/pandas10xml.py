# XML로 제공되는 날씨자료 처리
from bs4 import BeautifulSoup
import urllib.request # 연습용, 코드가 장황 
import requests # 실전용, 코드가 간결
import pandas as pd


url = "https://www.kma.go.kr/XML/weather/sfc_web_map.xml"
# data = urllib.request.urlopen(url).read()
# print(data.decode('utf-8'))
soup = BeautifulSoup(urllib.request.urlopen(url), "xml")
print(soup)

# 데이터 긁어서 데이터 프레임 만들기
# <local> 태그의 정보 추출
data = []
for local in soup.find_all('local'):
    # print("local 태그:", local)
    row = {
        '지역코드': local.get('std_id'),
        '지역': local.text,
        '날씨': local.get('desc'),
        '온도': local.get('ta'),
        '시간 당 강수량': local.get('rn_hr1'),  # 추가
    }
    data.append(row)

df = pd.DataFrame(data)
print(df)
# 지역, 온도 잡아서 df에 넣어줌   
df = df[['지역', '온도']]
print(df)

# df.to_csv('weather.csv', index=False, encoding='utf-8-sig')  # CSV 파일로 저장

df = pd.read_csv('weather.csv', encoding='utf-8-sig')  # CSV 파일로 읽기
print(df[0:2]) # 첫 두 행 출력

print(df.tail(5))  # 마지막 5행 출력
print(df[-2:len(df)])  # 마지막 2행 출력

print()

print(df.iloc[0:2, :])  # 첫 번째 행 출력
print(df.loc[1:3, ['온도']]) # 1~3행의 '온도' 열 출력

print(df.info())  # 데이터프레임 정보 출력
print(df['온도'].mean())  # '온도' 열의 평균 출력
print(df['온도'] >= 30) # '온도' 열이 30 이상인 행의 불리언 시리즈 출력
print(df.loc[df['온도'] >= 32]) # '온도' 열이 30 이상인 행 출력
print(df.sort_values(['온도'], ascending=True)) # '온도' 열을 기준으로 오름차순 정렬