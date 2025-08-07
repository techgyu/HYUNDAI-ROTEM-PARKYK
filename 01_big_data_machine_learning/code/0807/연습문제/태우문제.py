import requests  # 웹 페이지 요청을 위한 라이브러리
from bs4 import BeautifulSoup  # HTML 파싱을 위한 라이브러리
import pandas as pd  # 데이터프레임 생성을 위한 라이브러리

# 웹 html 크롤링
url = "https://web.dominos.co.kr/goods/list?dsp_ctgr=C0101"  # 도미노피자 메뉴 페이지 URL
response = requests.get(url)  # 해당 URL로 GET 요청을 보냄
response.raise_for_status()  # 요청이 실패하면 예외 발생(에러 체크)
soup = BeautifulSoup(response.text, 'html.parser')  # 받아온 HTML을 파싱해서 soup 객체 생성
# print(soup)  # (필요시) 전체 HTML 구조를 확인할 때 사용

names = []  # 메뉴명을 저장할 리스트
sizes = []  # 사이즈명을 저장할 리스트
prices = []  # 가격을 저장할 리스트

# 모든 메뉴 리스트 ul > li 순회 (각 메뉴 아이템마다 반복)
for li in soup.select('div.menu-list ul li'):
    subject_div = li.select_one('div.prd-cont > div.subject')  # 메뉴명 영역 선택
    if subject_div:
        # 메뉴명에서 "NEW"와 "subzza" 등 불필요한 텍스트 제거
        menu_name = subject_div.get_text(strip=True)  # 메뉴명 전체 텍스트 추출(양쪽 공백 제거)
        menu_name = menu_name.replace("NEW", "").replace("Subzza", "").strip()  # 불필요한 단어 제거 후 다시 공백 제거
    else:
        continue  # subject_div가 없으면 다음 li로 넘어감

    # 가격 정보 추출 (사이즈별로 여러 개 있을 수 있음)
    result = li.select('div.prd-price div.price-box span.price')
    print((len)(result))  # (디버깅용) 가격 영역의 span 태그 개수 확인
    for price_span in li.select('div.prd-price div.price-box span.price'):
        print("price_span:", price_span)  # (디버깅용) 가격 영역 확인
        # 사이즈 추출 (예: L, M, S 등)
        size_tag = price_span.find(['span'], class_=['size_l', 'size_m', 'size_M', 'size_s', 'size_S'])
        size = size_tag.text.strip() if size_tag else ''  # 사이즈가 있으면 텍스트 추출, 없으면 빈 문자열
        print("size:", size)  # (디버깅용) 사이즈 텍스트 확인
        # 가격 추출 (사이즈 span 다음 텍스트)
        price_text = price_span.get_text(separator='|', strip=True)  # 가격 영역 전체 텍스트 추출, 구분자 |로 합침
        print("price_text:", price_text)  # (디버깅용) 가격 텍스트 확인
        # 가격만 남기기 (예: 'L|33,900원~' → '33,900원~')
        price = int(price_text.replace(size, '').replace('|', '').replace('~', '').replace(',', '').replace('원', '').strip())  # 사이즈와 구분자 제거 후 가격만 남김
        print("price:", price)
        names.append(menu_name)  # 메뉴명 리스트에 추가
        sizes.append(size)  # 사이즈 리스트에 추가
        prices.append(price)  # 가격 리스트에 추가

# 추출한 데이터를 데이터프레임으로 변환
df = pd.DataFrame({
    '메뉴명': names,
    '사이즈': sizes,
    '가격': prices
})
# print(df)