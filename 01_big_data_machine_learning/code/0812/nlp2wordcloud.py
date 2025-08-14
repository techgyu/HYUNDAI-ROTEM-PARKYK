# 동아일보 검색 기능으로 문자열을 읽어 형태소 분석 후 워드클라우드로 출력

from bs4 import BeautifulSoup
import urllib.request
from urllib.parse import quote

keyword = input("검색어:")
print(quote(keyword))

target_url = "https://www.donga.com/news/search?query=" + quote(keyword)
# target_url = "https://www.donga.com/news/search?query=" + keyword
# print(target_url)

source_code = urllib.request.urlopen(target_url)
soup = BeautifulSoup(source_code, 'lxml', from_encoding='utf-8')
print(soup)