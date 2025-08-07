from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time

# 크롬 드라이버 경로 지정 (chromedriver.exe 위치에 맞게 수정)
chrome_path = "./01_big_data_machine_learning/code/0807/연습문제/chromedriver.exe"
service = Service(chrome_path)
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # 창 안 띄우고 실행

driver = webdriver.Chrome(service=service, options=options)
driver.get("https://nenechicken.com/home_menu.asp")
time.sleep(2)  # 페이지 로딩 대기

html = driver.page_source
soup = BeautifulSoup(html, "html.parser")

with open("downloaded_menu.html", "w", encoding="utf-8") as f:
    f.write(soup.prettify())

driver.quit()