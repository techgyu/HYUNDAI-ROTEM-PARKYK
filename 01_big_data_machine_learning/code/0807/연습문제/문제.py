from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 한글 폰트
plt.rcParams['axes.unicode_minus'] = False     # 마이너스 깨짐 방지

# 봉구스 밥버거

with open("./01_big_data_machine_learning/code/0807/연습문제/____ 봉구스밥버거 ____.html", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

names = [tag.text.strip() for tag in soup.select('ul#bonmenu_list li div.txtbox > div.tit')]
prices = [int(tag.text.strip().replace(',', '')) for tag in soup.select('ul#bonmenu_list li div.txtbox > div.price')]
# 읽어온 거 출력
print("메뉴명:", names)
print("가격:", prices)

# 메뉴명    가격    설명
df = pd.DataFrame({
    '상품명': names,
    '가격': prices,
    '설명': ['저는 봉구킹을 좋아합니다!'] * len(names)
})
print(df)

# 건수:
print('메뉴 개수: \t', len(names), '개')
# 가격평균:
print('가격 평균: \t', round(df['가격'].mean(), 2))
# 표준편차:
print('표준 편차: \t', round(df['가격'].std(), 2))
# 최고가격:
print('최고 가격: \t', df['가격'].max())
# 최저가격:
print('최저 가격: \t', df['가격'].min())

# 시각화
plt.figure(figsize=(15, 9))

plt.subplot(6, 2, 1)
plt.plot(range(len(df)), df['가격'], 'r+') 
plt.xlabel('순서')
plt.ylabel('가격')
plt.grid()

plt.subplot(6, 2, 1)
plt.plot(range(len(df)), df['가격'], 'r+') 
plt.xlabel('순서')
plt.ylabel('가격')
plt.grid()

plt.subplot(6, 2, 2)
plt.bar(range(len(df)), df['가격'], color='skyblue')
plt.xlabel('순서')
plt.ylabel('가격')
plt.grid()

plt.show()