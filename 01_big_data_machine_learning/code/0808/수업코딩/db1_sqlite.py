# Local Database 연동 후 자료를 읽어 DataFrame에 저장

import sqlite3

sql = "create table if not exists tips (product varchar(10), maker varchar(10), weight real, price integer)"
conn = sqlite3.connect(':memory:')
conn.execute(sql)
conn.commit()

data1 = ('mouse', 'samsung', 12.5, 5000)

stmt = "insert into tips values(?, ?, ?, ?)"  # 테이블 이름 수정
data1 = ("mouse", "samsung", 12.5, 5000)
data2 = ('mouse2', 'samsung', 15.5, 8000)
conn.execute(stmt, data1)

# 복수 개 추가
datas = [('mouse3', 'lg', 22.5, 15000), ('mouse4', 'lg', 25.5, 15500)]
conn.executemany(stmt, datas)

cursor = conn.execute("select * from tips")  # 테이블 이름 수정
rows = cursor.fetchall()
print(rows[0], ' ', rows[0][0])

for a in rows:
    print(a)

import pandas as pd
df = pd.DataFrame(rows, columns=['product', 'maker', 'weight', 'price'])
print(df)
print(df.to_html())
df2 = pd.read_sql("select * from tips", conn)  # 테이블 이름 수정
print(df2)
print()

pdata = {
    'product':['연필', '볼펜', '지우개'],
    'maker': ['동아', '모나미', '모나미'],
    'weight': [1.5, 5.5, 10.0],
    'price': [500, 1000, 1500]
}
frame = pd.DataFrame(pdata)
# print(frame)
frame.to_sql("test", conn, if_exists='append', index=False) # 테이블 이름 수정
df3 = pd.read_sql("select product, maker, price 가격, weight as 무게 from test", conn)  # 테이블 이름 수정
print(df3)

cursor.close()
conn.close()