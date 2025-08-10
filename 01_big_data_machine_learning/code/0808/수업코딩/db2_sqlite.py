import MySQLdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False

# conn = MySQLdb.connect(
#     host='127.0.0.1',
#     user='root',
#     password='1234',
#     db='mydb',
#     port=3306,
#     charset='utf8'
# )

config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '1234',
    'database': 'mydb',
    'port': 3306,
    'charset': 'utf8'
}

try:    
    conn = MySQLdb.connect(**config)
    sql = "SELECT jikwonno, jikwonname from jikwon"
    cursor = conn.cursor()
    cursor.execute(sql)

    for (a, b) in cursor:
        print(a, b)
except Exception as e:
    print("Error:", e)
finally:
    conn.close()