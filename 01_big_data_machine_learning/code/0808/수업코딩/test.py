import MySQLdb 
import numpy as np 
import pandas as pd        
import matplotlib.pyplot as plt                            
import sys   
import pickle


plt.rc('font', family = 'malgun gothic') 
plt.rcParams['axes.unicode_minus']=False  

"""# 파이선 객체 자체를 파일로 만듦
conn=MySQLdb.connect(pip
    #dict 타입
    host='127.0.0.1'
    ,user='root'
    ,password='1234'
    ,database='mydb'
    ,port=3306
    ,charset='utf8'
    ) """

#dic니까 : 으로 변경 키니까 문자열로 바꿈, 집합형 자료
#피클은 파이선 객체 그대로 저장함
"""config={
     'host':'127.0.0.1'
    ,'user':'root'
    ,'password':'1234'
    ,'database':'mydb'
    ,'port':3306
    ,'charset':'utf8' 
}""" 

try:
    with open('./01_big_data_machine_learning/data/mymaria.dat', 'rb') as obj:
        config=pickle.load(obj)
except Exception as e:
    print("!!!!!!!!!!!!위 오류:!!!!!!!!!!!!",e)
    sys.exit()
    
      
try:  
    conn=MySQLdb.connect(**config) # 딕셔너리 언패킹.. **로 -> 딕셔너리에서 =형식으로 다시 바뀜
    sql="""
    select jikwonno,jikwonname,busernum,jikwonjik,jikwonpay,jikwonibsail,jikwongen,jikwonrating
    from jikwon inner join buser
    on jikwon.busernum=buser.buserno
    """
    cursor=conn.cursor() 
    cursor.execute(sql)
    
    #출력1 
    for row in cursor:
        print(row)
except Exception as e:
    print("!!!!!!!!!!!!처리오류:!!!!!!!!!!!!",e)
finally:
    conn.close()