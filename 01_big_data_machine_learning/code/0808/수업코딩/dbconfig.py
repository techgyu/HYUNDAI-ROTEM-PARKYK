import pickle

config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '1234',
    'database': 'mydb',
    'port': 3306,
    'charset': 'utf8'
}

with open('./01_big_data_machine_learning/data/mymaria.dat', mode='wb') as obj:
    pickle.dump(config, obj)