import pymongo
import pandas as pd
conn = pymongo.MongoClient('localhost', 27017)
db = conn.get_database('final')

collection = db.get_collection('hospital')

files = pd.read_csv('final_project\data\hospital.csv', encoding='utf-8')
collection.insert_many(files.to_dict('records'))