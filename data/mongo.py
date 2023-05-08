import pymongo

conn = pymongo.MongoClient('localhost', 27017)
db = conn.get_database('final')

collection = db.get_collection('hospital')