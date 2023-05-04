from django.shortcuts import render
import pymongo

conn = pymongo.MongoClient('localhost', 27017)
db = conn.get_database('final')

# Create your views here.
def search_hospital(request, pk):
    collection = db.get_collection('hospital')
    info = collection.find({"진료과" : pk})
    return render(request,'search_hospital/search_hospital.html', {"info" : info})