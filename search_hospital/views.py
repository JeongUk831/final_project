from django.shortcuts import render
import pymongo
from django.core.paginator import Paginator
from djongo.models import QuerySet

conn = pymongo.MongoClient('localhost', 27017)
db = conn.get_database('final')

# Create your views here.
def search_hospital(request, pk):
    collection = db.get_collection('hospital')
    info = collection.find({"진료과" : pk})
    hospital_list = list(info)

    paginator = Paginator(hospital_list, 10)
    pagenumber = request.GET.get('page')
    page_obj = paginator.get_page(pagenumber)

    return render(request,'search_hospital/search_hospital.html', {'page_obj': page_obj})


def hospital_info(request, pk) :
    collection = db.get_collection('hospital')
    info = collection.find({'기관ID' : pk})
    return render(request,'search_hospital/hospital_info.html', {'info': info})