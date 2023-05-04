from urllib.request import HTTPRedirectHandler
from django.shortcuts import render
from .predict_diseases import predict_diseases
import pymongo
from bson.objectid import ObjectId


# Create your views here.
conn = pymongo.MongoClient('localhost', 27017)
db = conn.get_database('final')

def search_symptom(request):
    return render(request, 'search_symptom/search_symptom.html')

def symptom1(request):
    result = []
    if 'q' in request.GET :
        question = request.GET['q']
        collection = db.get_collection('diseases')
        if question :
            predicts = predict_diseases(question)
            # print(predicts)
        
            for pr in predicts[5:12] :
                result.append(collection.find({"질병명" : pr, "진료과" : predicts[0]}))

        return render(request, 'search_symptom/doubt_disease.html', {"result" : result})
    else : 
        return render(request, 'search_symptom/symptom1.html')


def disease_info(request, dept, pk) :
    if request.method == 'GET' :
        collection = db.get_collection('diseases')
        detail = collection.find({"질병명" : pk, "진료과" : dept })
        # print(detail)
        return render(request, 'search_symptom/disease_info.html', {'detail' : detail})
    return render(request, 'search_symtom/disease_info.html')

def symptom2(request):
    return render(request, 'search_symptom/symptom2.html')

def doubt_disease(request):
    return render(request, 'search_symptom/doubt_disease.html')




