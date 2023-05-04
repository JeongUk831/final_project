from django.shortcuts import render
import pymongo

conn = pymongo.MongoClient('localhost', 27017)
db = conn.get_database('final')


# Create your views here.
def disease_dict(request):
    collection = db.get_collection('diseases')
    # diseases = []
    query = request.GET.get('q')  # 검색어 가져오기
    if not query:
        results = "질병을 입력하세요."
        return render(request, 'disease_dict/disease_dict.html', {'results':results})  # 질병명을 검색하지 않은 경우
    else :
        info = []
        info.append(collection.find({"질병명" : {'$regex' : query}}))
        return render(request, 'disease_dict/disease_list.html', {'info':info})