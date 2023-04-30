from django.shortcuts import render

# Create your views here.
def search_symptom(request):
    return render(request, 'search_symptom/search_symptom.html')

def symptom1(request):
    return render(request, 'search_symptom/symptom1.html')

def symptom2(request):
    return render(request, 'search_symptom/symptom2.html')

def doubt_disease(request):
    return render(request, 'search_symptom/doubt_disease.html')
