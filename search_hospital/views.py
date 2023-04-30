from django.shortcuts import render

# Create your views here.
def search_hospital(request):
    return render(
        request,
        'search_hospital/search_hospital.html'
    )