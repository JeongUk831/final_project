from django.urls import path, include
from . import views

app_name = "landing_page"
urlpatterns = [
    path('', views.landing),  # 랜딩(메인) 페이지 진입
    path('search_symptom/', include('search_symptom.urls'), name="search_symptom"),  # /search_symptom/
    path('search_hospital/', include('search_hospital.urls'), name="search_hospital"),  # /search_hospital/
    path('disease_dict/', include('disease_dict.urls'), name="disease_dict"),  # /disease_dict/
]