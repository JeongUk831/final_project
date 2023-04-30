from django.urls import path, include
from . import views

app_name = "symptom"
urlpatterns = [
    path('', views.search_symptom, name="search_symptom"),  # /search_symptom/
    path('symptom1/', views.symptom1, name="symptom1"),  # /search_symptom/symptom1/
    path('symptom2/', views.symptom2, name="symptom2"),  # /search_symptom/symptom2/
    path('symptom1/doubt_disease/', views.doubt_disease, name="doubt_disease1"),  # /search_symptom/symptom1/doubt_disease/
    path('symptom2/doubt_disease/', views.doubt_disease, name="doubt_disease2"),  # /search_symptom/symptom2/doubt_disease/
    path('search_hospital/', include('search_hospital.urls'), name="search_hospital"),  # /search_hospital/
]