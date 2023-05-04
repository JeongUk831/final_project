from django.urls import path, include
from . import views

app_name="disease_dict"
urlpatterns = [
    path('', views.disease_dict, name="search_diease"),
    path('search_symptom/', include('search_symptom.urls'), name='search_symptom'),
    path('search_hospital/', include('search_hospital.urls'), name="disease_dict"),
]
