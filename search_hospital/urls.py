from django.urls import path, include
from . import views

urlpatterns = [
    path('<str:pk>/', views.search_hospital),
    path('hospital_info/<str:pk>', views.hospital_info),
]
