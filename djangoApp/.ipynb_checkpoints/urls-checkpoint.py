from django.urls import path

from . import views
from .views import *

urlpatterns = [
    path('', views.get_queryset, name='home'),
    path('result/', views.result, name='result'),
]