from django.urls import path

from . import views
from .views import *

urlpatterns = [
    path('', views.get_queryset, name='home'),
    path('result/', views.result, name='result'),
    path('api', views.ChartData.as_view()),
]