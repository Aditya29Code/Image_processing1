from django.contrib import admin
from django.urls import path
from . import views
from .views import final2020

urlpatterns = [
 path('',views.final2020,name="home"),
 path('upload',views.kaam,name="kaam"),
]
