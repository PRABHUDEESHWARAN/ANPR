from django.urls import path
from .import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.index,name='home'),
    path('webcam/', views.webcamdetect,name='webcam'),
    path('image/', views.imagedetect,name='image'),
    path('video/', views.videodetect,name='video'),
    path('output/', views.output,name='output'),
    ]