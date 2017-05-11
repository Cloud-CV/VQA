from django.conf.urls import include, url
from demo import views

urlpatterns = [
    url(r'^upload/', views.file_upload, name='upload'),
    url(r'^', views.vqa, name='vqa'),
]
