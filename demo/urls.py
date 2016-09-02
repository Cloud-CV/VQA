from django.conf.urls import include, url
from demo import views

urlpatterns = [
    url(r'^vqa/', views.vqa, name='vqa'),
    url(r'^upload/', views.file_upload, name='upload'),
]
