from django.conf.urls import patterns, include, url

urlpatterns = patterns('',
    url(r'^vqa', 'demo.views.vqa', name='vqa'),
    url(r'^upload', 'demo.views.file_upload', name='upload'),
)
