from django.urls import path

from . import views

app_name = 'testing'
urlpatterns = [
    path("", views.testing, name="testing"),
    path("result", views.result, name="result"),
    path("upload", views.upload, name="upload")
]