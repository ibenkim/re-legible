from django.urls import path

from . import views

app_name = 'testing'
urlpatterns = [
    path("", views.testing, name="testing")
]