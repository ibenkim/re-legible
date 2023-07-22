from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("ben", views.ben, name="ben"),
    path("<str:name>", views.greet, name="greet")
]