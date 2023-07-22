from django.urls import path

from . import views

app_name = 'screen'
urlpatterns = [
    path("", views.screen, name="screen"),
    path("ben", views.ben, name="ben")
]