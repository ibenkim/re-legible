from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
def project(request):
    return render(request, "project/index.html")