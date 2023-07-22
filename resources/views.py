from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
def resources(request):
    return render(request, "resources/index.html")