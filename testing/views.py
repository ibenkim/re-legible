from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
def testing(request):
    return render(request, "testing/index.html")

def result(request):
    return render(request, "testing/result.html")

def upload(request):
    return render(request, "testing/upload.html")