from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
def screen(request):
    return render(request, "screen/index.html")

def ben(request):
    return HttpResponse("Welcome, master.")
