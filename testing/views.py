from django.http import HttpResponse
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from ocr import predict
import os

# Create your views here.
def testing(request):
    return render(request, "testing/index.html")

def result(request):
    return render(request, "testing/result.html")

def upload(request):
    threshold = 140
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        print('uploaded url: ' + uploaded_file_url)
        result = predict(uploaded_file_url[1:], threshold)
        print('result: ' + str(result))
        return render(request, 'testing/result.html', {
            'uploaded_file_url': uploaded_file_url,
            'predict_result': result
        })
    return render(request, 'testing/index.html')