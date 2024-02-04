from django.shortcuts import render
from django.http import JsonResponse
from .serializers import *
from .models import *
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, permission_classes
import json
from django.contrib.auth import login,authenticate,logout
import librosa
import librosa.display
import numpy as np
from tensorflow.keras.models import load_model
import os
import pandas as pd
import pickle

# Create your views here.

""" FRONTED MUST SEND DATA AS FORMDATA """
@csrf_exempt
def register_user(request):
    if request.method=='POST':
        user_data=UserModelSerializer(data=request.POST)
        if user_data.is_valid():
            user_data.save()
            return JsonResponse({'success':True})
        else:
            return JsonResponse({'success':False,'message':str(user_data.errors)})

@csrf_exempt
def signin(request):
    if request.method=='POST':
        email=request.POST['email']
        password=request.POST['password']
        print(email,password)
        user=authenticate(request,email=email,password=password)
        print(user)
        if user is not None:
            login(request,user)
            return JsonResponse("You are logged in",safe=False)
        else:
            return JsonResponse("Invalid Details",safe=False)


@csrf_exempt
def signout(request):
    logout(request)
    return JsonResponse("You are logged out",safe=False)

@csrf_exempt
def dysarthria_detection(request):
    audio_file=request.FILES.get("audio_file")
    y, sr = librosa.load(audio_file)
    mean_mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128),axis=1)
    model = load_model("D:\Datathon\datathon\hackathon\dysarthria.h5")
    mean_mfcc_shaped=mean_mfcc.reshape(-1,16,8,1)
    prediction=model.predict(mean_mfcc_shaped)
    rounded_number=round(prediction[0][0])
    dyserthria_obj=DysarthriaDetection(email=User.objects.get(email=request.user),audio_file=audio_file,mfccs=str(mean_mfcc_shaped),dysarthria_detected=1 if rounded_number==1 else 0)
    dyserthria_obj.save()
    if rounded_number==1:
        return JsonResponse("Dysarthria",safe=False)
    else:
        return JsonResponse("Not Dyserthria",safe=False)

@csrf_exempt
def stroke_prediction(request):
    gender=request.POST["gender"]
    age=request.POST["age"]
    hypertension=request.POST["hypertension"]
    heart_disease=request.POST["heart_disease"]
    ever_married=request.POST["ever_married"]
    work_type=request.POST["work_type"]
    Residence_type=request.POST["Residence_type"]
    avg_glucose_level=request.POST["avg_glucose_level"]
    bmi=request.POST["bmi"]
    smoking_status=request.POST["smoking_status"]

    data={"gender":gender,"age":age,"hypertension":hypertension,"heart_disease":heart_disease,"ever_married":ever_married,"work_type":work_type,"Residence_type":Residence_type,"avg_glucose_level":avg_glucose_level,"bmi":bmi,"smoking_status":smoking_status}

    df = pd.DataFrame(data, index=[0])
    print(df.dtypes())

    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['avg_glucose_level'] = pd.to_numeric(df['avg_glucose_level'], errors='coerce')
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')


    pickle_file_path = os.path.join(os.path.dirname(__file__), 'xgb_classifier.pickle')

    with open(pickle_file_path, 'rb') as file:
        model = pickle.load(file)

    prediction=model.predict(df)



    stroke_obj=StrokePrediction(gender=gender,email=User.objects.get(email=request.user),age=age,hypertension=hypertension,heart_disease=heart_disease,ever_married=ever_married,work_type=work_type,Residence_type=Residence_type,avg_glucose_level=avg_glucose_level,bmi=bmi,smoking_status=smoking_status,stroke=prediction)

    stroke_obj.save()

    if prediction==1:
        return JsonResponse("Stroke",safe=False)
    else:
        return JsonResponse("Not Stroke",safe=False)



