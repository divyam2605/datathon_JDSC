from django.db import models
import uuid
from django.contrib.auth.models import AbstractBaseUser,PermissionsMixin
from phonenumber_field.modelfields import PhoneNumberField
from django.conf import settings
from .managers import CustomUserManager
import datetime

# Create your models here.
class User(AbstractBaseUser,PermissionsMixin):

    name        =models.CharField(max_length=20)
    email       =models.EmailField(unique=True)
    password    =models.CharField(max_length=20)
    is_staff    =models.BooleanField(default=False)
    is_admin    =models.BooleanField(default=False)
    is_superuser    =models.BooleanField(default=False)

    objects=CustomUserManager()
    USERNAME_FIELD = 'email'

    def __str__(self):
        return str(self.email)

class StrokePrediction(models.Model):
    email=models.ForeignKey(User,on_delete=models.CASCADE)
    gender=models.IntegerField()
    age=models.FloatField()
    hypertension=models.IntegerField()
    heart_disease=models.IntegerField()
    ever_married=models.IntegerField()
    work_type=models.IntegerField()
    Residence_type=models.IntegerField()
    avg_glucose_level=models.FloatField()
    bmi=models.FloatField()
    smoking_status=models.IntegerField()
    stroke=models.IntegerField()

    def __str__(self):
        return self.email


class DysarthriaDetection(models.Model):
    email=models.ForeignKey(User,on_delete=models.CASCADE)
    mfccs=models.CharField(max_length=1000000)
    dysarthria_detected=models.BooleanField()
    audio_file=models.FileField(upload_to="media/")

    def __str__(self):
        return str(self.email)



class PneumoniaDetection(models.Model):
    email=models.ForeignKey(User,on_delete=models.CASCADE)
    image=models.ImageField(upload_to="media/")
    pneumonia=models.IntegerField()

    def __str__(self):
        return str(self.email)


        

