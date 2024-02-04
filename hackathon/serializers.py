from rest_framework import serializers
from .models import *

class UserModelSerializer(serializers.ModelSerializer):
    class Meta:
        model=User
        fields='__all__'
    
    def create(self,validated_data):
        user=User(
            email=self.validated_data['email'],
            password=self.validated_data['password'],
            name=self.validated_data['name']
        )
        user.set_password(validated_data['password'])
        user.save()
        return user