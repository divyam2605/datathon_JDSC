from django.urls import path
from .views import *
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path("register",register_user,name="register"),
    path("login",signin,name="login"),
    path("logout",signout,name="logout"),
    path("dysarthria_test",dysarthria_detection),
    path("stroke_test",stroke_prediction),
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)