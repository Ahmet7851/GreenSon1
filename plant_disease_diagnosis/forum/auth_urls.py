from django.urls import path
from . import auth_views

urlpatterns = [
    path('', auth_views.register, name='register'),
] 