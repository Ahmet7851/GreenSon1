from django.contrib import admin
from django.urls import path, include
#Selam
urlpatterns = [
    path('admin/', admin.site.urls),
    path('forum/', include('forum.urls')),
] 
#Deneme