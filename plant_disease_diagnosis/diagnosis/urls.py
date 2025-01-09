from django.urls import path
from . import views

urlpatterns = [ 
    path('', views.home, name='home'),
    path('supplements/', views.supplements, name='supplements'),
    path('hakkinda/', views.hakkinda, name='hakkinda'),
    path('tespit/', views.tespit, name='tespit'),
]