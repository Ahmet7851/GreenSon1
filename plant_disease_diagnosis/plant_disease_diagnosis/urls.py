from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('diagnosis.urls')),
    path('forum/', include('forum.urls', namespace='forum')),
    
    # Authentication URLs
    path('login/', auth_views.LoginView.as_view(template_name='forum/auth/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='home'), name='logout'),
    path('register/', include('forum.auth_urls')),  # Kayıt işlemi için ayrı URL'ler
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
