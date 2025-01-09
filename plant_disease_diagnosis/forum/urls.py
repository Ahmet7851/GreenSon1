from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

app_name = 'forum'

urlpatterns = [
    path('', views.forum_home, name='home'),
    path('posts/', views.posts, name='posts'),
    path('post/<int:post_id>/', views.post_detail, name='post_detail'),
    path('post/create/', views.create_post, name='create_post'),
    path('post/<int:post_id>/edit/', views.edit_post, name='edit_post'),
    path('post/<int:post_id>/delete/', views.delete_post, name='delete_post'),
    path('profile/<str:username>/', views.profile, name='profile'),
    path('profile/update/', views.update_profile, name='update_profile'),
    path('login/', auth_views.LoginView.as_view(template_name='forum/auth/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='forum:home'), name='logout'),
    
    # Yorum URL'leri
    path('post/<int:post_id>/comment/', views.add_comment, name='add_comment'),
    path('comment/<int:comment_id>/edit/', views.edit_comment, name='edit_comment'),
    path('comment/<int:comment_id>/delete/', views.delete_comment, name='delete_comment'),
    path('categories/', views.categories, name='categories'),
    path('members/', views.members, name='members'),
    path('category/<slug:slug>/', views.category_detail, name='category'),
    path('new-posts/', views.new_posts, name='new_posts'),
    path('get-posts/<str:tab_type>/', views.get_posts_by_tab, name='get_posts_by_tab'),
] 