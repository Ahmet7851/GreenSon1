from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, authenticate
from django.contrib import messages
from .models import UserProfile, Category, Post, Comment, PostImage, PostFile
from .forms import CustomUserCreationForm
from django.db.models import Count, Max, F, Q, OuterRef, Subquery
from django.contrib.auth.models import User
from django.http import JsonResponse
import json
from django.views.decorators.http import require_POST
from diagnosis.models import BlogPost  # GreenAI ana sitesindeki blog modeli
from django.template.loader import render_to_string
from django.utils import timezone
from datetime import timedelta
from django.db.models.functions import Coalesce

def forum_home(request):
    posts = Post.objects.select_related('author', 'category').prefetch_related('comments').order_by('-created_at')
    
    # Blog yazılarını dictionary olarak oluştur
    blog_posts = [
        {
            'title': 'Bitki Beslenmesi',
            'description': 'Bitkilerinizin sağlıklı gelişimi için bilmeniz gerekenler.',
            'image_url': 'https://plus.unsplash.com/premium_photo-1663034421287-bbde71a7d581',
            'link': 'https://tarfin.com/blog/bitki-besleme-hakkinda-bilmeniz-gerekenler'
        },
        {
            'title': 'Toprak Analizi',
            'description': 'Toprak analizi yaparak verimi nasıl artırabilirsiniz?',
            'image_url': 'https://images.unsplash.com/photo-1615811361523-6bd03d7748e7',
            'link': 'https://teoxfarm.com/blog/icerik/toprak-analizi-sonuclarina-gore-gubreleme-uygulamalari/'
        },
        {
            'title': 'Sürdürülebilir Tarım',
            'description': 'Gelecek nesiller için sürdürülebilir tarım yöntemleri.',
            'image_url': 'https://plus.unsplash.com/premium_photo-1661963586402-b20023897431',
            'link': 'https://canbeltarim.com/2024/09/14/surdurulebilir-tarim-ve-biyocesitliligin-korunmasi/'
        }
    ]
    
    context = {
        'posts': posts,
        'blog_posts': blog_posts,
    }
    return render(request, 'forum/home.html', context)

def posts(request):
    posts = Post.objects.select_related('author', 'category').order_by('-created_at')
    return render(request, 'forum/posts.html', {'posts': posts})

def post_detail(request, post_id):
    post = get_object_or_404(Post, id=post_id)
    comments = post.comments.all().order_by('-created_at')
    return render(request, 'forum/post_detail.html', {
        'post': post,
        'comments': comments
    })

@login_required
def create_post(request):
    if request.method == 'POST':
        title = request.POST.get('title')
        content = request.POST.get('content')
        category_id = request.POST.get('category')
        category = get_object_or_404(Category, id=category_id)
        
        post = Post.objects.create(
            title=title,
            content=content,
            author=request.user,
            category=category
        )

        # Görselleri kaydet
        if request.FILES.getlist('images'):
            for image in request.FILES.getlist('images'):
                PostImage.objects.create(
                    post=post,
                    image=image
                )

        # Dosyaları kaydet
        if request.FILES.getlist('files'):
            for file in request.FILES.getlist('files'):
                PostFile.objects.create(
                    post=post,
                    file=file,
                    file_name=file.name,
                    file_size=file.size
                )
        
        messages.success(request, 'Gönderi başarıyla oluşturuldu.')
        return redirect('forum:post_detail', post_id=post.id)
    
    categories = Category.objects.all()
    return render(request, 'forum/create_post.html', {'categories': categories})

@login_required
def edit_post(request, post_id):
    post = get_object_or_404(Post, id=post_id, author=request.user)
    
    if request.method == 'POST':
        post.title = request.POST.get('title')
        post.content = request.POST.get('content')
        category_id = request.POST.get('category')
        post.category = get_object_or_404(Category, id=category_id)
        post.save()
        
        messages.success(request, 'Gönderi başarıyla güncellendi.')
        return redirect('forum:post_detail', post_id=post.id)
    
    categories = Category.objects.all()
    return render(request, 'forum/edit_post.html', {
        'post': post,
        'categories': categories
    })

@login_required
def delete_post(request, post_id):
    post = get_object_or_404(Post, id=post_id, author=request.user)
    
    if request.method == 'POST':
        post.delete()
        messages.success(request, 'Gönderi başarıyla silindi.')
        return redirect('forum:posts')
    
    return render(request, 'forum/delete_post.html', {'post': post})

@login_required
def add_comment(request, post_id):
    post = get_object_or_404(Post, id=post_id)
    
    if request.method == 'POST':
        content = request.POST.get('content')
        if content:
            comment = Comment.objects.create(
                post=post,
                author=request.user,
                content=content
            )
            messages.success(request, 'Yorumunuz başarıyla eklendi.')
        else:
            messages.error(request, 'Yorum içeriği boş olamaz.')
    
    return redirect('forum:post_detail', post_id=post.id)

@login_required
@require_POST
def edit_comment(request, comment_id):
    try:
        comment = Comment.objects.get(id=comment_id)
        
        # Yorum sahibi kontrolü
        if comment.author != request.user:
            return JsonResponse({
                'success': False,
                'error': 'Bu yorumu düzenleme yetkiniz yok'
            }, status=403)
        
        data = json.loads(request.body)
        content = data.get('content', '').strip()
        
        if not content:
            return JsonResponse({
                'success': False,
                'error': 'Yorum içeriği boş olamaz'
            }, status=400)
        
        comment.content = content
        comment.save()
        
        return JsonResponse({'success': True})
        
    except Comment.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'Yorum bulunamadı'
        }, status=404)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

@login_required
@require_POST
def delete_comment(request, comment_id):
    try:
        comment = Comment.objects.get(id=comment_id)
        
        # Yorum sahibi kontrolü
        if comment.author != request.user:
            return JsonResponse({
                'success': False,
                'error': 'Bu yorumu silme yetkiniz yok'
            }, status=403)
        
        comment.delete()
        return JsonResponse({'success': True})
        
    except Comment.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'Yorum bulunamadı'
        }, status=404)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

def categories(request):
    categories = Category.objects.all()
    for category in categories:
        category.post_count = category.posts.count()
    return render(request, 'forum/categories.html', {'categories': categories})

def members(request):
    members = User.objects.all()
    return render(request, 'forum/members.html', {'members': members})

def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            UserProfile.objects.create(user=user)
            login(request, user)
            messages.success(request, 'Hesabınız başarıyla oluşturuldu!')
            return redirect('forum:home')
    else:
        form = CustomUserCreationForm()
    return render(request, 'forum/auth/register.html', {'form': form})

@login_required
def profile(request, username):
    user_profile = get_object_or_404(UserProfile, user__username=username)
    user_posts = Post.objects.filter(author=user_profile.user).order_by('-created_at')
    user_comments = Comment.objects.filter(author=user_profile.user).order_by('-created_at')
    
    context = {
        'profile': user_profile,
        'posts': user_posts,
        'comments': user_comments,
    }
    return render(request, 'forum/profile.html', context)

def category_detail(request, slug):
    category = get_object_or_404(Category, slug=slug)
    posts = Post.objects.filter(category=category).order_by('-created_at')
    
    context = {
        'category': category,
        'posts': posts,
    }
    return render(request, 'forum/category.html', context)

def user_list(request):
    users = UserProfile.objects.select_related('user').all()
    return render(request, 'forum/users.html', {'users': users})

def profile_detail(request, username):
    user = get_object_or_404(User, username=username)
    profile = get_object_or_404(UserProfile, user=user)
    posts = Post.objects.filter(author=user).select_related('category')
    comments = Comment.objects.filter(author=user).select_related('post')
    
    context = {
        'profile': profile,
        'posts': posts,
        'comments': comments
    }
    return render(request, 'forum/profile.html', context)

def new_posts(request):
    posts = Post.objects.select_related('author', 'category').prefetch_related('comments').order_by('-created_at')[:20]
    blog_posts = [
        # Mevcut blog posts dictionary'si
    ]
    
    context = {
        'posts': posts,
        'blog_posts': blog_posts,
        'category': 'new'
    }
    return render(request, 'forum/home.html', context)

def get_posts(request, tab_type):
    # Son 1 hafta için tarih hesapla
    one_week_ago = timezone.now() - timedelta(days=7)
    
    if tab_type == 'new-messages':
        posts = Post.objects.select_related('author', 'category').prefetch_related('comments').order_by('-created_at')
    else:  # new-topics
        posts = Post.objects.select_related('author', 'category')\
                          .prefetch_related('comments')\
                          .filter(created_at__gte=one_week_ago)\
                          .order_by('-created_at')
    
    # Blog yazıları dictionary'si aynı kalacak
    blog_posts = [
        {
            'title': 'Bitki Beslenmesi',
            'description': 'Bitkilerinizin sağlıklı gelişimi için bilmeniz gerekenler.',
            'image_url': 'https://plus.unsplash.com/premium_photo-1663034421287-bbde71a7d581',
            'link': 'https://tarfin.com/blog/bitki-besleme-hakkinda-bilmeniz-gerekenler'
        },
        {
            'title': 'Toprak Analizi',
            'description': 'Toprak analizi yaparak verimi nasıl artırabilirsiniz?',
            'image_url': 'https://images.unsplash.com/photo-1615811361523-6bd03d7748e7',
            'link': 'https://teoxfarm.com/blog/icerik/toprak-analizi-sonuclarina-gore-gubreleme-uygulamalari/'
        },
        {
            'title': 'Sürdürülebilir Tarım',
            'description': 'Gelecek nesiller için sürdürülebilir tarım yöntemleri.',
            'image_url': 'https://plus.unsplash.com/premium_photo-1661963586402-b20023897431',
            'link': 'https://canbeltarim.com/2024/09/14/surdurulebilir-tarim-ve-biyocesitliligin-korunmasi/'
        }
    ]
    
    html = render_to_string('forum/partials/topic_list.html', {
        'posts': posts,
        'blog_posts': blog_posts
    })
    
    return JsonResponse({
        'html': html,
        'blog_posts': blog_posts
    })

def get_posts_by_tab(request, tab_type):
    try:
        last_comment = Comment.objects.filter(
            post=OuterRef('pk')
        ).order_by('-created_at')
        
        if tab_type == "recent-comments":
            posts = Post.objects.annotate(
                last_comment=Subquery(
                    last_comment.values('author__username')[:1]
                )
            ).order_by('-comments__created_at')[:10]
        else:
            posts = Post.objects.annotate(
                last_comment=Subquery(
                    last_comment.values('author__username')[:1]
                )
            ).order_by('-created_at')[:10]

        context = {'posts': posts}
        html = render_to_string('forum/partials/topic_list.html', context, request=request)
        return JsonResponse({'status': 'success', 'html': html})
        
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

def get_recent_comments(request):
    # Her post için en son yorumu al ve postları son yorum tarihine göre sırala
    posts_with_latest_comments = Post.objects.annotate(
        last_comment_date=Max('comments__created_at')
    ).filter(
        last_comment_date__isnull=False  # Yorumu olmayan postları filtrele
    ).order_by('-last_comment_date')[:10]  # Son 10 post
    
    # Her post için sadece en son yorumu al
    latest_comments = []
    seen_posts = set()  # Tekrarlanan postları kontrol etmek için set
    
    for post in posts_with_latest_comments:
        if post.id not in seen_posts:  # Post daha önce eklenmemişse
            latest_comment = Comment.objects.filter(
                post=post
            ).select_related(
                'author',
                'author__userprofile',
                'post'
            ).latest('created_at')
            
            latest_comments.append(latest_comment)
            seen_posts.add(post.id)  # Post ID'sini sete ekle
    
    context = {
        'comments': latest_comments
    }
    
    html = render_to_string('forum/partials/comment_list.html', context)
    return JsonResponse({'status': 'success', 'html': html})

@login_required
@require_POST
def update_profile(request):
    try:
        user = request.user
        profile = user.profile

        # Kullanıcı adı kontrolü
        new_username = request.POST.get('username')
        if new_username and new_username != user.username:
            if User.objects.filter(username=new_username).exists():
                messages.error(request, 'Bu kullanıcı adı zaten kullanılıyor.')
                return JsonResponse({'success': False, 'error': 'Bu kullanıcı adı zaten kullanılıyor.'})
            user.username = new_username

        # Email kontrolü
        new_email = request.POST.get('email')
        if new_email and new_email != user.email:
            if User.objects.filter(email=new_email).exists():
                messages.error(request, 'Bu e-posta adresi zaten kullanılıyor.')
                return JsonResponse({'success': False, 'error': 'Bu e-posta adresi zaten kullanılıyor.'})
            user.email = new_email

        # Profil bilgilerini güncelle
        bio = request.POST.get('bio', '')
        profile.bio = bio

        # Avatar güncelleme
        if 'avatar' in request.FILES:
            profile.avatar = request.FILES['avatar']

        user.save()
        profile.save()

        messages.success(request, 'Profiliniz başarıyla güncellendi.')
        return JsonResponse({'success': True})
    except Exception as e:
        messages.error(request, 'Profil güncellenirken bir hata oluştu.')
        return JsonResponse({'success': False, 'error': 'Profil güncellenirken bir hata oluştu.'})

def profile_view(request, username):
    profile = get_object_or_404(UserProfile, user__username=username)
    return render(request, 'forum/profile.html', {'profile': profile})
