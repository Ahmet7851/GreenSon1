from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import login
from .forms import CustomUserCreationForm

def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Hoş geldiniz! Hesabınız başarıyla oluşturuldu.')
            return redirect('forum:home')
        else:
            error_messages = {
                'username': 'Kullanıcı adı',
                'password1': 'Şifre',
                'password2': 'Şifre tekrarı'
            }
            for field, errors in form.errors.items():
                field_name = error_messages.get(field, field)
                for error in errors:
                    messages.error(request, f'{field_name}: {error}')
    else:
        form = CustomUserCreationForm()
    return render(request, 'forum/auth/register.html', {'form': form}) 