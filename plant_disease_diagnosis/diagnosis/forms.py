from django import forms

class UploadImageForm(forms.Form):
    image = forms.ImageField(
        label='Bitki Fotoğrafı',
        widget=forms.FileInput(attrs={
            'class': 'file-input',
            'accept': 'image/*'
        })
    )
