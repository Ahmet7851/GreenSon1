from django.shortcuts import render
from .forms import UploadImageForm
from django.conf import settings
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image as PILImage
import os
import os.path
import google.generativeai as genai
import markdown

# Modeli yükleyin
tf_model = load_model(os.path.join(settings.BASE_DIR, 'diagnosis', 'assets', 'bitki_hastaligi_model.h5'))

# Gemini API yapılandırması
genai.configure(api_key=settings.GEMINI_API_KEY)

def home(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            image_path = os.path.join(settings.MEDIA_ROOT, image.name)
            with open(image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)
            
            prediction = predict_image(image_path)
            return render(request, 'result.html', {'image_path': os.path.join(settings.MEDIA_URL, image.name), 'prediction': prediction})
    else:
        form = UploadImageForm()
    return render(request, 'home.html', {'form': form})

# ANA SAYFA BUTOUNU
def home2(request):
    return render(request, 'home.html')

# Bitkimin Neyi Var Butonu
def tespit(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            image_path = os.path.join(settings.MEDIA_ROOT, image.name)
            with open(image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)
            
            prediction = predict_image(image_path)
            return render(request, 'result.html', {
                'image_path': os.path.join(settings.MEDIA_URL, image.name),
                'prediction': prediction
            })
    return render(request, 'tespit.html')

# İLAÇLAR BUTONU
def supplements(request):
    return render(request, 'supplements.html')


 #GREEAN AI BUTONU
def hakkinda(request):
    return render(request, 'hakkinda.html')

 #Forum BUTONU
def forum(request):
    return render(request, 'forum.html')



# TAHMİN
def predict_image(image_path):
    image = PILImage.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    image = image.astype("float32")
    
    tahmin = tf_model.predict(image)
    en_yuksek_olasilik_sinifi = np.argmax(tahmin)
    sinif_etiketleri = ["Bakteri", "Mantar", "Sağlıklı/Diğer", "Zararlı Haşere", "Virüs"]
    tahmin_etiket = sinif_etiketleri[en_yuksek_olasilik_sinifi]

    hastalik = tahmin_etiket
    
    try:
        # Gemini model yapılandırması
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Prompt hazırlama ve içerik oluşturma
        prompt = f"""
        Bitkimde {hastalik} sorunu var. Bu sorunu çözmek için izleyeceğim adımlar nelerdir?
        Lütfen madde madde, detaylı ve Türkçe olarak yanıtla.
        Çözüm önerilerini uygulama sırası, kullanılacak malzemeler ve dikkat edilmesi gereken noktalar ile birlikte açıkla.
        Markdown formatında yanıt ver.
        """
        
        response = model.generate_content(prompt)
        
        # Response kontrolü ve dönüştürme
        if not response or not response.text:
            cozum_onerileri = "Çözüm önerileri alınamadı. Lütfen daha sonra tekrar deneyin."
        else:
            md = markdown.Markdown(extensions=['extra'])
            cozum_onerileri = md.convert(response.text)
    except Exception as e:
        print(f"Gemini API Hatası: {e}")
        cozum_onerileri = f"Çözüm önerileri alınamadı. Hata: {str(e)}"

    formatted_response = f"""
    <div class="diagnosis-result">
        <h1 class="disease-title">Tahmin edilen hastalık: {tahmin_etiket}</h1>
        <div class="solution-steps">
            {cozum_onerileri}
        </div>
    </div>
    """
    
    return formatted_response