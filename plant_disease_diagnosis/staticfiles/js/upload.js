document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('id_image');
    const fileNameDisplay = document.getElementById('selected-file-name');
    const notification = document.getElementById('popup-message');

    if (fileInput) {
        fileInput.addEventListener('change', function() {
            if (this.files && this.files[0]) {
                // Seçilen dosyanın adını göster
                fileNameDisplay.textContent = this.files[0].name;
                
                // Bildirim göster
                notification.classList.add('show');
                setTimeout(() => {
                    notification.classList.remove('show');
                }, 3000);
            }
        });
    }
}); 