document.addEventListener('DOMContentLoaded', function() {
    const messages = document.querySelectorAll('.message-alert');
    
    messages.forEach(message => {
        // Kapatma butonu işlevselliği
        const closeBtn = message.querySelector('.message-close');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                message.classList.add('fade-out');
                setTimeout(() => message.remove(), 300);
            });
        }
        
        // Otomatik kaybolma
        setTimeout(() => {
            if (message && message.parentElement) {
                message.classList.add('fade-out');
                setTimeout(() => message.remove(), 300);
            }
        }, 5000);
    });
}); 