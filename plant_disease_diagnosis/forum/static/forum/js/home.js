document.addEventListener('DOMContentLoaded', function() {
    const tabs = document.querySelectorAll('.tab-item');
    const contentContainer = document.querySelector('.forum-main-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Aktif tab'ı güncelle
            tabs.forEach(t => t.classList.remove('active'));
            this.classList.add('active');
            
            const tabType = this.dataset.tab;
            
            // AJAX isteği gönder
            fetch(`/forum/get-posts/${tabType}/`)
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        // İçeriği yumuşak bir geçişle güncelle
                        contentContainer.style.opacity = '0';
                        setTimeout(() => {
                            contentContainer.innerHTML = data.html;
                            contentContainer.style.opacity = '1';
                        }, 300);
                    }
                })
                .catch(error => console.error('Error:', error));
        });
    });
}); 