document.addEventListener('DOMContentLoaded', function() {
    const userSearch = document.getElementById('userSearch');
    const usersGrid = document.getElementById('usersGrid');
    const users = Array.from(document.querySelectorAll('.user-card'));

    // Arama fonksiyonu
    function filterUsers() {
        const searchTerm = userSearch.value.toLowerCase();
        
        users.forEach(user => {
            const username = user.dataset.username;
            const isVisible = username.includes(searchTerm);
            user.style.display = isVisible ? 'flex' : 'none';
        });

        // Sonuç yoksa mesaj göster
        const visibleUsers = users.filter(user => user.style.display !== 'none');
        if (visibleUsers.length === 0) {
            if (!document.querySelector('.no-results')) {
                const noResults = document.createElement('div');
                noResults.className = 'no-results';
                noResults.innerHTML = `
                    <i class="fas fa-search"></i>
                    <p>"${searchTerm}" için sonuç bulunamadı</p>
                `;
                usersGrid.appendChild(noResults);
            }
        } else {
            const noResults = document.querySelector('.no-results');
            if (noResults) {
                noResults.remove();
            }
        }
    }

    // Lazy loading için IntersectionObserver
    const userAvatars = document.querySelectorAll('.user-avatar img');
    const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                observer.unobserve(img);
            }
        });
    });

    userAvatars.forEach(img => {
        if (img.dataset.src) {
            imageObserver.observe(img);
        }
    });

    // Event listeners
    if (userSearch) {
        userSearch.addEventListener('input', filterUsers);
    }

    // Animasyonlar için IntersectionObserver
    const fadeObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
                fadeObserver.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1
    });

    users.forEach(user => {
        fadeObserver.observe(user);
    });
}); 