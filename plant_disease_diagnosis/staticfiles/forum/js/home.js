document.addEventListener('DOMContentLoaded', function() {
    // Kategori arama fonksiyonu
    const categorySearch = document.getElementById('categorySearch');
    const categoryCards = document.querySelectorAll('.category-card');

    if (categorySearch) {
        categorySearch.addEventListener('input', function(e) {
            const searchTerm = e.target.value.toLowerCase();

            categoryCards.forEach(card => {
                const categoryName = card.querySelector('h3').textContent.toLowerCase();
                const categoryDesc = card.querySelector('p').textContent.toLowerCase();

                if (categoryName.includes(searchTerm) || categoryDesc.includes(searchTerm)) {
                    card.style.display = 'flex';
                } else {
                    card.style.display = 'none';
                }
            });
        });
    }

    // İstatistik sayaçları animasyonu
    const statNumbers = document.querySelectorAll('.stat-number');
    
    function animateNumber(element) {
        const target = parseInt(element.textContent);
        const duration = 2000; // 2 saniye
        const step = target / (duration / 16); // 60fps
        let current = 0;

        const timer = setInterval(() => {
            current += step;
            if (current >= target) {
                element.textContent = target;
                clearInterval(timer);
            } else {
                element.textContent = Math.floor(current);
            }
        }, 16);
    }

    // Intersection Observer ile görünür olduğunda animasyonu başlat
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateNumber(entry.target);
                observer.unobserve(entry.target);
            }
        });
    });

    statNumbers.forEach(number => observer.observe(number));
}); 