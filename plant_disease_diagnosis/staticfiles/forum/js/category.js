document.addEventListener('DOMContentLoaded', function() {
    const postsGrid = document.getElementById('postsGrid');
    const postSearch = document.getElementById('postSearch');
    const sortSelect = document.getElementById('sortPosts');
    let posts = Array.from(document.querySelectorAll('.post-card'));
    let currentPosts = [...posts];

    // Arama fonksiyonu
    function filterPosts(searchTerm) {
        currentPosts = posts.filter(post => {
            const title = post.dataset.title;
            return title.includes(searchTerm.toLowerCase());
        });
        sortPosts(sortSelect.value);
    }

    // Sıralama fonksiyonu
    function sortPosts(sortBy) {
        currentPosts.sort((a, b) => {
            switch(sortBy) {
                case 'latest':
                    return b.dataset.date - a.dataset.date;
                case 'popular':
                    return b.dataset.views - a.dataset.views;
                case 'most_commented':
                    return b.dataset.comments - a.dataset.comments;
                case 'most_viewed':
                    return b.dataset.views - a.dataset.views;
                default:
                    return 0;
            }
        });
        
        // DOM'u güncelle
        postsGrid.innerHTML = '';
        currentPosts.forEach(post => postsGrid.appendChild(post));
        
        // Animasyon efekti
        animatePosts();
    }

    // Gönderi animasyonu
    function animatePosts() {
        const posts = document.querySelectorAll('.post-card');
        posts.forEach((post, index) => {
            post.style.opacity = '0';
            post.style.transform = 'translateY(20px)';
            setTimeout(() => {
                post.style.transition = 'all 0.3s ease';
                post.style.opacity = '1';
                post.style.transform = 'translateY(0)';
            }, index * 100);
        });
    }

    // Arama olayı
    if (postSearch) {
        let searchTimeout;
        postSearch.addEventListener('input', (e) => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                filterPosts(e.target.value);
            }, 300);
        });
    }

    // Sıralama olayı
    if (sortSelect) {
        sortSelect.addEventListener('change', (e) => {
            sortPosts(e.target.value);
        });
    }

    // Görsel önizleme
    const imageContainers = document.querySelectorAll('.image-preview');
    imageContainers.forEach(container => {
        const img = container.querySelector('img');
        if (img) {
            img.addEventListener('click', () => {
                openImageViewer(img.src);
            });
        }
    });

    // Görsel görüntüleyici
    function openImageViewer(src) {
        const viewer = document.createElement('div');
        viewer.className = 'image-viewer';
        viewer.innerHTML = `
            <div class="viewer-content">
                <span class="close-viewer">&times;</span>
                <img src="${src}" alt="Büyük görsel">
            </div>
        `;
        
        document.body.appendChild(viewer);
        document.body.style.overflow = 'hidden';
        
        // Kapatma olayları
        viewer.addEventListener('click', (e) => {
            if (e.target === viewer || e.target.classList.contains('close-viewer')) {
                viewer.remove();
                document.body.style.overflow = 'auto';
            }
        });

        // ESC tuşu ile kapatma
        document.addEventListener('keydown', function closeOnEsc(e) {
            if (e.key === 'Escape') {
                viewer.remove();
                document.body.style.overflow = 'auto';
                document.removeEventListener('keydown', closeOnEsc);
            }
        });
    }

    // Sonsuz kaydırma (opsiyonel)
    let isLoading = false;
    let currentPage = 1;

    function loadMorePosts() {
        if (isLoading) return;
        
        const lastPost = document.querySelector('.post-card:last-child');
        if (!lastPost) return;

        const observer = new IntersectionObserver((entries) => {
            const lastEntry = entries[0];
            if (!lastEntry.isIntersecting) return;

            isLoading = true;
            currentPage++;

            // AJAX isteği ile yeni gönderileri yükle
            fetch(`?page=${currentPage}&ajax=1`)
                .then(response => response.text())
                .then(html => {
                    const parser = new DOMParser();
                    const doc = parser.parseFromString(html, 'text/html');
                    const newPosts = doc.querySelectorAll('.post-card');
                    
                    if (newPosts.length > 0) {
                        newPosts.forEach(post => {
                            postsGrid.appendChild(post.cloneNode(true));
                        });
                        posts = Array.from(document.querySelectorAll('.post-card'));
                        currentPosts = [...posts];
                    }
                    
                    isLoading = false;
                })
                .catch(() => {
                    isLoading = false;
                });

            observer.unobserve(lastPost);
        }, {
            threshold: 0.5
        });

        observer.observe(lastPost);
    }

    // Sonsuz kaydırmayı etkinleştir
    if (document.querySelector('.pagination')) {
        loadMorePosts();
    }
}); 