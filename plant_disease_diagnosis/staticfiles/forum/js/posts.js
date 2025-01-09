document.addEventListener('DOMContentLoaded', function() {
    const postSearch = document.getElementById('postSearch');
    const sortPosts = document.getElementById('sortPosts');
    const postsGrid = document.getElementById('postsGrid');
    const posts = Array.from(document.querySelectorAll('.post-card'));

    // Arama fonksiyonu
    function filterPosts() {
        const searchTerm = postSearch.value.toLowerCase();
        
        posts.forEach(post => {
            const title = post.dataset.title;
            const isVisible = title.includes(searchTerm);
            post.style.display = isVisible ? 'block' : 'none';
        });
    }

    // Sıralama fonksiyonu
    function sortPostsBy(criteria) {
        const sortedPosts = posts.sort((a, b) => {
            switch(criteria) {
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
        sortedPosts.forEach(post => {
            if (post.style.display !== 'none') {
                postsGrid.appendChild(post);
            }
        });
    }

    // Event listeners
    if (postSearch) {
        postSearch.addEventListener('input', filterPosts);
    }

    if (sortPosts) {
        sortPosts.addEventListener('change', (e) => {
            sortPostsBy(e.target.value);
        });
    }

    // Sayfa yüklendiğinde varsayılan sıralama
    sortPostsBy('latest');
}); 