document.addEventListener('DOMContentLoaded', function() {
    const filterSection = document.querySelector('.filter-section');
    const products = document.querySelectorAll('.product-card');
    const productsCount = document.querySelector('.products-count');

    // Filtre tıklama olaylarını yönet
    filterSection.addEventListener('click', function(event) {
        // Tıklanan öğe veya üst öğesi bir filter-option mu kontrol et
        const filterOption = event.target.closest('.filter-option');
        if (!filterOption) return;

        // Checkbox'ı bul ve durumunu değiştir
        const checkbox = filterOption.querySelector('.custom-checkbox');
        if (!checkbox) return;

        // Checkbox durumunu değiştir
        checkbox.classList.toggle('checked');

        // Seçili filtreleri topla
        const selectedCategories = [];
        const selectedPriceRanges = [];
        
        document.querySelectorAll('.filter-option .custom-checkbox.checked').forEach(checked => {
            const option = checked.closest('.filter-option');
            if (option.dataset.category) {
                selectedCategories.push(option.dataset.category);
            }
            if (option.dataset.price) {
                selectedPriceRanges.push(option.dataset.price);
            }
        });

        // Ürünleri filtrele
        let visibleCount = 0;
        products.forEach(product => {
            const productCategory = product.dataset.category;
            const productPrice = parseFloat(product.dataset.price);
            
            let showProduct = true;

            // Kategori filtresi
            if (selectedCategories.length > 0) {
                showProduct = selectedCategories.includes(productCategory);
            }

            // Fiyat filtresi
            if (showProduct && selectedPriceRanges.length > 0) {
                showProduct = selectedPriceRanges.some(range => {
                    const [min, max] = range.split('-');
                    const minPrice = parseFloat(min);
                    const maxPrice = max === '+' ? Infinity : parseFloat(max);
                    return productPrice >= minPrice && productPrice <= maxPrice;
                });
            }

            // Ürünü göster/gizle
            product.style.display = showProduct ? '' : 'none';
            if (showProduct) visibleCount++;
        });

        // Ürün sayısını güncelle
        productsCount.textContent = `${visibleCount} Ürün Bulundu`;
    });

    // Sıralama işlevselliği
    const sortSelect = document.querySelector('.sort-select');
    const productsGrid = document.querySelector('.products-grid');

    if (sortSelect && productsGrid) {
        sortSelect.addEventListener('change', function() {
            console.log('Sıralama değiştirildi:', this.value);
            const sortedProducts = Array.from(products);
            
            sortedProducts.sort((a, b) => {
                const priceA = parseFloat(a.dataset.price);
                const priceB = parseFloat(b.dataset.price);
                const nameA = a.querySelector('.product-title').textContent.toLowerCase();
                const nameB = b.querySelector('.product-title').textContent.toLowerCase();
                const popularityA = parseInt(a.dataset.popularity || 0);
                const popularityB = parseInt(b.dataset.popularity || 0);
                
                switch(this.value) {
                    case 'price-low':
                        return priceA - priceB;
                    case 'price-high':
                        return priceB - priceA;
                    case 'name':
                        return nameA.localeCompare(nameB, 'tr');
                    case 'popular':
                        return popularityB - popularityA;
                    default:
                        return 0;
                }
            });

            console.log('Sıralanmış ürünler:', sortedProducts);

            productsGrid.innerHTML = '';
            sortedProducts.forEach(product => productsGrid.appendChild(product));
        });
    }
});
