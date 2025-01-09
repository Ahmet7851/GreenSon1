document.addEventListener('DOMContentLoaded', function() {
    // Quill editor başlatma
    var quill = new Quill('#editor', {
        theme: 'snow',
        modules: {
            toolbar: [
                [{ 'header': [1, 2, 3, false] }],
                ['bold', 'italic', 'underline', 'strike'],
                [{ 'color': [] }, { 'background': [] }],
                [{ 'list': 'ordered'}, { 'list': 'bullet' }],
                ['link', 'image', 'code-block'],
                ['clean']
            ]
        }
    });

    // Form gönderilmeden önce içeriği gizli alana aktar
    document.getElementById('postForm').onsubmit = function() {
        var content = document.getElementById('content');
        content.value = quill.root.innerHTML;
    };

    // Görsel önizleme
    document.getElementById('imageInput').addEventListener('change', function(e) {
        handleImageUpload(e.target.files);
    });

    // Dosya önizleme
    document.getElementById('fileInput').addEventListener('change', function(e) {
        handleFileUpload(e.target.files);
    });

    // Sürükle-bırak işlemleri
    setupDragAndDrop('imageUpload', 'image/*');
    setupDragAndDrop('fileUpload', '*');
});

function handleImageUpload(files) {
    const preview = document.getElementById('imagePreview');
    preview.innerHTML = '';

    Array.from(files).forEach(file => {
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const div = document.createElement('div');
                div.className = 'preview-item';
                div.innerHTML = `
                    <img src="${e.target.result}" alt="Preview">
                    <button type="button" class="remove-btn">
                        <i class="fas fa-times"></i>
                    </button>
                `;
                preview.appendChild(div);

                div.querySelector('.remove-btn').onclick = function() {
                    div.remove();
                };
            };
            reader.readAsDataURL(file);
        }
    });
}

function handleFileUpload(files) {
    const preview = document.getElementById('filePreview');
    preview.innerHTML = '';

    Array.from(files).forEach(file => {
        const div = document.createElement('div');
        div.className = 'preview-item file-preview';
        div.innerHTML = `
            <div class="file-info">
                <i class="fas fa-file"></i>
                <span>${file.name}</span>
                <small>${formatFileSize(file.size)}</small>
            </div>
            <button type="button" class="remove-btn">
                <i class="fas fa-times"></i>
            </button>
        `;
        preview.appendChild(div);

        div.querySelector('.remove-btn').onclick = function() {
            div.remove();
        };
    });
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function setupDragAndDrop(elementId, accept) {
    const element = document.getElementById(elementId);
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        element.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        element.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        element.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        element.classList.add('drag-active');
    }

    function unhighlight(e) {
        element.classList.remove('drag-active');
    }

    element.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (elementId === 'imageUpload') {
            handleImageUpload(files);
        } else {
            handleFileUpload(files);
        }
    }
} 