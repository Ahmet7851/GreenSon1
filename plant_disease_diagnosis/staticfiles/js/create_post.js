// Quill editörünü başlat
var quill = new Quill('#editor', {
    theme: 'snow',
    placeholder: 'Sorununuzu detaylı bir şekilde açıklayın...',
    modules: {
        toolbar: [
            ['bold', 'italic', 'underline'],
            [{ 'list': 'ordered'}, { 'list': 'bullet' }],
            ['clean']
        ]
    }
});

// Form gönderildiğinde içeriği gizli alana aktar
document.getElementById('postForm').addEventListener('submit', function() {
    var content = document.getElementById('content');
    content.value = quill.root.innerHTML;
}); 