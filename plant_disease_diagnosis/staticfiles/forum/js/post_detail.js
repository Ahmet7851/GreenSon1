document.addEventListener('DOMContentLoaded', function() {
    let currentCommentId = null;

    // Fonksiyonları global scope'a ekle
    window.confirmDelete = function(commentId) {
        const modal = document.getElementById(`deleteModal-${commentId}`);
        if (modal) {
            modal.style.display = 'flex';
            document.body.style.overflow = 'hidden';
        }
    };

    window.closeDeleteModal = function(commentId) {
        const modal = document.getElementById(`deleteModal-${commentId}`);
        if (modal) {
            modal.style.display = 'none';
            document.body.style.overflow = '';
        }
    };

    window.deleteComment = function(commentId) {
        fetch(`/forum/comment/${commentId}/delete/`, {
            method: 'POST',
            headers: {
                'X-CSRFToken': getCookie('csrftoken')
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const commentDiv = document.getElementById(`comment-${commentId}`);
                if (commentDiv) {
                    commentDiv.remove();
                }
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    };

    window.editComment = function(commentId) {
        const contentDiv = document.getElementById(`comment-content-${commentId}`);
        const editForm = document.getElementById(`comment-edit-form-${commentId}`);
        
        if (contentDiv && editForm) {
            contentDiv.style.display = 'none';
            editForm.style.display = 'block';
            
            // Textarea'yı seç ve içeriğini al
            const textarea = editForm.querySelector('.edit-textarea');
            if (textarea) {
                textarea.value = contentDiv.textContent.trim();
                textarea.focus();
            }
        }
    };

    window.saveComment = function(commentId) {
        const editForm = document.getElementById(`comment-edit-form-${commentId}`);
        const textarea = editForm.querySelector('.edit-textarea');
        const content = textarea.value.trim();
        
        if (!content) return;
        
        fetch(`/forum/comment/${commentId}/edit/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')
            },
            body: JSON.stringify({ content: content })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const contentDiv = document.getElementById(`comment-content-${commentId}`);
                contentDiv.textContent = content;
                cancelEdit(commentId);
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    };

    window.cancelEdit = function(commentId) {
        const contentDiv = document.getElementById(`comment-content-${commentId}`);
        const editForm = document.getElementById(`comment-edit-form-${commentId}`);
        
        if (contentDiv && editForm) {
            contentDiv.style.display = 'block';
            editForm.style.display = 'none';
        }
    };

    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }

    // Modal dışına tıklandığında kapatma
    const modal = document.getElementById('deleteModal');
    
    window.addEventListener('click', function(e) {
        if (e.target === modal) {
            closeDeleteModal();
        }
    });

    // ESC tuşu ile kapatma
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && modal.classList.contains('show')) {
            closeDeleteModal();
        }
    });
}); 