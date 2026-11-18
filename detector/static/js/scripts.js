document.addEventListener('DOMContentLoaded', function () {
    const newsField = document.querySelector('textarea');
    if (newsField) {
        newsField.addEventListener('focus', function () {
            this.style.borderColor = '#4134ff';
        });
        newsField.addEventListener('blur', function () {
            this.style.borderColor = '';
        });
    }
});