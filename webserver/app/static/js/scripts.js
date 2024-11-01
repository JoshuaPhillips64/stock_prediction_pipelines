// app/static/js/scripts.js

document.addEventListener('DOMContentLoaded', function () {
    // Accordion Icon Toggle
    const accordionButtons = document.querySelectorAll('.accordion-button');

    accordionButtons.forEach(button => {
        button.addEventListener('click', function () {
            const icon = this.querySelector('i');
            if (this.classList.contains('collapsed')) {
                icon.classList.remove('fa-minus');
                icon.classList.add('fa-plus');
            } else {
                icon.classList.remove('fa-plus');
                icon.classList.add('fa-minus');
            }
        });
    });

    // Prediction Form Submission with reCAPTCHA
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(event) {
            event.preventDefault(); // Prevent form from submitting immediately
            grecaptcha.ready(function() {
                grecaptcha.execute('{{ recaptcha_site_key }}', {action: 'prediction'}).then(function(token) {
                    document.getElementById('recaptcha_token').value = token;
                    predictionForm.submit();
                });
            });
        });
    }

    // Contact Form Submission with reCAPTCHA (if applicable)
    const contactForm = document.getElementById('contactForm');
    if (contactForm) {
        contactForm.addEventListener('submit', function(event) {
            event.preventDefault(); // Prevent form from submitting immediately
            grecaptcha.ready(function() {
                grecaptcha.execute('{{ recaptcha_site_key }}', {action: 'contact'}).then(function(token) {
                    document.getElementById('recaptcha_token').value = token;
                    contactForm.submit();
                });
            });
        });
    }
});