from flask import render_template
from .routes import main_bp

errors_bp = main_bp

@errors_bp.app_errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@errors_bp.app_errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500