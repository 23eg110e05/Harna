import os
import time
from datetime import timedelta
from flask import Flask, session, request, redirect, url_for, g, flash, render_template, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, current_user
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)

    # Security & Environment Config
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'harna-fallback-secret-key-382910')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', "sqlite:///project.db")
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SESSION_TYPE'] = 'filesystem'
    
    # Configure session cookie security based on environment
    cookie_secure = os.environ.get('SESSION_COOKIE_SECURE', 'False').lower() in ('true', '1', 't')
    app.config['SESSION_COOKIE_SECURE'] = cookie_secure
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    app.config['SESSION_USE_SIGNER'] = True

    db.init_app(app)

    loginmanager = LoginManager()
    loginmanager.login_view = 'auth.login'
    loginmanager.init_app(app)
    loginmanager.session_protection = "strong"

    from .models import User
    @loginmanager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))
    
    # Pre-cache ML models and recipe database at startup
    from .utils.cache import cache
    try:
        cache.load_models()
    except Exception as e:
        print(f"[Harna Warning] Deferred model loading: {e}")

    # Session activity timestamp
    @app.before_request
    def check_valid_session():
        if current_user.is_authenticated:
            session['login_time'] = time.time()
    
    # Cache control headers
    @app.after_request
    def add_security_headers(response):
        if current_user.is_authenticated:
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
        # Basic security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        return response

    # 404 & 500 Error Handlers
    @app.errorhandler(404)
    def page_not_found(e):
        if request.path.startswith('/api/'):
            return jsonify({'success': False, 'error': 'Endpoint not found', 'status_code': 404}), 404
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def server_error(e):
        if request.path.startswith('/api/'):
            return jsonify({'success': False, 'error': 'Internal server error', 'status_code': 500}), 500
        return render_template('500.html'), 500

    # Register Blueprints
    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    from .auth import auth as auth_blueprint
    app.register_blueprint(auth_blueprint)

    return app