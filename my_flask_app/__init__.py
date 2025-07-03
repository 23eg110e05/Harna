from flask import Flask, session, request, redirect, url_for, g, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, current_user
from datetime import timedelta
import time

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)

    app.config['SECRET_KEY'] = 'secret-key'
    app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///project.db"
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['SESSION_COOKIE_SECURE'] = True 
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    app.config['SESSION_USE_SIGNER'] = True

    db.__init__(app)

    loginmanager=LoginManager()
    loginmanager.login_view='auth.login'
    loginmanager.init_app(app)
    loginmanager.session_protection = "strong"  # Enhanced session protection

    from .models import User
    @loginmanager.user_loader
    def load_user(user_id):
       return User.query.get(int(user_id))
    
    # Add before request handler to check and validate sessions
    @app.before_request
    def check_valid_session():
        # Only apply to authenticated requests
        if current_user.is_authenticated:
            # Refresh the login timestamp for active users
            session['login_time'] = time.time()
    
    # Add after request handler to set cache control headers
    @app.after_request
    def add_header(response):
        # Prevent caching for all authenticated routes
        if current_user.is_authenticated:
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
        return response

    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    from .auth import auth as auth_blueprint
    app.register_blueprint(auth_blueprint)

    return app