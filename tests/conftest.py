import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from my_flask_app import create_app
from my_flask_app.models import db, User
from werkzeug.security import generate_password_hash
import time

@pytest.fixture
def app():
    app = create_app()
    app.config["TESTING"] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'  
    with app.app_context():
        db.create_all() 
        yield app
        db.drop_all() 

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def login(client):
    """Logs in a user and returns the test client."""
    with client.application.app_context():
        unique_email = f'testuser_{time.time_ns()}@example.com'
        test_user = User(name='Test User', email=unique_email)
        test_user.set_password('testpassword')
        db.session.add(test_user)
        db.session.commit()

    client.post('/login', data={
        'email': unique_email,
        'password': 'testpassword'
    }, follow_redirects=True)

    return client  

