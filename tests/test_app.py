import pytest
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
    """Create a test user and log in with a unique email."""
    unique_email = f'testuser_{int(time.time())}@example.com'  
    test_user = User(name='Test User', email=unique_email)
    test_user.set_password('testpassword')  # Hash the password
    db.session.add(test_user)
    db.session.commit()
    
    # Perform login
    client.post('/login', data={
        'email': unique_email,  # Use the correct unique email for login
        'password': 'testpassword'
    })
    yield client

def test_home(client):
    response = client.get("/")
    assert response.status_code == 200

def test_recipe_recommendation(login):
    response1 = login.get("/results?ingredients=chicken,egg", follow_redirects=True)
    assert response1.status_code == 200
    assert b"Chicken And Egg Soup Recipe" in response1.data
    assert b"Fried Egg Recipe - Sunny Side Up" in response1.data
    assert b"Egg Pakora Recipe - Egg Fritters" in response1.data
    assert b"Boiled Egg With Salt And Pepper Recipe - Finger Food For Babies Above 9 Months" in response1.data

    response2 = login.get("/results?ingredients=paneer,butter", follow_redirects=True)
    assert response2.status_code == 200
    assert b"Paneer Butter Masala Recipe" in response2.data
    assert b"Paneer Matar Butter Masala (Indian Cottage Cheese and Peas Masala With Butter) Recipe" in response2.data
    assert b"Lahsuni Paneer Recipe - Paneer Flavoured With Garlic" in response2.data
    assert b"Layered Paneer Butter Masala Biryani Recipe" in response2.data

    response3 = login.get("/results?ingredients=dog", follow_redirects=True)
    assert response3.status_code == 200
    assert b"Dog curry" not in response3.data
   
    response4 = login.get("/results?ingredients=maida,cheese", follow_redirects=True)
    assert response4.status_code == 200
    assert b"Heart Shaped Sugar Cookies Recipe" in response4.data 

    response5 = login.get("/results?ingredients=laptop", follow_redirects=True)
    assert response5.status_code == 200
    assert b"Laptop features" not in response5.data

    response6 = login.get("/results?ingredients=paneer,butter", follow_redirects=True)
    assert response6.status_code == 200
    assert b"Chilka Roti Recipe (Jharkhand Style Rice and Lentil Roti)" in response6.data
    assert b"Panchmel Dal Recipe | Rajasthani Dal | Panchkuti Dal" in response6.data
    assert b"Lahsuni Paneer Recipe - Paneer Flavoured With Garlic" in response6.data
    assert b"Layered Paneer Butter Masala Biryani Recipe" in response6.data
    assert b"Lahsuni Paneer Recipe - Paneer Flavoured With Garlic" in response6.data
    assert b"Layered Paneer Butter Masala Biryani Recipe" in response6.data