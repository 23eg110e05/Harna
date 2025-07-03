from my_flask_app.models import db, User
import time

def test_login(client):
    unique_email = f'testuser_{int(time.time())}@example.com'  
    test_user = User(name='Test User', email=unique_email)
    test_user.set_password('testpassword')
    db.session.add(test_user)
    db.session.commit()

    response = client.post('/login', data={
        'email': unique_email,  
        'password': 'testpassword'
    })
    assert response.status_code == 302
