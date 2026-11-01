import pytest
from my_flask_app import create_app
from my_flask_app.models import db, User, MealPlan
import time
from datetime import date, timedelta

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
def authenticated_client(client):
    unique_email = f'chef_{int(time.time() * 1000)}@example.com'
    test_user = User(name='Chef Tester', email=unique_email)
    test_user.set_password('pass123')
    db.session.add(test_user)
    db.session.commit()

    client.post('/login', data={
        'email': unique_email,
        'password': 'pass123'
    })
    return client

def test_planner_unauthenticated_redirect(client):
    res = client.get('/planner')
    assert res.status_code == 302
    assert '/login' in res.location

def test_planner_authenticated_view(authenticated_client):
    res = authenticated_client.get('/planner')
    assert res.status_code == 200
    assert b"Weekly Meal Planner" in res.data
    assert b"Auto-Fill Week" in res.data
    assert b"Smart Grocery" in res.data

def test_api_planner_search(authenticated_client):
    res = authenticated_client.get('/api/planner/search-recipes?q=karela&limit=5')
    assert res.status_code == 200
    json_data = res.get_json()
    assert json_data['success'] is True
    assert json_data['count'] > 0
    assert any('Karela' in r['name'] for r in json_data['recipes'])

def test_planner_add_and_remove_meal(authenticated_client):
    today_str = date.today().strftime('%Y-%m-%d')
    # Add Karela (Srno=1)
    res_add = authenticated_client.post('/api/planner/add', json={
        'recipe_id': 1,
        'date': today_str,
        'meal_type': 'lunch',
        'servings': 4
    })
    assert res_add.status_code == 200
    add_data = res_add.get_json()
    assert add_data['success'] is True
    plan_id = add_data['id']

    # Verify meal appears on planner page
    res_page = authenticated_client.get(f'/planner?week={today_str}')
    assert res_page.status_code == 200
    assert b"Karela" in res_page.data

    # Update servings
    res_serv = authenticated_client.post(f'/api/planner/update-servings/{plan_id}', json={
        'servings': 6
    })
    assert res_serv.status_code == 200
    assert res_serv.get_json()['servings'] == 6

    # Remove meal
    res_del = authenticated_client.post(f'/api/planner/remove/{plan_id}')
    assert res_del.status_code == 200
    assert res_del.get_json()['success'] is True

def test_planner_auto_generation_and_grocery(authenticated_client):
    today_str = date.today().strftime('%Y-%m-%d')

    # Auto generate a vegetarian week
    res_gen = authenticated_client.post('/api/planner/generate', json={
        'diet': 'vegetarian',
        'week': today_str
    })
    assert res_gen.status_code == 200
    gen_data = res_gen.get_json()
    assert gen_data['success'] is True
    assert gen_data['count'] == 28

    # Check that grocery list was populated with aisle categories
    res_groc = authenticated_client.get(f'/planner/grocery?week={today_str}')
    assert res_groc.status_code == 200
    assert b"Smart Grocery List" in res_groc.data
    assert b"Contributing Recipes" in res_groc.data

    # Clear the week
    res_clear = authenticated_client.post('/api/planner/clear', json={
        'week': today_str
    })
    assert res_clear.status_code == 200
    assert res_clear.get_json()['success'] is True
