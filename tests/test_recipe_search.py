def test_recipe_recommendation(login):
    # 1. Chicken & Egg search
    response1 = login.get("/results?ingredients=chicken,egg", follow_redirects=True)
    assert response1.status_code == 200
    assert b"Chicken" in response1.data or b"Egg" in response1.data
    assert b"100% Available" in response1.data

    # 2. Paneer & Butter search
    response2 = login.get("/results?ingredients=paneer,butter", follow_redirects=True)
    assert response2.status_code == 200
    assert b"Paneer" in response2.data

    # 3. Non-food / animal search
    response3 = login.get("/results?ingredients=dog,cat", follow_redirects=True)
    assert response3.status_code == 200
    assert b"Dog curry" not in response3.data
    assert b"Cat playing" not in response3.data
   
    # 4. Electronics / Nonsense search
    response5 = login.get("/results?ingredients=laptop,computer", follow_redirects=True)
    assert response5.status_code == 200
    assert b"Laptop features" not in response5.data
    assert b"Computer model" not in response5.data

    # 5. Rice & Dal search
    response6 = login.get("/results?ingredients=rice,dal", follow_redirects=True)
    assert response6.status_code == 200
    assert b"Rice" in response6.data or b"Dal" in response6.data or b"Sadam" in response6.data

    # 6. Clothes search
    response7 = login.get("/results?ingredients=shirts,pants", follow_redirects=True)
    assert response7.status_code == 200
    assert b"pants for kids" not in response7.data
    assert b"Shirts for men" not in response7.data

    # 7. Rava, Rice search
    response8 = login.get("/results?ingredients=rava,rice,oil", follow_redirects=True)
    assert response8.status_code == 200
    assert b"Rice" in response8.data or b"Rava" in response8.data or b"Dosa" in response8.data or b"Upma" in response8.data
