def test_recipe_recommendation(login):
    response1 = login.get("/results?ingredients=chicken,egg", follow_redirects=True)
    assert response1.status_code == 200
    assert b"Chicken And Egg Soup Recipe" in response1.data
    assert b"Fried Egg Recipe - Sunny Side Up" in response1.data
    assert b"Egg Pakora Recipe - Egg Fritters" in response1.data
    assert b"Boiled Egg With Salt And Pepper Recipe - Finger Food For Babies Above 9 Months" in response1.data
    assert b"Spinach Egg Muffins Recipe" in response1.data
    assert b"Egg Dosa Recipe" in response1.data

    response2 = login.get("/results?ingredients=paneer,butter", follow_redirects=True)
    assert response2.status_code == 200
    assert b"Paneer Butter Masala Recipe" in response2.data
    assert b"Paneer Matar Butter Masala (Indian Cottage Cheese and Peas Masala With Butter) Recipe" in response2.data
    assert b"Lahsuni Paneer Recipe - Paneer Flavoured With Garlic" in response2.data
    assert b"Layered Paneer Butter Masala Biryani Recipe" in response2.data
    assert b"Crispy Paneer Pakora Recipe With Garlic Chutney" in response2.data
    assert b"Paneer Capsicum Sandwich Recipe" in response2.data

    response3 = login.get("/results?ingredients=dog,cat", follow_redirects=True)
    assert response3.status_code == 200
    assert b"Dog curry" not in response3.data
    assert b"Cat playing" not in response3.data
    #assert b"No matches found" in response3.data  
   
    response4 = login.get("/results?ingredients=maida,cheese", follow_redirects=True)
    assert response4.status_code == 200
    assert b"Heart Shaped Sugar Cookies Recipe" in response4.data 

    response5 = login.get("/results?ingredients=laptop,computer", follow_redirects=True)
    assert response5.status_code == 200
    assert b"Laptop features" not in response5.data
    assert b"Computer model" not in response5.data
    # assert b"No matches found" in response5.data

    response6 = login.get("/results?ingredients=rice,dal", follow_redirects=True)
    assert response6.status_code == 200
    assert b"Chilka Roti Recipe (Jharkhand Style Rice and Lentil Roti)" in response6.data
    assert b"Panchmel Dal Recipe | Rajasthani Dal | Panchkuti Dal" in response6.data
    assert b"How To Make Homemade Idiyappam Recipe - Rice Sevai Recipe" in response6.data
    assert b"Chana Dal Khichdi Recipe" in response6.data
    assert b"Homemade Rice Puttu Recipe | Kerala Matta Rice or Basmati Rice" in response6.data
    assert b"Jeera Rice Recipe - Cumin And Ghee Flavored Rice" in response6.data

    response7 = login.get("/results?ingredients=shirts,pants", follow_redirects=True)
    assert response7.status_code == 200
    assert b"pants for kids" not in response7.data
    assert b"Shirts for men" not in response7.data
    assert b"No matches found" in response5.data

    response8 = login.get("/results?ingredients=rava,rice,oil", follow_redirects=True)
    assert response8.status_code == 200
    assert b"Rava Chakli Recipe" in response8.data
    assert b"Kara Kadubu Recipe (Malanad Style Steamed Spiced Rice Dumplings)" in response8.data
    assert b"Andhra Style Uppu Pindi Recipe (Rice Rava and Moong Dal Pudding Recipe)" in response8.data
    assert b"Nagli Rava Upma Recipe - Healthy Ragi Rava Upma Recipe" in response8.data
    assert b"Togarikalu Akki Recipe - Karnataka Style Rice Rava Upma with Fresh Pigeon Peas" in response8.data
    assert b"Dibba Rotti/Minapa Rotti Recipe (Pan Idli Recipe)" in response8.data
