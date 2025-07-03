import os
import pickle
from flask import Blueprint, request, render_template,redirect,url_for,flash
from flask_login import login_required,current_user
from my_flask_app.utils.preprocess import preprocess_user_ingredients
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import re
import ast
from . import db
from .models import Favourite
import pandas as pd

main = Blueprint('main', __name__)

def safe_literal_eval(val):
    try:
        val = re.sub(r'\s+', ' ', val)  
        return ast.literal_eval(val)  
    except (ValueError, SyntaxError):
        return val  

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "..", "data")

@main.route('/home', methods=['GET', 'POST'])
@login_required
def dish_finder():
    if request.method == 'POST':
        user_ingredients = request.form['ingredients']
        cooking_time_filter = request.form.get('cooking_time', 'all')
        return redirect(url_for('main.results', ingredients=user_ingredients, cooking_time=cooking_time_filter))
    return render_template('dish_finder.html')


@main.route('/results', methods=['GET'])
@login_required
def results():
    user_ingredients = request.args.get('ingredients')
    cooking_time_filter = request.args.get('cooking_time', 'all')
    print(f"Selected cooking time filter: {cooking_time_filter}")

    if not user_ingredients:
        return redirect(url_for('main.dish_finder'))

    print(f"Original user input: '{user_ingredients}'")
    
    ingredients_list = user_ingredients.split(',')
    ingredients_list = [ing.strip() for ing in ingredients_list]
    print(f"Individual ingredients: {ingredients_list}")
    
    preprocessed_user_ingredients = preprocess_user_ingredients(user_ingredients)
    print(f"Preprocessed user input: '{preprocessed_user_ingredients}'")

    with open(os.path.join(data_dir, "processed", "Vectorizer_names.pkl"), 'rb') as file:
        vectorizer_name = pickle.load(file)
    with open(os.path.join(data_dir, "processed", "Vectorizer_ingredients.pkl"), 'rb') as file:
        vectorizer_ing = pickle.load(file)
    with open(os.path.join(data_dir, "processed", "TFidf_matrix.pkl"), 'rb') as file:
        tfidf_combined = pickle.load(file)
    with open(os.path.join(data_dir, "processed", "Recipes.pkl"), 'rb') as file:
        recipe_df = pickle.load(file)

    has_zero_similarity = False
    threshold = 0.25
    for ingredient in ingredients_list:
        preprocessed_ingredient = preprocess_user_ingredients(ingredient)
        if preprocessed_ingredient.strip():
            ingredient_vector_name = vectorizer_name.transform([preprocessed_ingredient])
            ingredient_vector_ing = vectorizer_ing.transform([preprocessed_ingredient])
            ingredient_vector_combined = hstack([0.5 * ingredient_vector_name, 0.5 * ingredient_vector_ing])
            ingredient_similarity = cosine_similarity(ingredient_vector_combined, tfidf_combined)
            max_similarity = ingredient_similarity.max()
            print(f"Ingredient '{ingredient}' has max similarity: {max_similarity}")
            if max_similarity < threshold:
                print(f"Ingredient '{ingredient}' has similarity below threshold!")
                has_zero_similarity = True
                break

    user_query = preprocessed_user_ingredients
    user_vector_name = vectorizer_name.transform([user_query])
    user_vector_ing = vectorizer_ing.transform([user_query])
    user_vector_combined = hstack([0.5 * user_vector_name, 0.5 * user_vector_ing])

    similarity_scores = cosine_similarity(user_vector_combined, tfidf_combined)
    scores = similarity_scores.flatten()

    print(f"Similarity threshold: {threshold}")
    all_valid_indices = [idx for idx, score in enumerate(scores) if score >= threshold]
    print(f"Indices above threshold: {all_valid_indices}")

    if not all_valid_indices or has_zero_similarity:
        print("No recipes met the criteria: either below similarity threshold or ingredient with zero similarity")
        empty_recipes = pd.DataFrame(columns=recipe_df.columns)
        return render_template('res.html', recipes=empty_recipes, has_results=False, user_ingredients=user_ingredients, cooking_time_filter=cooking_time_filter)

    recommended_df = recipe_df.iloc[all_valid_indices].copy()
    recommended_df['Similarity'] = [scores[idx] for idx in all_valid_indices]

    if cooking_time_filter != 'all':
        if cooking_time_filter == 'quick':
            recommended_df = recommended_df[recommended_df['TotalTimeInMins'] <= 30]
        elif cooking_time_filter == 'medium':
            recommended_df = recommended_df[
                (recommended_df['TotalTimeInMins'] > 30) & (recommended_df['TotalTimeInMins'] <= 60)
            ]
        elif cooking_time_filter == 'long':
            recommended_df = recommended_df[recommended_df['TotalTimeInMins'] > 60]

    recommended_df = recommended_df.sort_values(by='Similarity', ascending=False)
    recommended_recipes = recommended_df.head(6).copy()

    if not recommended_recipes.empty:
        recommended_recipes.loc[:, 'Instructions'] = recommended_recipes['Instructions'].apply(safe_literal_eval)
        print(f"Final recommended recipe names: {recommended_recipes['RecipeName'].tolist()}")

    has_results = not recommended_recipes.empty

    return render_template('res.html', recipes=recommended_recipes, has_results=has_results, user_ingredients=user_ingredients, cooking_time_filter=cooking_time_filter)


@main.route('/add_to_favourites/<int:id>', methods=['GET'])
@login_required
def add_to_favourites(id):
    existing_fav = Favourite.query.filter_by(user_id=current_user.id, recipe_id=id).first()
    if not existing_fav:
        new_fav = Favourite(user_id=current_user.id, recipe_id=id)
        db.session.add(new_fav)
        db.session.commit()
        flash('Added to favourites!', 'success')
    else:
        flash('Already added to Favourites', 'info')

    next_page = request.args.get('next')
    if next_page:
        return redirect(next_page)
    return redirect(request.referrer)

@main.route('/remove_from_favourites/<int:id>', methods=['GET'])
@login_required
def remove_from_favourites(id):
    existing_fav = Favourite.query.filter_by(user_id=current_user.id, recipe_id=id).first()
    if existing_fav:
        db.session.delete(existing_fav)
        db.session.commit()
        flash('Removed from favourites!', 'success')
    else:
        flash('Some error occured', 'error')
    
    return redirect(url_for('main.favourites'))

@main.route('/recipe_details/<int:id>',methods=['GET'])
@login_required
def recipe_details(id):
    with open(os.path.join(data_dir, "processed", "Recipes.pkl"), 'rb') as file:
        recipe_df = pickle.load(file)

    recipe = recipe_df[recipe_df['Srno'] == id].iloc[0]
    recipe['Instructions'] = safe_literal_eval(recipe['Instructions'])

    return render_template('recipe_details.html',recipe=recipe)

@main.route('/favourites')
@login_required
def favourites():
    fav_entries = Favourite.query.filter_by(user_id=current_user.id).all()
    recipe_ids = [fav.recipe_id for fav in fav_entries]
    with open(os.path.join(data_dir, "processed", "Recipes.pkl"), 'rb') as file:
        recipe_df = pickle.load(file)

    favs = recipe_df[recipe_df['Srno'].isin(recipe_ids)]

    return render_template('fav.html',favs=favs)

@main.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user=current_user)

@main.route('/save_profile', methods=['POST'])
@login_required
def save_profile():
    if request.method == 'POST':
        
        phone = request.form.get('phone')
        gender = request.form.get('gender')
        
        gender_map = {
            'Male': 'M',
            'Female': 'F'
        }
        
        user = current_user
        if phone:
            user.phone = int(phone) if phone.isdigit() else None
        user.gender = gender_map.get(gender, None)
        
        db.session.commit()
        
        
        flash('Profile updated successfully!', 'success')
        
        return redirect(url_for('main.profile'))

@main.route('/discover')
@login_required
def discover():
    with open(os.path.join(data_dir, "processed", "Recipes.pkl"), 'rb') as file:
        recipe_df = pickle.load(file)
    
   
    categories = ['Lunch', 'Snack', 'Dinner', 'South Indian Breakfast', 'North Indian Breakfast', 'Brunch']
    filtered_recipes = {}
    
    for category in categories:
        category_recipes = recipe_df[recipe_df['Course'] == category]
        if len(category_recipes) > 6:
            category_recipes = category_recipes.head(6)
        filtered_recipes[category] = category_recipes
    
    return render_template('discover.html', recipes=filtered_recipes)

@main.route('/')
def home():
    return render_template('index.html')
  
