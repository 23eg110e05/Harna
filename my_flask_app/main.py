import os
from flask import Blueprint, request, render_template, redirect, url_for, flash, jsonify, abort
from flask_login import login_required, current_user
from my_flask_app.utils.preprocess import preprocess_user_ingredients
from my_flask_app.utils.cache import cache
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import re
import ast
from . import db
from .models import Favourite
import pandas as pd

main = Blueprint('main', __name__)

def safe_literal_eval(val):
    if not isinstance(val, str):
        return val
    try:
        val = re.sub(r'\s+', ' ', val)
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val

def compute_recipe_recommendations(user_ingredients, cooking_time_filter='all', top_n=6, threshold=0.25):
    """
    Core reusable recommendation engine using the pre-cached in-memory models.
    Executes in <15ms.
    """
    if not cache._loaded:
        cache.load_models()

    ingredients_list = [ing.strip() for ing in user_ingredients.split(',') if ing.strip()]
    if not ingredients_list:
        return []

    preprocessed_user_ingredients = preprocess_user_ingredients(user_ingredients)

    vectorizer_name = cache.vectorizer_name
    vectorizer_ing = cache.vectorizer_ing
    tfidf_combined = cache.tfidf_combined
    recipe_df = cache.recipe_df

    # Verify ingredient threshold to prevent false-positive matches
    for ingredient in ingredients_list:
        preprocessed_ingredient = preprocess_user_ingredients(ingredient)
        if preprocessed_ingredient.strip():
            ingredient_vector_name = vectorizer_name.transform([preprocessed_ingredient])
            ingredient_vector_ing = vectorizer_ing.transform([preprocessed_ingredient])
            ingredient_vector_combined = hstack([0.5 * ingredient_vector_name, 0.5 * ingredient_vector_ing])
            ingredient_similarity = cosine_similarity(ingredient_vector_combined, tfidf_combined)
            if ingredient_similarity.max() < threshold:
                return []

    # Compute overall query similarity
    user_query = preprocessed_user_ingredients
    user_vector_name = vectorizer_name.transform([user_query])
    user_vector_ing = vectorizer_ing.transform([user_query])
    user_vector_combined = hstack([0.5 * user_vector_name, 0.5 * user_vector_ing])

    similarity_scores = cosine_similarity(user_vector_combined, tfidf_combined)
    scores = similarity_scores.flatten()

    all_valid_indices = [idx for idx, score in enumerate(scores) if score >= threshold]
    if not all_valid_indices:
        return []

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
    final_recipes = recommended_df.head(top_n).copy()

    if not final_recipes.empty:
        final_recipes['Instructions'] = final_recipes['Instructions'].apply(safe_literal_eval)

    return final_recipes

# -------------------------------------------------------------
# WEB ROUTES
# -------------------------------------------------------------

@main.route('/home', methods=['GET', 'POST'])
@login_required
def dish_finder():
    if request.method == 'POST':
        user_ingredients = request.form.get('ingredients', '').strip()
        cooking_time_filter = request.form.get('cooking_time', 'all')
        if not user_ingredients:
            flash('Please enter at least one ingredient.', 'warning')
            return redirect(url_for('main.dish_finder'))
        return redirect(url_for('main.results', ingredients=user_ingredients, cooking_time=cooking_time_filter))
    return render_template('dish_finder.html')


@main.route('/results', methods=['GET'])
@login_required
def results():
    user_ingredients = request.args.get('ingredients', '').strip()
    cooking_time_filter = request.args.get('cooking_time', 'all')

    if not user_ingredients:
        return redirect(url_for('main.dish_finder'))

    recommended_recipes = compute_recipe_recommendations(user_ingredients, cooking_time_filter, top_n=6)
    has_results = bool(len(recommended_recipes) > 0)

    if not has_results:
        empty_recipes = pd.DataFrame()
        return render_template('res.html', recipes=empty_recipes, has_results=False, user_ingredients=user_ingredients, cooking_time_filter=cooking_time_filter)

    return render_template('res.html', recipes=recommended_recipes, has_results=has_results, user_ingredients=user_ingredients, cooking_time_filter=cooking_time_filter)


@main.route('/recipe_details/<int:id>', methods=['GET'])
@login_required
def recipe_details(id):
    if not cache._loaded:
        cache.load_models()

    matches = cache.recipe_df[cache.recipe_df['Srno'] == id]
    if matches.empty:
        abort(404)

    recipe = matches.iloc[0].to_dict()
    recipe['Instructions'] = safe_literal_eval(recipe.get('Instructions', ''))

    return render_template('recipe_details.html', recipe=recipe)


@main.route('/add_to_favourites/<int:id>', methods=['GET'])
@login_required
def add_to_favourites(id):
    existing_fav = Favourite.query.filter_by(user_id=current_user.id, recipe_id=id).first()
    if not existing_fav:
        new_fav = Favourite(user_id=current_user.id, recipe_id=id)
        db.session.add(new_fav)
        db.session.commit()
        flash('Saved to your Harna favourites!', 'success')
    else:
        flash('Already saved in your favourites.', 'info')

    next_page = request.args.get('next')
    if next_page:
        return redirect(next_page)
    return redirect(request.referrer or url_for('main.dish_finder'))


@main.route('/remove_from_favourites/<int:id>', methods=['GET'])
@login_required
def remove_from_favourites(id):
    existing_fav = Favourite.query.filter_by(user_id=current_user.id, recipe_id=id).first()
    if existing_fav:
        db.session.delete(existing_fav)
        db.session.commit()
        flash('Removed from favourites.', 'success')
    else:
        flash('Recipe not found in your favourites.', 'error')

    return redirect(url_for('main.favourites'))


@main.route('/favourites')
@login_required
def favourites():
    fav_entries = Favourite.query.filter_by(user_id=current_user.id).all()
    recipe_ids = [fav.recipe_id for fav in fav_entries]

    if not cache._loaded:
        cache.load_models()

    favs = cache.recipe_df[cache.recipe_df['Srno'].isin(recipe_ids)]
    return render_template('fav.html', favs=favs)


@main.route('/discover')
@login_required
def discover():
    if not cache._loaded:
        cache.load_models()

    categories = ['Lunch', 'Snack', 'Dinner', 'South Indian Breakfast', 'North Indian Breakfast', 'Brunch']
    filtered_recipes = {}

    for category in categories:
        cat_matches = cache.recipe_df[cache.recipe_df['Course'] == category]
        filtered_recipes[category] = cat_matches.head(6)

    return render_template('discover.html', recipes=filtered_recipes)


@main.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user=current_user)


@main.route('/save_profile', methods=['POST'])
@login_required
def save_profile():
    phone = request.form.get('phone', '').strip()
    gender = request.form.get('gender', '').strip()

    gender_map = {'Male': 'M', 'Female': 'F', 'M': 'M', 'F': 'F'}

    if phone:
        if phone.isdigit() and len(phone) == 10:
            current_user.phone = int(phone)
        else:
            flash('Please enter a valid 10-digit phone number.', 'warning')
            return redirect(url_for('main.profile'))
    else:
        current_user.phone = None

    current_user.gender = gender_map.get(gender, None)
    db.session.commit()
    flash('Profile updated successfully!', 'success')
    return redirect(url_for('main.profile'))


@main.route('/')
def home():
    return render_template('index.html')


# -------------------------------------------------------------
# REST API ENDPOINTS (For Mobile, Sidecars & Integrations)
# -------------------------------------------------------------

@main.route('/api/recommend', methods=['POST', 'GET'])
def api_recommend():
    """
    JSON API for recipe recommendations.
    Accepts: { "ingredients": "paneer, tomato", "cooking_time": "quick|medium|long|all", "limit": 6 }
    """
    if request.method == 'POST':
        data = request.get_json(silent=True) or request.form
        ingredients = data.get('ingredients', '')
        cooking_time = data.get('cooking_time', 'all')
        limit = int(data.get('limit', 6))
    else:
        ingredients = request.args.get('ingredients', '')
        cooking_time = request.args.get('cooking_time', 'all')
        limit = int(request.args.get('limit', 6))

    if not ingredients.strip():
        return jsonify({
            'success': False,
            'error': 'Missing required parameter: ingredients',
            'results': []
        }), 400

    results_df = compute_recipe_recommendations(ingredients, cooking_time, top_n=limit)

    if isinstance(results_df, pd.DataFrame) and not results_df.empty:
        records = results_df.to_dict(orient='records')
        # Clean NaN values for valid JSON output
        for r in records:
            for k, v in r.items():
                if pd.isna(v):
                    r[k] = None
        return jsonify({
            'success': True,
            'count': len(records),
            'query': {
                'ingredients': ingredients,
                'cooking_time': cooking_time
            },
            'results': records
        })

    return jsonify({
        'success': True,
        'count': 0,
        'query': {
            'ingredients': ingredients,
            'cooking_time': cooking_time
        },
        'results': []
    })


@main.route('/api/recipes/<int:recipe_id>', methods=['GET'])
def api_recipe_detail(recipe_id):
    """
    JSON API to fetch full recipe details by ID.
    """
    recipe = cache.get_recipe_by_id(recipe_id)
    if not recipe:
        return jsonify({'success': False, 'error': f'Recipe ID {recipe_id} not found'}), 404

    # Format instructions and clean NaN
    recipe['Instructions'] = safe_literal_eval(recipe.get('Instructions', ''))
    for k, v in recipe.items():
        if pd.isna(v):
            recipe[k] = None

    return jsonify({
        'success': True,
        'recipe': recipe
    })


@main.route('/api/health', methods=['GET'])
def api_health():
    """
    System health status endpoint.
    """
    return jsonify({
        'status': 'healthy',
        'service': 'Harna Recipe Intelligence',
        'models_cached': cache._loaded,
        'recipe_count': len(cache.recipe_df) if cache._loaded else 0
    })
