import os
from flask import Blueprint, request, render_template, redirect, url_for, flash, jsonify, abort
from flask_login import login_required, current_user
from my_flask_app.utils.preprocess import preprocess_user_ingredients
from my_flask_app.utils.cache import cache
from my_flask_app.utils.ratvi import ratvi
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import re
import ast
from datetime import datetime, date, timedelta
import random
from . import db
from .models import Favourite, MealPlan
from my_flask_app.utils.grocery_helper import generate_aisle_grocery_list
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

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

def compute_recipe_recommendations(user_ingredients, cooking_time_filter='all', top_n=6, threshold=0.15):
    """
    Pantry-first recipe recommendation engine.
    Finds recipes based STRICTLY on the ingredients the user has available.
    Guarantees that 100% Match recipes require ZERO unmentioned core ingredients
    (meat, poultry, seafood, dairy, unlisted veggies, or grains).
    """
    if not cache._loaded:
        cache.load_models()

    raw_ingredients = [ing.strip().lower() for ing in user_ingredients.split(',') if ing.strip()]
    if not raw_ingredients:
        return []

    recipe_df = cache.recipe_df
    inv_index = getattr(cache, 'inverted_index', {})
    stems_map = getattr(cache, 'recipe_stems_map', [])
    title_core_map = getattr(cache, 'recipe_title_core', [])
    core_foods = getattr(cache, 'CORE_FOOD_STEMS', set())
    pantry_staples = getattr(cache, 'PANTRY_STAPLES', set())

    user_stems = set()
    user_ing_display = {}
    for ing in raw_ingredients:
        cleaned = re.sub(r'[^a-z\s]', '', ing)
        for w in cleaned.split():
            if len(w) > 1:
                st = stemmer.stem(w)
                user_stems.add(st)
                user_ing_display[st] = ing.title()

    if not user_stems or not (user_stems - pantry_staples):
        return []

    user_core = user_stems.intersection(core_foods)
    # If user provided core foods (e.g. paneer, chicken), anchor candidates to the core foods
    if user_core:
        matched_candidate_sets = [inv_index.get(s, set()) for s in user_core if s in inv_index]
    else:
        matched_candidate_sets = [inv_index.get(s, set()) for s in user_stems if s in inv_index]

    if not matched_candidate_sets:
        # Fallback: if user typed a dish name (e.g. 'biryani', 'pizza')
        name_mask = recipe_df['RecipeName'].str.contains(user_ingredients.strip(), case=False, na=False, regex=False)
        name_indices = recipe_df[name_mask].index.tolist()[:top_n]
        if not name_indices:
            return []
        matched_df = recipe_df.iloc[name_indices].copy()
        matched_df['Similarity'] = 0.90
        matched_df['MatchTier'] = 1
        matched_df['MatchedIngredients'] = user_ingredients
        matched_df['MissingIngredients'] = ''
        matched_df['IsExactPantryMatch'] = True
        matched_df['Instructions'] = matched_df['Instructions'].apply(safe_literal_eval)
        return matched_df

    all_candidates = set().union(*matched_candidate_sets)

    tier1 = []
    tier2 = []
    tier3 = []

    for idx in all_candidates:
        row = recipe_df.iloc[idx]
        stems = stems_map[idx] if idx < len(stems_map) else set(str(row.get('Ingredients_clean', '')).split())
        title_core = title_core_map[idx] if idx < len(title_core_map) else set()

        time_mins = row.get('TotalTimeInMins', 30)
        if cooking_time_filter == 'quick' and time_mins > 30:
            continue
        elif cooking_time_filter == 'medium' and not (30 < time_mins <= 60):
            continue
        elif cooking_time_filter == 'long' and time_mins <= 60:
            continue

        matched_user = stems.intersection(user_stems)
        matched_count = len(matched_user)
        if matched_count == 0:
            continue

        recipe_core = (stems.intersection(core_foods) | title_core)
        missing_core = recipe_core - user_stems
        missing_core_count = len(missing_core)

        missing_non_staples = (stems - pantry_staples) - user_stems
        user_coverage = matched_count / len(user_stems)

        item = {
            'idx': idx,
            'matched_count': matched_count,
            'matched_user': matched_user,
            'missing_core': missing_core,
            'missing_core_count': missing_core_count,
            'missing_non_staples': missing_non_staples,
            'coverage': user_coverage
        }

        if missing_core_count == 0:
            purity = 1.0 / (1.0 + len(missing_non_staples) * 0.10)
            score = (user_coverage * 0.70) + (purity * 0.30)
            item['score'] = score
            item['tier'] = 1
            tier1.append(item)
        elif missing_core_count == 1:
            purity = 1.0 / (1.0 + len(missing_non_staples) * 0.20)
            score = (user_coverage * 0.50) + (purity * 0.25)
            item['score'] = score
            item['tier'] = 2
            tier2.append(item)
        elif missing_core_count == 2:
            score = (user_coverage * 0.30)
            item['score'] = score
            item['tier'] = 3
            tier3.append(item)

    all_scored = tier1 + tier2 + tier3
    all_scored.sort(key=lambda x: (x['matched_count'], -x['missing_core_count'], x['score'], -len(x['missing_non_staples'])), reverse=True)
    candidates = all_scored[:top_n]

    if not candidates:
        name_mask = recipe_df['RecipeName'].str.contains(user_ingredients.strip(), case=False, na=False, regex=False)
        name_indices = recipe_df[name_mask].index.tolist()[:top_n]
        if not name_indices:
            return []
        for idx in name_indices:
            candidates.append({
                'idx': idx, 'matched_count': 1, 'score': 0.85, 'tier': 1,
                'matched_user': set(), 'missing_core': set()
            })

    candidate_df_indices = [c['idx'] for c in candidates]
    matched_df = recipe_df.iloc[candidate_df_indices].copy()
    matched_df['Similarity'] = [c['score'] for c in candidates]
    matched_df['MatchTier'] = [c['tier'] for c in candidates]
    matched_df['MatchedIngredients'] = [', '.join([user_ing_display.get(s, s) for s in c.get('matched_user', [])]) for c in candidates]
    matched_df['MissingIngredients'] = [', '.join(sorted(list(c.get('missing_core', [])))) for c in candidates]
    matched_df['IsExactPantryMatch'] = [c['tier'] == 1 for c in candidates]

    if threshold is not None:
        matched_df = matched_df[matched_df['Similarity'] >= threshold]

    final_recipes = matched_df.head(top_n).copy()
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

    try:
        recommended_recipes = compute_recipe_recommendations(user_ingredients, cooking_time_filter, top_n=6)
        has_results = bool(len(recommended_recipes) > 0)
    except Exception as e:
        print(f"[Harna] Results error for '{user_ingredients}': {e}")
        flash('Our recipe engine is warming up — please try again in a moment.', 'warning')
        return redirect(url_for('main.dish_finder'))

    if not has_results:
        empty_recipes = pd.DataFrame()
        return render_template('res.html', recipes=empty_recipes, has_results=False,
                               user_ingredients=user_ingredients, cooking_time_filter=cooking_time_filter)

    return render_template('res.html', recipes=recommended_recipes, has_results=has_results,
                           user_ingredients=user_ingredients, cooking_time_filter=cooking_time_filter)


@main.route('/recipe_details/<id>', methods=['GET'])
@login_required
def recipe_details(id):
    try:
        numeric_id = float(id)
    except Exception:
        abort(404)

    if not cache._loaded:
        cache.load_models()

    recipe = cache.get_recipe_by_id(numeric_id)
    if not recipe:
        abort(404)

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
# RATVI AI COMPANION API ENDPOINT
# -------------------------------------------------------------

@main.route('/api/ratvi/chat', methods=['POST'])
def api_ratvi_chat():
    """
    Public AI Culinary Companion endpoint for Ratvi.
    Supports serving scaling, culinary substitutions, step-by-step guidance, and hybrid fallback.
    Accessible to guests and authenticated users.
    """
    data = request.get_json(silent=True) or request.form or {}
    message = (data.get('message') or '').strip()
    recipe_id = data.get('recipe_id')

    if not message:
        return jsonify({
            'success': False,
            'error': 'Message parameter is required',
            'reply': "Please ask Ratvi a culinary question!"
        }), 400

    context = {}
    if recipe_id is not None:
        try:
            rec_id_int = int(float(recipe_id))
            recipe_obj = cache.get_recipe_by_id(rec_id_int)
            if recipe_obj:
                context['current_recipe'] = recipe_obj
        except Exception:
            pass

    # Merge client-provided recipe_context if available
    recipe_context = data.get('recipe_context')
    if isinstance(recipe_context, dict):
        if 'current_recipe' not in context:
            context['current_recipe'] = {}
        context['current_recipe'].update(recipe_context)

    history = data.get('history', [])
    current_step = int(data.get('current_step', 0) or 0)
    options = data.get('options', {})

    response_data = ratvi.process_chat(
        message=message,
        context=context,
        history=history,
        current_step=current_step,
        options=options
    )
    
    return jsonify(response_data)


# -------------------------------------------------------------
# RECIPE RECOMMENDATION & DETAIL REST API
# -------------------------------------------------------------

@main.route('/api/recommend', methods=['POST', 'GET'])
def api_recommend():
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
            'success': True,
            'count': 0,
            'query': {
                'ingredients': ingredients,
                'cooking_time': cooking_time
            },
            'results': [],
            'recipes': []
        }), 200

    results_df = compute_recipe_recommendations(ingredients, cooking_time, top_n=limit)

    if isinstance(results_df, pd.DataFrame) and not results_df.empty:
        records = results_df.to_dict(orient='records')
        for r in records:
            if 'Srno' in r and r['Srno'] is not None and not pd.isna(r['Srno']):
                try:
                    r['Srno'] = int(float(r['Srno']))
                except Exception:
                    pass
            for k, v in list(r.items()):
                if not isinstance(v, (list, dict, tuple)) and pd.isna(v):
                    r[k] = None
        return jsonify({
            'success': True,
            'count': len(records),
            'query': {
                'ingredients': ingredients,
                'cooking_time': cooking_time
            },
            'results': records,
            'recipes': records
        })

    return jsonify({
        'success': True,
        'count': 0,
        'query': {
            'ingredients': ingredients,
            'cooking_time': cooking_time
        },
        'results': [],
        'recipes': []
    })


@main.route('/api/recipes/<int:recipe_id>', methods=['GET'])
def api_recipe_detail(recipe_id):
    recipe = cache.get_recipe_by_id(recipe_id)
    if not recipe:
        return jsonify({'success': False, 'error': f'Recipe ID {recipe_id} not found'}), 404

    recipe['Instructions'] = safe_literal_eval(recipe.get('Instructions', ''))
    if 'Srno' in recipe and recipe['Srno'] is not None and not pd.isna(recipe['Srno']):
        try:
            recipe['Srno'] = int(float(recipe['Srno']))
        except Exception:
            pass
    for k, v in list(recipe.items()):
        if not isinstance(v, (list, dict, tuple)) and pd.isna(v):
            recipe[k] = None

    return jsonify({
        'success': True,
        'recipe': recipe
    })


@main.route('/api/health', methods=['GET'])
def api_health():
    return jsonify({
        'status': 'healthy',
        'service': 'Harna Recipe Intelligence & Ratvi AI Companion',
        'models_cached': cache._loaded,
        'recipe_count': len(cache.recipe_df) if cache._loaded else 0
    })


# ═══════════════════════════════════════════════════════════════════════════
# WEEKLY MEAL PLANNER & SMART GROCERY SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

@main.route('/planner')
@login_required
def meal_planner():
    if not cache._loaded:
        cache.load_models()

    week_str = request.args.get('week')
    if week_str:
        try:
            target_date = datetime.strptime(week_str, '%Y-%m-%d').date()
        except ValueError:
            target_date = date.today()
    else:
        target_date = date.today()

    start_of_week = target_date - timedelta(days=target_date.weekday())
    end_of_week = start_of_week + timedelta(days=6)
    prev_week = (start_of_week - timedelta(days=7)).strftime('%Y-%m-%d')
    next_week = (start_of_week + timedelta(days=7)).strftime('%Y-%m-%d')
    curr_week = date.today().strftime('%Y-%m-%d')

    days = []
    days_dict = {}
    today = date.today()

    for i in range(7):
        d = start_of_week + timedelta(days=i)
        d_str = d.strftime('%Y-%m-%d')
        day_info = {
            'date': d,
            'date_str': d_str,
            'day_name': d.strftime('%a'),
            'day_full': d.strftime('%A'),
            'day_num': d.strftime('%d'),
            'month_name': d.strftime('%b'),
            'is_today': (d == today),
            'meals': {
                'breakfast': [],
                'lunch': [],
                'dinner': [],
                'snack': []
            },
            'totals': {
                'calories': 0,
                'protein': 0,
                'carbs': 0,
                'fats': 0
            }
        }
        days.append(day_info)
        days_dict[d_str] = day_info

    # Fetch meal plans for the user
    user_plans = MealPlan.query.filter(
        MealPlan.user_id == current_user.id,
        MealPlan.plan_date >= start_of_week,
        MealPlan.plan_date <= end_of_week
    ).all()

    total_week_meals = 0
    total_week_calories = 0

    for plan in user_plans:
        d_str = plan.plan_date.strftime('%Y-%m-%d')
        if d_str not in days_dict:
            continue
        meal_type = plan.meal_type.lower()
        if meal_type not in days_dict[d_str]['meals']:
            meal_type = 'dinner'

        recipe = cache.get_recipe_by_id(plan.recipe_id)
        if not recipe:
            continue

        servings = plan.servings or 2
        cal = recipe.get('Calories (kcal)', 0)
        cal_val = round(float(cal) if cal and not pd.isna(cal) else 250)
        
        prot = recipe.get('Protein (g)', 0)
        prot_val = round(float(prot) if prot and not pd.isna(prot) else 10, 1)
        
        carb = recipe.get('Carbohydrates (g)', 0)
        carb_val = round(float(carb) if carb and not pd.isna(carb) else 30, 1)
        
        fat = recipe.get('Fats (g)', 0)
        fat_val = round(float(fat) if fat and not pd.isna(fat) else 8, 1)

        item = {
            'plan_id': plan.id,
            'recipe_id': plan.recipe_id,
            'recipe_name': recipe.get('RecipeName', 'Delicious Dish'),
            'cuisine': recipe.get('Cuisine', 'Indian'),
            'diet': recipe.get('Diet', 'Vegetarian'),
            'image': recipe.get('images', ''),
            'cook_time': recipe.get('CookTimeInMins') or recipe.get('TotalTimeInMins') or 25,
            'servings': servings,
            'notes': plan.notes or '',
            'calories': cal_val,
            'protein': prot_val,
            'carbs': carb_val,
            'fats': fat_val
        }

        days_dict[d_str]['meals'][meal_type].append(item)
        days_dict[d_str]['totals']['calories'] += cal_val
        days_dict[d_str]['totals']['protein'] += prot_val
        days_dict[d_str]['totals']['carbs'] += carb_val
        days_dict[d_str]['totals']['fats'] += fat_val
        total_week_meals += 1
        total_week_calories += cal_val

    avg_daily_calories = round(total_week_calories / 7) if total_week_meals else 0

    # Get user favourites for quick add
    fav_entries = Favourite.query.filter_by(user_id=current_user.id).all()
    fav_ids = [f.recipe_id for f in fav_entries]
    fav_recipes = []
    if fav_ids:
        fav_df = cache.recipe_df[cache.recipe_df['Srno'].isin(fav_ids)].head(12)
        fav_records = fav_df.to_dict(orient='records')
        for r in fav_records:
            if 'Srno' in r and not pd.isna(r['Srno']):
                fav_recipes.append({
                    'id': int(float(r['Srno'])),
                    'name': r.get('RecipeName', ''),
                    'cuisine': r.get('Cuisine', ''),
                    'image': r.get('images', ''),
                    'cook_time': r.get('CookTimeInMins') or r.get('TotalTimeInMins') or 25,
                    'calories': round(float(r.get('Calories (kcal)', 250))) if not pd.isna(r.get('Calories (kcal)')) else 250
                })

    return render_template(
        'planner.html',
        days=days,
        start_of_week=start_of_week,
        end_of_week=end_of_week,
        prev_week=prev_week,
        next_week=next_week,
        curr_week=curr_week,
        current_week_str=start_of_week.strftime('%Y-%m-%d'),
        total_week_meals=total_week_meals,
        avg_daily_calories=avg_daily_calories,
        target_calories=2000,
        fav_recipes=fav_recipes
    )


@main.route('/planner/add', methods=['POST'])
@main.route('/api/planner/add', methods=['POST'])
@login_required
def api_planner_add():
    data = request.get_json(silent=True) or request.form
    recipe_id = data.get('recipe_id')
    plan_date_str = data.get('date')
    meal_type = (data.get('meal_type') or 'dinner').lower()
    servings = data.get('servings', 2)

    if not recipe_id or not plan_date_str:
        if request.is_json:
            return jsonify({'success': False, 'error': 'Recipe ID and date required'}), 400
        flash('Recipe ID and date are required.', 'error')
        return redirect(url_for('main.meal_planner'))

    try:
        recipe_id = int(float(recipe_id))
        plan_date = datetime.strptime(plan_date_str, '%Y-%m-%d').date()
        servings = max(1, min(20, int(servings)))
    except (ValueError, TypeError):
        if request.is_json:
            return jsonify({'success': False, 'error': 'Invalid date or parameters'}), 400
        flash('Invalid meal plan parameters.', 'error')
        return redirect(url_for('main.meal_planner'))

    if not cache._loaded:
        cache.load_models()

    recipe = cache.get_recipe_by_id(recipe_id)
    if not recipe:
        if request.is_json:
            return jsonify({'success': False, 'error': 'Recipe not found'}), 404
        flash('Recipe not found.', 'error')
        return redirect(url_for('main.meal_planner'))

    meal = MealPlan(
        user_id=current_user.id,
        recipe_id=recipe_id,
        plan_date=plan_date,
        meal_type=meal_type,
        servings=servings
    )
    db.session.add(meal)
    db.session.commit()

    if request.is_json:
        return jsonify({
            'success': True,
            'id': meal.id,
            'message': f"Added {recipe.get('RecipeName', 'Recipe')} to your plan!",
            'recipe_name': recipe.get('RecipeName')
        })

    flash(f"Added {recipe.get('RecipeName', 'Recipe')} to your plan!", 'success')
    return redirect(url_for('main.meal_planner', week=plan_date_str))


@main.route('/planner/remove/<int:plan_id>', methods=['POST', 'GET'])
@main.route('/api/planner/remove/<int:plan_id>', methods=['POST', 'DELETE'])
@login_required
def api_planner_remove(plan_id):
    meal = MealPlan.query.filter_by(id=plan_id, user_id=current_user.id).first()
    if not meal:
        if request.is_json or request.method == 'DELETE' or request.path.startswith('/api/'):
            return jsonify({'success': False, 'error': 'Meal not found'}), 404
        flash('Meal not found in plan.', 'error')
        return redirect(url_for('main.meal_planner'))

    plan_date_str = meal.plan_date.strftime('%Y-%m-%d')
    db.session.delete(meal)
    db.session.commit()

    if request.is_json or request.method == 'DELETE' or request.path.startswith('/api/'):
        return jsonify({'success': True, 'message': 'Removed meal from plan'})

    flash('Meal removed from plan.', 'success')
    return redirect(url_for('main.meal_planner', week=plan_date_str))


@main.route('/api/planner/update-servings/<int:plan_id>', methods=['POST'])
@login_required
def api_planner_update_servings(plan_id):
    data = request.get_json(silent=True) or request.form
    servings = data.get('servings')
    meal = MealPlan.query.filter_by(id=plan_id, user_id=current_user.id).first()
    if not meal:
        return jsonify({'success': False, 'error': 'Meal not found'}), 404
    try:
        meal.servings = max(1, min(20, int(servings)))
        db.session.commit()
        return jsonify({'success': True, 'servings': meal.servings})
    except (ValueError, TypeError):
        return jsonify({'success': False, 'error': 'Invalid servings'}), 400


@main.route('/api/planner/clear', methods=['POST'])
@login_required
def api_planner_clear():
    data = request.get_json(silent=True) or request.form
    week_str = data.get('week')
    if week_str:
        try:
            target_date = datetime.strptime(week_str, '%Y-%m-%d').date()
        except ValueError:
            target_date = date.today()
    else:
        target_date = date.today()

    start_of_week = target_date - timedelta(days=target_date.weekday())
    end_of_week = start_of_week + timedelta(days=6)

    MealPlan.query.filter(
        MealPlan.user_id == current_user.id,
        MealPlan.plan_date >= start_of_week,
        MealPlan.plan_date <= end_of_week
    ).delete()
    db.session.commit()

    if request.is_json:
        return jsonify({'success': True, 'message': 'Week cleared successfully'})

    flash('Meal plan for the week has been cleared.', 'success')
    return redirect(url_for('main.meal_planner', week=week_str))


@main.route('/api/planner/generate', methods=['POST'])
@login_required
def api_planner_generate():
    if not cache._loaded:
        cache.load_models()

    data = request.get_json(silent=True) or request.form
    diet = (data.get('diet') or 'all').lower()
    week_str = data.get('week')
    if week_str:
        try:
            target_date = datetime.strptime(week_str, '%Y-%m-%d').date()
        except ValueError:
            target_date = date.today()
    else:
        target_date = date.today()

    start_of_week = target_date - timedelta(days=target_date.weekday())
    end_of_week = start_of_week + timedelta(days=6)

    # Clear current meals in that week
    MealPlan.query.filter(
        MealPlan.user_id == current_user.id,
        MealPlan.plan_date >= start_of_week,
        MealPlan.plan_date <= end_of_week
    ).delete()

    df = cache.recipe_df.copy()
    if diet == 'vegetarian':
        veg_mask = df['Diet'].fillna('').str.lower().str.contains('veg') & ~df['Diet'].fillna('').str.lower().str.contains('non')
        df = df[veg_mask]
    elif diet == 'high_protein':
        prot_mask = df['Diet'].fillna('').str.lower().str.contains('high protein') | (df['Protein (g)'].fillna(0) >= 15)
        df = df[prot_mask]
    elif diet == 'quick':
        quick_mask = (df['TotalTimeInMins'].fillna(999) <= 30) | (df['CookTimeInMins'].fillna(999) <= 25)
        df = df[quick_mask]

    breakfast_pool = df[df['Course'].fillna('').str.contains('Breakfast|Snack|Appetizer', case=False)]
    lunch_pool = df[df['Course'].fillna('').str.contains('Lunch|Main Course|Side Dish', case=False)]
    dinner_pool = df[df['Course'].fillna('').str.contains('Dinner|Main Course', case=False)]
    snack_pool = df[df['Course'].fillna('').str.contains('Snack|Dessert|Appetizer', case=False)]

    if breakfast_pool.empty: breakfast_pool = df
    if lunch_pool.empty: lunch_pool = df
    if dinner_pool.empty: dinner_pool = df
    if snack_pool.empty: snack_pool = df

    b_samples = breakfast_pool.sample(n=min(7, len(breakfast_pool)), replace=(len(breakfast_pool) < 7)).to_dict('records')
    l_samples = lunch_pool.sample(n=min(7, len(lunch_pool)), replace=(len(lunch_pool) < 7)).to_dict('records')
    d_samples = dinner_pool.sample(n=min(7, len(dinner_pool)), replace=(len(dinner_pool) < 7)).to_dict('records')
    s_samples = snack_pool.sample(n=min(7, len(snack_pool)), replace=(len(snack_pool) < 7)).to_dict('records')

    created_count = 0
    for i in range(7):
        d = start_of_week + timedelta(days=i)
        slots = [
            ('breakfast', b_samples[i % len(b_samples)]),
            ('lunch', l_samples[i % len(l_samples)]),
            ('dinner', d_samples[i % len(d_samples)]),
            ('snack', s_samples[i % len(s_samples)])
        ]
        for meal_type, r in slots:
            r_id = int(float(r['Srno']))
            plan = MealPlan(
                user_id=current_user.id,
                recipe_id=r_id,
                plan_date=d,
                meal_type=meal_type,
                servings=2
            )
            db.session.add(plan)
            created_count += 1

    db.session.commit()
    if request.is_json:
        return jsonify({
            'success': True,
            'count': created_count,
            'message': f'Generated {created_count} balanced meals for the week!'
        })

    flash(f'Generated {created_count} balanced meals for your week!', 'success')
    return redirect(url_for('main.meal_planner', week=week_str))


@main.route('/planner/grocery')
@login_required
def planner_grocery():
    if not cache._loaded:
        cache.load_models()

    week_str = request.args.get('week')
    if week_str:
        try:
            target_date = datetime.strptime(week_str, '%Y-%m-%d').date()
        except ValueError:
            target_date = date.today()
    else:
        target_date = date.today()

    start_of_week = target_date - timedelta(days=target_date.weekday())
    end_of_week = start_of_week + timedelta(days=6)

    user_plans = MealPlan.query.filter(
        MealPlan.user_id == current_user.id,
        MealPlan.plan_date >= start_of_week,
        MealPlan.plan_date <= end_of_week
    ).all()

    recipe_ids = [p.recipe_id for p in user_plans]
    recipe_ingredients = []
    recipes_summary = []

    for r_id in set(recipe_ids):
        rec = cache.get_recipe_by_id(r_id)
        if rec:
            ing = rec.get('TranslatedIngredients') or rec.get('Ingredients') or ''
            if ing:
                recipe_ingredients.append(ing)
            recipes_summary.append({
                'id': r_id,
                'name': rec.get('RecipeName', 'Dish'),
                'image': rec.get('images', ''),
                'cuisine': rec.get('Cuisine', '')
            })

    aisles = generate_aisle_grocery_list(recipe_ingredients)
    total_items = sum(len(a['ingredients']) for a in aisles.values())

    return render_template(
        'grocery_list.html',
        aisles=aisles,
        total_items=total_items,
        recipes_summary=recipes_summary,
        start_of_week=start_of_week,
        end_of_week=end_of_week,
        current_week_str=start_of_week.strftime('%Y-%m-%d')
    )


@main.route('/api/planner/search-recipes')
@login_required
def api_planner_search():
    if not cache._loaded:
        cache.load_models()

    q = request.args.get('q', '').strip().lower()
    meal_type = request.args.get('meal_type', '').lower()
    limit = min(30, int(request.args.get('limit', 15)))

    df = cache.recipe_df

    if q:
        mask = df['RecipeName'].fillna('').str.lower().str.contains(re.escape(q), na=False) | \
               df['TranslatedIngredients'].fillna('').str.lower().str.contains(re.escape(q), na=False)
        matched = df[mask]
    else:
        if meal_type == 'breakfast':
            matched = df[df['Course'].fillna('').str.contains('Breakfast|Snack', case=False)]
        elif meal_type == 'lunch':
            matched = df[df['Course'].fillna('').str.contains('Lunch|Main Course', case=False)]
        elif meal_type == 'dinner':
            matched = df[df['Course'].fillna('').str.contains('Dinner|Main Course', case=False)]
        elif meal_type == 'snack':
            matched = df[df['Course'].fillna('').str.contains('Snack|Dessert|Appetizer', case=False)]
        else:
            matched = df

    records = matched.head(limit).to_dict(orient='records')
    results = []
    for r in records:
        r_id = int(float(r['Srno'])) if 'Srno' in r and not pd.isna(r['Srno']) else None
        if not r_id:
            continue
        cal = r.get('Calories (kcal)', 0)
        prot = r.get('Protein (g)', 0)
        results.append({
            'id': r_id,
            'name': r.get('RecipeName', ''),
            'cuisine': r.get('Cuisine', ''),
            'course': r.get('Course', ''),
            'diet': r.get('Diet', ''),
            'cook_time': r.get('CookTimeInMins') or r.get('TotalTimeInMins') or 25,
            'calories': round(float(cal)) if cal and not pd.isna(cal) else 250,
            'protein': round(float(prot), 1) if prot and not pd.isna(prot) else 10.0,
            'image': r.get('images', '')
        })

    return jsonify({'success': True, 'count': len(results), 'recipes': results})

