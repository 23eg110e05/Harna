import os
import sys
import re
import ast
import pickle
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception:
    pass

stopwords_set = set(nltk_stopwords.words('english'))
stemmer = PorterStemmer()
paren_pattern = re.compile(r'\(.*\)')
non_alpha_pattern = re.compile(r'[^a-z\s]+')

measurements = {
    'tbsp', 'tablespoon', 'tablespoons', 'tsp', 'teaspoon', 'teaspoons',
    'oz', 'ounce', 'ounces', 'c', 'cup', 'cups', 'qt', 'quart', 'quarts',
    'pt', 'pint', 'pints', 'pinch', 'gal', 'gallon', 'lb', 'pound', 'pounds',
    'ml', 'millilitre', 'milliliter', 'g', 'gram', 'grams', 'kg', 'kilogram', 'l', 'liter', 'inch'
}

unwanted_words = {
    'chop', 'chopped', 'slice', 'sliced', 'cut', 'deseed', 'cooked', 'boil', 'boiled',
    'soak', 'fry', 'fried', 'grate', 'grated', 'puree', 'pureed', 'mince', 'minced',
    'dice', 'diced', 'shred', 'shredded', 'piece', 'pieces', 'large', 'medium', 'small',
    'drain', 'rinsed', 'fresh', 'crushed', 'garnish', 'and', 'the', 'per', 'taste', 'clean', 'washed'
}

def clean_ingredient_text(ing_text):
    if not isinstance(ing_text, str):
        return ""
    if ing_text.startswith('[') and ing_text.endswith(']'):
        try:
            items = ast.literal_eval(ing_text)
        except Exception:
            items = ing_text.split(',')
    else:
        items = ing_text.split(',')

    cleaned_tokens = []
    for item in items:
        item = item.lower()
        item = paren_pattern.sub('', item)
        item = non_alpha_pattern.sub(' ', item)
        words = item.split()
        words = [stemmer.stem(w) for w in words if w not in stopwords_set and w not in measurements and w not in unwanted_words and len(w) > 2]
        if words:
            cleaned_tokens.append(" ".join(words))

    return " ".join(cleaned_tokens)

def clean_recipe_name(name):
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = re.sub(r'[^a-z\s]', '', name)
    return name.strip()

def main():
    print("[Harna Trainer] Starting dataset aggregation...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed")

    indian_file = os.path.join(raw_dir, "Final_Indian_Dataset.csv")
    df_indian = pd.read_csv(indian_file)
    print(f"[Harna Trainer] Loaded Indian dataset ({len(df_indian)} recipes).")

    new_file = os.path.join(raw_dir, "recipes_data.csv")
    if os.path.exists(new_file):
        print("[Harna Trainer] Loading curated global recipes from recipes_data.csv...")
        df_new = pd.read_csv(new_file, nrows=20000)
        max_srno = int(df_indian['Srno'].max() or 0) + 1

        new_rows = []
        for idx, row in df_new.iterrows():
            title = str(row.get('title') or '').strip()
            if not title or title.lower() == 'nan':
                continue
            
            ner_raw = row.get('NER')
            if isinstance(ner_raw, str) and ner_raw.startswith('['):
                try:
                    ing_str = ", ".join(ast.literal_eval(ner_raw))
                except Exception:
                    ing_str = str(row.get('ingredients') or '')
            else:
                ing_str = str(row.get('ingredients') or '')

            dirs = str(row.get('directions') or '')
            new_rows.append({
                'Srno': max_srno + idx,
                'RecipeName': title,
                'TranslatedRecipeName': title,
                'Ingredients': ing_str,
                'TranslatedIngredients': ing_str,
                'PrepTimeInMins': 15.0,
                'CookTimeInMins': 25.0,
                'TotalTimeInMins': 40.0,
                'Servings': 4.0,
                'Cuisine': 'Global / Fusion',
                'Course': 'Main Course',
                'Diet': 'General',
                'Instructions': dirs,
                'TranslatedInstructions': dirs,
                'URL': str(row.get('link') or ''),
                'images': 'https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=400',
                'Calories (kcal)': 320.0,
                'Protein (g)': 14.0,
                'Carbohydrates (g)': 35.0,
                'Fats (g)': 12.0,
                'Fiber (g)': 2.0,
                'Calories (kcal).1': 80.0,
                'Protein (g).1': 3.5,
                'Carbohydrates (g).1': 8.8,
                'Fats (g).1': 3.0,
                'Fiber (g).1': 0.5,
                'Difficulty': 'Medium'
            })

        df_mapped = pd.DataFrame(new_rows)
        print(f"[Harna Trainer] Successfully mapped {len(df_mapped)} global recipes.")
        df_combined = pd.concat([df_indian, df_mapped], ignore_index=True)
    else:
        df_combined = df_indian.copy()

    print(f"[Harna Trainer] Total combined training records: {len(df_combined)}")

    df_combined['Ingredients_clean'] = df_combined['Ingredients'].fillna('').astype(str).apply(clean_ingredient_text)
    df_combined['RecipeName_clean'] = df_combined['RecipeName'].fillna('').astype(str).apply(clean_recipe_name)

    print("[Harna Trainer] Training TF-IDF vectorizers...")
    vec_ing = TfidfVectorizer(max_features=12000, stop_words='english')
    vec_name = TfidfVectorizer(max_features=8000, stop_words='english')

    tfidf_ing = vec_ing.fit_transform(df_combined['Ingredients_clean'])
    tfidf_name = vec_name.fit_transform(df_combined['RecipeName_clean'])
    tfidf_combined = hstack([0.65 * tfidf_name, 0.60 * tfidf_ing]).tocsr()

    os.makedirs(processed_dir, exist_ok=True)
    with open(os.path.join(processed_dir, 'TFidf_matrix.pkl'), 'wb') as f:
        pickle.dump(tfidf_combined, f)
    with open(os.path.join(processed_dir, 'Vectorizer_names.pkl'), 'wb') as f:
        pickle.dump(vec_name, f)
    with open(os.path.join(processed_dir, 'Vectorizer_ingredients.pkl'), 'wb') as f:
        pickle.dump(vec_ing, f)
    with open(os.path.join(processed_dir, 'Recipes.pkl'), 'wb') as f:
        pickle.dump(df_combined, f)

    print(f"[Harna Trainer] Models successfully trained and serialized to {processed_dir}!")

if __name__ == '__main__':
    main()
