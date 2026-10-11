import os
import re
import ast
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from my_flask_app.utils.preprocess import preprocess_ingredients

def preprocess_recipe_name(name):
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = re.sub(r'[^a-z\s]', '', name)
    return name.strip()

def train_and_cache_models():
    print("[Harna Model Trainer] Loading datasets...")
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data")
    raw_dir = os.path.join(data_dir, "raw")
    processed_dir = os.path.join(data_dir, "processed")

    # 1. Load Primary Indian Dataset
    indian_path = os.path.join(raw_dir, "Final_Indian_Dataset.csv")
    df_indian = pd.read_csv(indian_path)
    print(f"[Harna Model Trainer] Loaded primary Indian dataset with {len(df_indian)} recipes.")

    # 2. Check and incorporate new recipes dataset if available
    new_data_path = os.path.join(raw_dir, "recipes_data.csv")
    if os.path.exists(new_data_path):
        print("[Harna Model Trainer] Found new dataset (recipes_data.csv). Ingesting curated slice...")
        try:
            # Ingest top 20,000 diverse recipes from the large dataset to enrich variety without memory bloat
            df_new = pd.read_csv(new_data_path, nrows=20000)
            
            # Map columns to standard Harna schema
            max_srno = int(df_indian['Srno'].max() or 0) + 1
            new_records = []
            
            for idx, row in df_new.iterrows():
                title = str(row.get('title') or '').strip()
                if not title or title.lower() == 'nan':
                    continue
                
                raw_ing = row.get('ingredients') or '[]'
                raw_dirs = row.get('directions') or '[]'
                
                # Clean NER / ingredients
                ner_val = row.get('NER')
                if isinstance(ner_val, str) and ner_val.startswith('['):
                    try:
                        clean_ing_list = ast.literal_eval(ner_val)
                        ing_str = ", ".join(clean_ing_list)
                    except Exception:
                        ing_str = str(raw_ing)
                else:
                    ing_str = str(raw_ing)

                new_records.append({
                    'Srno': max_srno + idx,
                    'RecipeName': title,
                    'TranslatedRecipeName': title,
                    'Ingredients': ing_str,
                    'TranslatedIngredients': ing_str,
                    'PrepTimeInMins': 15.0,
                    'CookTimeInMins': 25.0,
                    'TotalTimeInMins': 40.0,
                    'Servings': 4.0,
                    'Cuisine': 'Continental / Global',
                    'Course': 'Main Course',
                    'Diet': 'General',
                    'Instructions': raw_dirs,
                    'TranslatedInstructions': raw_dirs,
                    'URL': str(row.get('link') or ''),
                    'images': 'https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=400',
                    'Calories (kcal)': 320.0,
                    'Protein (g)': 14.0,
                    'Carbohydrates (g)': 35.0,
                    'Fats (g)': 12.0,
                    'Difficulty': 'Medium'
                })
            
            df_new_mapped = pd.DataFrame(new_records)
            print(f"[Harna Model Trainer] Successfully mapped {len(df_new_mapped)} new global recipes.")
            
            # Combine both datasets
            df_combined = pd.concat([df_indian, df_new_mapped], ignore_index=True)
        except Exception as e:
            print(f"[Harna Model Trainer Warning] Error ingesting new dataset: {e}. Fallback to primary dataset.")
            df_combined = df_indian.copy()
    else:
        df_combined = df_indian.copy()

    print(f"[Harna Model Trainer] Training on combined dataset totaling {len(df_combined)} recipes...")

    # Data Preprocessing
    df_combined['Ingredients_clean'] = df_combined['Ingredients'].fillna('').astype(str).apply(preprocess_ingredients)
    df_combined['RecipeName_clean'] = df_combined['RecipeName'].fillna('').astype(str).apply(preprocess_recipe_name)

    # Vectorizers
    vectorizer_ing = TfidfVectorizer(max_features=12000, stop_words='english')
    vectorizer_name = TfidfVectorizer(max_features=8000, stop_words='english')

    print("[Harna Model Trainer] Fitting TF-IDF matrices...")
    tfidf_ing = vectorizer_ing.fit_transform(df_combined['Ingredients_clean'])
    tfidf_name = vectorizer_name.fit_transform(df_combined['RecipeName_clean'])

    # Weighted combined sparse feature representation
    tfidf_combined = hstack([0.65 * tfidf_name, 0.60 * tfidf_ing]).tocsr()

    # Save to disk
    os.makedirs(processed_dir, exist_ok=True)
    with open(os.path.join(processed_dir, 'TFidf_matrix.pkl'), 'wb') as f:
        pickle.dump(tfidf_combined, f)

    with open(os.path.join(processed_dir, 'Vectorizer_names.pkl'), 'wb') as f:
        pickle.dump(vectorizer_name, f)

    with open(os.path.join(processed_dir, 'Vectorizer_ingredients.pkl'), 'wb') as f:
        pickle.dump(vectorizer_ing, f)

    with open(os.path.join(processed_dir, 'Recipes.pkl'), 'wb') as f:
        pickle.dump(df_combined, f)

    print(f"[Harna Model Trainer] Complete! Models saved to {processed_dir}. Total indexed recipes: {len(df_combined)}")

if __name__ == '__main__':
    train_and_cache_models()
