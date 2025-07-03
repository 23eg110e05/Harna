import pandas as pd
from my_flask_app.utils.preprocess import preprocess_ingredients
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import pickle
import re

def preprocess_recipe_name(name):
    name = name.lower()
    name = re.sub(r'[^a-z\s]', '', name)
    return name

df = pd.read_csv("data/raw/Final_Indian_Dataset.csv")
df_original = df.copy(deep=True)

df['Ingredients'] = df['Ingredients'].apply(preprocess_ingredients)
df['RecipeName_clean'] = df['RecipeName'].apply(preprocess_recipe_name)

vectorizer_ing = TfidfVectorizer()
vectorizer_name = TfidfVectorizer()

tfidf_ing = vectorizer_ing.fit_transform(df['Ingredients'])
tfidf_name = vectorizer_name.fit_transform(df['RecipeName_clean'])

tfidf_combined = hstack([0.65 * tfidf_name, 0.6 * tfidf_ing])

with open('data/processed/TFidf_matrix.pkl', 'wb') as file:
    pickle.dump(tfidf_combined, file)

with open('data/processed/Vectorizer_names.pkl', 'wb') as file:
    pickle.dump(vectorizer_name, file)

with open('data/processed/Vectorizer_ingredients.pkl', 'wb') as file:
    pickle.dump(vectorizer_ing, file)

with open('data/processed/Recipes.pkl', 'wb') as file:
    pickle.dump(df_original, file)

print("Combined TF-IDF matrix, vectorizers, and recipes data saved.")
