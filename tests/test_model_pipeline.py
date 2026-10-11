import os
import pickle
import importlib.util
import pytest
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def load_cache():
    cache_path = os.path.join(BASE_DIR, "my_flask_app", "utils", "cache.py")
    spec = importlib.util.spec_from_file_location("cache_module", cache_path)
    cache_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cache_mod)
    cache_instance = cache_mod.ModelCache.get_instance()
    cache_instance.load_models(os.path.join(BASE_DIR, "data"))
    return cache_instance


def test_processed_artifacts_exist_and_non_empty():
    expected_files = [
        "TFidf_matrix.pkl",
        "Vectorizer_names.pkl",
        "Vectorizer_ingredients.pkl",
        "Recipes.pkl"
    ]
    for filename in expected_files:
        filepath = os.path.join(PROCESSED_DIR, filename)
        assert os.path.exists(filepath), f"Missing artifact: {filename}"
        assert os.path.getsize(filepath) > 0, f"Artifact is empty: {filename}"


def test_recipes_dataframe_structure_and_counts():
    cache = load_cache()
    df = cache.recipe_df
    assert len(df) == 25655, f"Expected 25,655 recipes, got {len(df)}"

    # Indian and global recipe partitions
    global_mask = df["Cuisine"] == "Global / Fusion"
    indian_count = (~global_mask).sum()
    global_count = global_mask.sum()
    assert indian_count == 5655, f"Expected 5,655 Indian recipes, got {indian_count}"
    assert global_count == 20000, f"Expected 20,000 Global recipes, got {global_count}"

    # IDs must be unique and non-colliding
    assert df["Srno"].nunique() == 25655, "Duplicate recipe IDs found in Srno"


def test_global_recipes_nutrition_defaults_not_nan():
    cache = load_cache()
    df = cache.recipe_df
    global_df = df[df["Cuisine"] == "Global / Fusion"]

    nutrition_cols = [
        "Calories (kcal)", "Protein (g)", "Carbohydrates (g)", "Fats (g)", "Fiber (g)",
        "Calories (kcal).1", "Protein (g).1", "Carbohydrates (g).1", "Fats (g).1", "Fiber (g).1"
    ]
    for col in nutrition_cols:
        assert col in global_df.columns, f"Missing column {col} in Recipes.pkl"
        null_count = global_df[col].isnull().sum()
        assert null_count == 0, f"Found {null_count} nulls in {col} for global recipes"

    # Spot-check specific default values
    sample = global_df.iloc[0]
    assert sample["Calories (kcal)"] == 320.0
    assert sample["Calories (kcal).1"] == 80.0
    assert sample["Protein (g).1"] == 3.5
    assert sample["Carbohydrates (g).1"] == 8.8
    assert sample["Fats (g).1"] == 3.0
    assert sample["Fiber (g)"] == 2.0
    assert sample["Fiber (g).1"] == 0.5


def test_tfidf_matrix_and_vectorizers_contract():
    cache = load_cache()
    m = cache.tfidf_combined
    vn = cache.vectorizer_name
    vi = cache.vectorizer_ing

    assert isinstance(m, csr_matrix), "TF-IDF matrix is not a csr_matrix"
    assert m.shape[0] == 25655, f"Expected 25,655 rows in matrix, got {m.shape[0]}"
    assert len(vn.vocabulary_) == 8000, f"Expected 8000 name features, got {len(vn.vocabulary_)}"
    assert len(vi.vocabulary_) > 1000, f"Ingredient vocabulary too small: {len(vi.vocabulary_)}"
    assert m.shape[1] == len(vn.vocabulary_) + len(vi.vocabulary_)


def test_model_cache_get_recipe_by_id_retrieval():
    cache = load_cache()
    indian_recipe = cache.get_recipe_by_id(1)
    assert indian_recipe is not None
    assert indian_recipe["RecipeName"] == "Masala Karela Recipe"
    assert "Indian" in indian_recipe["Cuisine"]

    # Global recipe at beginning of mapped range (14168)
    global_recipe = cache.get_recipe_by_id(14168)
    assert global_recipe is not None
    assert global_recipe["Cuisine"] == "Global / Fusion"
    assert global_recipe["Calories (kcal).1"] == 80.0


@pytest.mark.parametrize("query,expected_cuisine_type,expected_keyword", [
    ("paneer, butter", "North Indian Recipes", "paneer"),
    ("beef, mushroom", "Global / Fusion", "beef"),
    ("chicken, egg", "Chinese", "chicken"),
    ("pasta, cheese", "Global / Fusion", "pasta"),
])
def test_recommendation_queries_across_cuisines(query, expected_cuisine_type, expected_keyword):
    cache = load_cache()
    m = cache.tfidf_combined
    vn = cache.vectorizer_name
    vi = cache.vectorizer_ing
    df = cache.recipe_df

    uv = hstack([0.5 * vn.transform([query]), 0.5 * vi.transform([query])])
    sim = cosine_similarity(uv, m).flatten()
    top_idx = sim.argmax()
    top_recipe = df.iloc[top_idx]
    top_score = sim[top_idx]

    assert top_score > 0.40, f"Query '{query}' top match score too low: {top_score}"
    assert expected_keyword in top_recipe["RecipeName"].lower(), (
        f"Query '{query}' top match '{top_recipe['RecipeName']}' does not contain '{expected_keyword}'"
    )
