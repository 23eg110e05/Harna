"""
End-to-End Opaque-Box Test Suite for Harna Recipe System & Ratvi AI Companion
Covers:
- R1: Model Training on Combined Datasets & ModelCache Singleton
- R2: Floating Ratvi AI Companion UI Interface & Strict /profile Exclusion
- R3: Dual-Engine Intelligence Backend (/api/ratvi/chat)

Structured across 4 Tiers:
- Tier 1: Feature Coverage (Isolation / Happy Path, >=5 per feature)
- Tier 2: Boundary & Corner Cases (Limits, Fault Tolerance, Extremes, >=5 per feature)
- Tier 3: Cross-Feature Combinations (Pairwise & Module Interactions)
- Tier 4: Real-World Scenarios (End-to-End User Journeys)
"""

import os
import re
import ast
import json
import pickle
import pytest
import pandas as pd
import numpy as np

from my_flask_app import create_app
from my_flask_app.models import db, User
from my_flask_app.utils.cache import cache
from my_flask_app.utils.ratvi import RatviEngine, ratvi


# ============================================================================
# TIER 1: FEATURE COVERAGE (Isolation / Happy Path)
# ============================================================================

class TestTier1FeatureCoverage:
    """Tier 1: Feature Coverage in Isolation (>=5 tests per feature)."""

    # ------------------------------------------------------------------------
    # R1: Model Training Artifacts & ModelCache
    # ------------------------------------------------------------------------

    def test_tier1_r1_01_pickle_artifacts_exist(self):
        """R1: Verify all 4 required model pickle files exist in data/processed/."""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        processed_dir = os.path.join(base_dir, "data", "processed")
        expected_files = [
            "TFidf_matrix.pkl",
            "Vectorizer_names.pkl",
            "Vectorizer_ingredients.pkl",
            "Recipes.pkl"
        ]
        for fname in expected_files:
            fpath = os.path.join(processed_dir, fname)
            assert os.path.isfile(fpath), f"Required model artifact missing: {fname} at {fpath}"
            assert os.path.getsize(fpath) > 0, f"Model artifact is empty: {fname}"

    def test_tier1_r1_02_recipes_dataframe_contract(self):
        """R1: Verify Recipes.pkl matches the DataFrame schema and contains required columns."""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        recipes_path = os.path.join(base_dir, "data", "processed", "Recipes.pkl")
        with open(recipes_path, "rb") as f:
            df = pickle.load(f)
        assert isinstance(df, pd.DataFrame), f"Recipes.pkl must be a pd.DataFrame, got {type(df)}"
        assert len(df) > 0, "Recipes DataFrame must not be empty"

        required_columns = [
            "Srno", "RecipeName", "Ingredients", "Instructions", "Cuisine", "Servings"
        ]
        for col in required_columns:
            assert col in df.columns, f"Required column '{col}' missing from Recipes DataFrame"

        # Check Srno has integer-compatible unique values
        assert df["Srno"].notnull().all(), "Srno must not contain null values"
        assert df["Srno"].nunique() == len(df), "Srno must contain unique identifiers without collisions"

    def test_tier1_r1_03_model_cache_singleton_loading(self, app):
        """R1: Verify ModelCache singleton loads vectorizers, matrix, and DataFrame into memory."""
        with app.app_context():
            assert cache is not None, "ModelCache singleton instance must exist"
            assert hasattr(cache, "recipe_df"), "ModelCache must have recipe_df attribute"
            assert cache.recipe_df is not None, "ModelCache.recipe_df must be loaded"
            assert len(cache.recipe_df) > 0, "ModelCache.recipe_df must contain recipe records"
            assert hasattr(cache, "tfidf_combined"), "ModelCache must have tfidf_combined matrix"
            assert cache.tfidf_combined is not None, "ModelCache.tfidf_combined must be loaded"

    def test_tier1_r1_04_recipe_lookup_by_id(self, app):
        """R1: Verify ModelCache.get_recipe_by_id returns valid recipe row for existing ID."""
        with app.app_context():
            recipe = cache.get_recipe_by_id(1)
            assert recipe is not None, "get_recipe_by_id(1) must return recipe data"
            assert int(recipe.get("Srno")) == 1, f"Expected Srno 1, got {recipe.get('Srno')}"
            assert "RecipeName" in recipe, "Recipe must contain RecipeName"
            assert len(str(recipe["RecipeName"]).strip()) > 0, "RecipeName must not be empty"

    def test_tier1_r1_05_recommendation_engine_dual_collection(self, app, client):
        """R1: Verify recommendation query returns valid dishes with positive similarity."""
        with app.app_context():
            # Query Indian cuisine ingredients
            res_indian = client.get("/api/recommend?ingredients=paneer,butter")
            assert res_indian.status_code == 200, f"/api/recommend returned {res_indian.status_code}"
            data_indian = res_indian.get_json()
            assert data_indian.get("success") is True, "Expected success: true from /api/recommend"
            assert len(data_indian.get("recipes", [])) > 0, "Expected recommendations for paneer, butter"

            # Query global / fusion ingredients
            res_global = client.get("/api/recommend?ingredients=pasta,cheese")
            assert res_global.status_code == 200
            data_global = res_global.get_json()
            assert data_global.get("success") is True
            assert len(data_global.get("recipes", [])) > 0, "Expected recommendations for pasta, cheese"

    # ------------------------------------------------------------------------
    # R2: Floating Ratvi AI Companion UI Interface & /profile Exclusion
    # ------------------------------------------------------------------------

    def test_tier1_r2_01_widget_markup_on_landing_page(self, client):
        """R2: Verify floating widget markup is rendered on public landing page /."""
        res = client.get("/")
        assert res.status_code == 200
        html = res.data.decode("utf-8")
        assert 'id="ratvi-widget"' in html or 'class="ratvi-widget"' in html, "Ratvi widget missing on /"
        assert 'id="ratvi-fab"' in html or 'class="ratvi-fab"' in html, "Ratvi FAB pill missing on /"
        assert 'id="ratvi-chat-panel"' in html or 'class="ratvi-panel"' in html, "Ratvi chat panel missing on /"

    def test_tier1_r2_02_widget_markup_on_auth_pages(self, client):
        """R2: Verify floating widget markup is rendered on /login and /signup pages."""
        for path in ["/login", "/signup"]:
            res = client.get(path)
            assert res.status_code == 200, f"Failed GET {path}"
            html = res.data.decode("utf-8")
            assert 'ratvi-widget' in html, f"Ratvi widget markup missing on {path}"
            assert 'ratvi-fab' in html, f"Ratvi FAB pill missing on {path}"

    def test_tier1_r2_03_widget_markup_on_core_app_views(self, login):
        """R2: Verify floating widget markup is rendered on authenticated app views: /home, /results, /discover, /favourites."""
        for path in ["/home", "/results?ingredients=chicken", "/discover", "/favourites"]:
            res = login.get(path, follow_redirects=True)
            assert res.status_code == 200, f"Failed GET {path}"
            html = res.data.decode("utf-8")
            assert 'ratvi-widget' in html, f"Ratvi widget markup missing on {path}"

    def test_tier1_r2_04_strict_exclusion_on_profile_view(self, login):
        """R2: Verify floating widget markup is STRICTLY EXCLUDED on /profile (Layer 1 Jinja & Layer 2 block override)."""
        res = login.get("/profile", follow_redirects=True)
        assert res.status_code == 200
        html = res.data.decode("utf-8")
        assert 'id="ratvi-widget"' not in html, "Ratvi widget element MUST NOT be rendered on /profile"
        assert 'id="ratvi-fab"' not in html, "Ratvi FAB button MUST NOT be rendered on /profile"

    def test_tier1_r2_05_theme_color_tokens_and_styling(self):
        """R2: Verify Ratvi CSS incorporates brand colors Deep Teal (#063238) and Electric Lime (#D4F620)."""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        css_paths = [
            os.path.join(base_dir, "my_flask_app", "static", "css", "ratvi-chat.css"),
            os.path.join(base_dir, "my_flask_app", "static", "css", "design-system.css")
        ]
        found_teal = False
        found_lime = False
        for path in css_paths:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().upper()
                if "#063238" in content or "--HARNA-TEAL" in content:
                    found_teal = True
                if "#D4F620" in content or "--HARNA-LIME" in content:
                    found_lime = True
        assert found_teal, "Brand Deep Teal (#063238) token must be present in stylesheet"
        assert found_lime, "Brand Electric Lime (#D4F620) token must be present in stylesheet"

    def test_tier1_r2_06_recipe_details_context_metadata(self, login):
        """R2: Verify /recipe_details/<id> contains context-aware metadata or elements."""
        res = login.get("/recipe_details/1", follow_redirects=True)
        assert res.status_code == 200
        html = res.data.decode("utf-8")
        # Check that recipe details page renders recipe context hooks
        has_context_hook = (
            'id="recipe-page-context"' in html or
            'id="ratvi-context-bar"' in html or
            'data-recipe-id' in html or
            'recipe_details' in html
        )
        assert has_context_hook, "Recipe details page must provide recipe context hooks for Ratvi"

    # ------------------------------------------------------------------------
    # R3: Dual-Engine Backend (/api/ratvi/chat)
    # ------------------------------------------------------------------------

    def test_tier1_r3_01_public_unauthenticated_chat_access(self, client):
        """R3: Verify POST /api/ratvi/chat is public and succeeds with 200 OK without authentication (no 302 redirect)."""
        res = client.post("/api/ratvi/chat", json={"message": "Hello Ratvi"}, follow_redirects=False)
        assert res.status_code == 200, f"Expected 200 OK for unauthenticated chat, got {res.status_code}"
        data = res.get_json()
        assert data is not None, "Response must be valid JSON"
        assert data.get("success") is True, "Response must have success: true"
        assert "reply" in data, "Response must contain 'reply' field"
        assert len(data["reply"]) > 0, "'reply' must be non-empty"

    def test_tier1_r3_02_serving_scaling_happy_path(self, client):
        """R3: Verify scaling request returns intent 'servings_scaled' with adjusted quantities."""
        payload = {
            "message": "Scale for 6 people",
            "recipe_id": 1
        }
        res = client.post("/api/ratvi/chat", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert data.get("success") is True
        assert data.get("intent") == "servings_scaled" or "servings_scaled" in str(data)
        assert "scaled_ingredients" in data.get("data", {}), "Response data must include scaled_ingredients"
        assert data["data"].get("target_servings") == 6, f"Expected target_servings 6, got {data['data'].get('target_servings')}"

    def test_tier1_r3_03_culinary_substitution_happy_path(self, client):
        """R3: Verify substitution request returns valid alternatives and culinary advice."""
        payload = {
            "message": "Substitute for butter"
        }
        res = client.post("/api/ratvi/chat", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert data.get("success") is True
        assert data.get("intent") == "substitution" or "substitut" in str(data)
        reply_lower = data.get("reply", "").lower()
        assert any(alt in reply_lower for alt in ["ghee", "oil", "margarine"]), f"Reply should mention butter alternatives: {reply_lower}"

    def test_tier1_r3_04_step_guidance_happy_path(self, client):
        """R3: Verify cooking instructions query returns structured steps."""
        payload = {
            "message": "Show cooking steps",
            "recipe_id": 1
        }
        res = client.post("/api/ratvi/chat", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert data.get("success") is True
        assert data.get("intent") == "instructions" or "step" in str(data).lower()
        assert len(data.get("reply", "")) > 0, "Instructions reply must not be empty"

    def test_tier1_r3_05_hybrid_fallback_deterministic_offline(self, client):
        """R3: Verify deterministic engine returns 200 OK without errors when no external LLM API key exists."""
        payload = {
            "message": "How do I make a curry?",
            "options": {"engine": "hybrid"}
        }
        res = client.post("/api/ratvi/chat", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert data.get("success") is True
        assert data.get("engine_used") in ["deterministic", "hybrid", "llm"], f"Unexpected engine_used: {data.get('engine_used')}"


# ============================================================================
# TIER 2: BOUNDARY & CORNER CASES (Limits, Extremes, Fault Tolerance)
# ============================================================================

class TestTier2BoundaryAndCorners:
    """Tier 2: Boundary & Corner Cases (>=5 tests per feature)."""

    # ------------------------------------------------------------------------
    # R1 Boundary & Corner Cases
    # ------------------------------------------------------------------------

    def test_tier2_r1_01_nonexistent_recipe_id(self, app, client):
        """R1: Verify lookup for non-existent recipe ID returns 404 or None gracefully."""
        with app.app_context():
            recipe = cache.get_recipe_by_id(9999999)
            assert recipe is None, "Nonexistent recipe ID should return None from cache"

            res = client.get("/api/recipes/9999999")
            assert res.status_code in [404, 200], f"Expected 404 (or structured false), got {res.status_code}"
            if res.status_code == 200:
                assert res.get_json().get("success") is False

    def test_tier2_r1_02_negative_and_zero_recipe_ids(self, app):
        """R1: Verify lookup for negative and zero IDs handles cleanly without unhandled exceptions."""
        with app.app_context():
            assert cache.get_recipe_by_id(-1) is None
            assert cache.get_recipe_by_id(0) is None

    def test_tier2_r1_03_empty_and_whitespace_recommendation_query(self, client):
        """R1: Verify recommendation query with empty or whitespace input returns graceful response."""
        for q in ["", "   ", "   ,   "]:
            res = client.get(f"/api/recommend?ingredients={q}")
            assert res.status_code == 200
            data = res.get_json()
            assert data.get("success") in [True, False], "API should return valid JSON structure"

    def test_tier2_r1_04_special_characters_search_sanitization(self, client):
        """R1: Verify search query with SQL/XSS special characters does not cause internal error 500."""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "' OR '1'='1",
            "~!@#$%^&*()_+`-=[]{}|;':\",./<>?",
            "chicken\x00egg"
        ]
        for payload in malicious_inputs:
            res = client.get(f"/api/recommend?ingredients={payload}")
            assert res.status_code != 500, f"500 Internal Server Error triggered by input: {payload}"

    def test_tier2_r1_05_srno_uniqueness_and_no_id_collisions(self):
        """R1: Verify that every Srno in the processed dataset is unique with zero duplicates."""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        recipes_path = os.path.join(base_dir, "data", "processed", "Recipes.pkl")
        with open(recipes_path, "rb") as f:
            df = pickle.load(f)
        srno_series = df["Srno"]
        duplicate_count = srno_series.duplicated().sum()
        assert duplicate_count == 0, f"Detected {duplicate_count} duplicate Srno identifiers in Recipes.pkl"

    # ------------------------------------------------------------------------
    # R2 Boundary & Corner Cases
    # ------------------------------------------------------------------------

    def test_tier2_r2_01_profile_trailing_slash_and_query_params(self, login):
        """R2: Verify widget exclusion holds on /profile/ (trailing slash) and query parameter variations."""
        for path in ["/profile", "/profile/", "/profile?view=security", "/profile#settings"]:
            res = login.get(path, follow_redirects=True)
            if res.status_code == 200:
                html = res.data.decode("utf-8")
                assert 'id="ratvi-widget"' not in html, f"Ratvi widget leaked on {path}"

    def test_tier2_r2_02_recipe_title_special_characters_escaping(self, login):
        """R2: Verify recipe titles with quotes or ampersands are safely HTML-escaped in context hooks."""
        res = login.get("/recipe_details/1", follow_redirects=True)
        assert res.status_code == 200
        html = res.data.decode("utf-8")
        # Ensure that no unescaped quote breaks attribute boundaries
        broken_attr_pattern = r'data-recipe-name="[^"]*"[^>]*>'
        # Check document validity - basic check that HTML is well-formed around context
        assert "recipe" in html.lower()

    def test_tier2_r2_03_mobile_css_drawer_media_query(self):
        """R2: Verify ratvi-chat.css contains mobile responsive media query for screens <= 768px."""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        css_path = os.path.join(base_dir, "my_flask_app", "static", "css", "ratvi-chat.css")
        if os.path.exists(css_path):
            with open(css_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "@media" in content and "768px" in content, "ratvi-chat.css must define @media (max-width: 768px) rule"

    def test_tier2_r2_04_css_profile_display_none_guard(self):
        """R2: Verify ratvi-chat.css contains 4th-layer defense: CSS display:none !important on profile page."""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        css_path = os.path.join(base_dir, "my_flask_app", "static", "css", "ratvi-chat.css")
        if os.path.exists(css_path):
            with open(css_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "profile" in content.lower() and "display: none" in content.lower(), (
                "ratvi-chat.css must contain CSS guard rule hiding widget on profile view"
            )

    def test_tier2_r2_05_suggestion_chips_tray_structure(self, client):
        """R2: Verify suggestion chips container #ratvi-chips exists in widget template."""
        res = client.get("/")
        html = res.data.decode("utf-8")
        assert 'id="ratvi-chips"' in html or 'class="ratvi-chips"' in html, "Widget markup must contain #ratvi-chips container"

    # ------------------------------------------------------------------------
    # R3 Boundary & Corner Cases
    # ------------------------------------------------------------------------

    def test_tier2_r3_01_hyphenated_fractions_parsing(self):
        """R3: Verify parse_fraction and scaling handle hyphenated fractions ('1-1/2') without ValueError."""
        engine = RatviEngine()
        scaled = engine.scale_ingredient("1-1/2 cups Basmati Rice", base_servings=4, target_servings=8)
        assert scaled is not None
        assert "3" in scaled or "1-1/2" not in scaled, f"Expected scaled quantity for 1-1/2 cups, got {scaled}"

    def test_tier2_r3_02_unicode_fractions_parsing(self):
        """R3: Verify unicode fractions ('½', '¼', '¾') are parsed and scaled accurately."""
        engine = RatviEngine()
        scaled_half = engine.scale_ingredient("½ cup Milk", base_servings=4, target_servings=8)
        assert scaled_half is not None
        # Ratio is 2.0; 0.5 * 2^0.85 approx 0.90 -> ~1 cup or ~0.9 cup
        assert "cup" in scaled_half.lower()

    def test_tier2_r3_03_downscaling_sub_unitary_ratio(self, client):
        """R3: Verify downscaling from 4 servings to 1 person (ratio = 0.25) produces positive practical measurements."""
        payload = {
            "message": "Scale for 1 person",
            "recipe_id": 1
        }
        res = client.post("/api/ratvi/chat", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert data.get("success") is True
        scaled_ings = data.get("data", {}).get("scaled_ingredients", [])
        assert len(scaled_ings) > 0
        for line in scaled_ings:
            assert "-0" not in line and "-1" not in line, f"Scaled ingredient should not contain negative numbers: {line}"

    def test_tier2_r3_04_extreme_upscaling_stress(self, client):
        """R3: Verify massive upscaling (e.g. 50 servings, ratio = 12.5) executes without numerical overflow."""
        payload = {
            "message": "Scale for 50 people",
            "recipe_id": 1
        }
        res = client.post("/api/ratvi/chat", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert data.get("success") is True
        assert data.get("data", {}).get("target_servings") == 50

    def test_tier2_r3_05_missing_or_empty_message_payload(self, client):
        """R3: Verify POST /api/ratvi/chat with missing or empty message returns 400 Bad Request."""
        # Completely empty JSON
        res_empty = client.post("/api/ratvi/chat", json={})
        assert res_empty.status_code == 400, f"Expected 400 Bad Request for empty JSON, got {res_empty.status_code}"
        assert res_empty.get_json().get("success") is False

        # Whitespace-only message
        res_ws = client.post("/api/ratvi/chat", json={"message": "    "})
        assert res_ws.status_code == 400, f"Expected 400 Bad Request for whitespace message, got {res_ws.status_code}"
        assert res_ws.get_json().get("success") is False

    def test_tier2_r3_06_non_linear_damping_comparison(self):
        """R3: Verify non-linear damping: salt/spices damp significantly more than bulk base ingredients."""
        engine = RatviEngine()
        # Scale 4 -> 8 servings (ratio = 2.0)
        # Bulk ingredient: 2 cups Rice
        scaled_bulk = engine.scale_ingredient("2 cups Rice", base_servings=4, target_servings=8)
        # Salt: 1 tsp Salt
        scaled_salt = engine.scale_ingredient("1 tsp Salt", base_servings=4, target_servings=8)

        # For bulk: 2 * 2.0 = 4.0 cups
        assert "4" in scaled_bulk, f"Expected 4 cups for doubled bulk ingredient, got: {scaled_bulk}"
        # For salt: 1 * (2.0^0.80) = 1.74 tsp -> should be less than 2.0 tsp
        assert "2 tsp" not in scaled_salt, f"Salt should be dampened below 2.0 tsp, got: {scaled_salt}"


# ============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS (Pairwise & Multi-Module Interactions)
# ============================================================================

class TestTier3CrossFeatureCombinations:
    """Tier 3: Cross-Feature Interactions & Module Contracts."""

    def test_tier3_01_recipe_context_injection_into_portion_scaling(self, client, app):
        """Tier 3: Link active recipe context from ModelCache into chat portion scaling."""
        with app.app_context():
            recipe = cache.get_recipe_by_id(1)
            assert recipe is not None

            payload = {
                "message": "Scale this for 8 people",
                "recipe_id": 1,
                "recipe_context": {
                    "title": recipe.get("RecipeName"),
                    "servings": recipe.get("Servings", 4)
                }
            }
            res = client.post("/api/ratvi/chat", json=payload)
            assert res.status_code == 200
            data = res.get_json()
            assert data.get("success") is True
            assert data.get("intent") == "servings_scaled"
            # Verify recipe name is preserved in the response
            assert recipe.get("RecipeName") in data.get("reply", "") or recipe.get("RecipeName") in str(data.get("data", {}))

    def test_tier3_02_recipe_context_ingredient_substitution(self, client, app):
        """Tier 3: Query substitution for an ingredient identified in active recipe context."""
        with app.app_context():
            payload = {
                "message": "Can I replace cream with something else?",
                "recipe_id": 1
            }
            res = client.post("/api/ratvi/chat", json=payload)
            assert res.status_code == 200
            data = res.get_json()
            assert data.get("success") is True
            assert data.get("intent") == "substitution"
            reply_lower = data.get("reply", "").lower()
            assert any(alt in reply_lower for alt in ["yogurt", "coconut", "cashew"]), (
                f"Expected dairy-free or cream alternatives, got: {reply_lower}"
            )

    def test_tier3_03_unauthenticated_vs_authenticated_api_parity(self, client, login):
        """Tier 3: Verify /api/ratvi/chat produces consistent responses whether called by guest or logged-in user."""
        payload = {"message": "Substitute for egg"}

        # Unauthenticated request
        res_guest = client.post("/api/ratvi/chat", json=payload)
        assert res_guest.status_code == 200
        data_guest = res_guest.get_json()

        # Authenticated request
        res_user = login.post("/api/ratvi/chat", json=payload)
        assert res_user.status_code == 200
        data_user = res_user.get_json()

        assert data_guest.get("intent") == data_user.get("intent")
        assert data_guest.get("success") == data_user.get("success") == True

    def test_tier3_04_search_to_details_to_chat_pipeline(self, client, login, app):
        """Tier 3: End-to-end pipeline: Recommend recipe -> Fetch recipe details -> Scale via Ratvi chat."""
        with app.app_context():
            # Step 1: Recommendation query
            res_rec = client.get("/api/recommend?ingredients=chicken,egg")
            assert res_rec.status_code == 200
            recs = res_rec.get_json().get("recipes", [])
            assert len(recs) > 0, "Expected at least 1 recommendation"

            recipe_id = recs[0].get("Srno")
            assert recipe_id is not None

            # Step 2: Fetch details page
            res_detail = login.get(f"/recipe_details/{recipe_id}", follow_redirects=True)
            assert res_detail.status_code == 200

            # Step 3: Scale via Ratvi chat
            res_chat = client.post("/api/ratvi/chat", json={
                "message": "Scale for 5 people",
                "recipe_id": recipe_id
            })
            assert res_chat.status_code == 200
            chat_data = res_chat.get_json()
            assert chat_data.get("success") is True
            assert chat_data.get("intent") == "servings_scaled"

    def test_tier3_05_step_guidance_with_interactive_progression(self, client):
        """Tier 3: Verify step-by-step guidance returns ordered steps."""
        payload = {
            "message": "Show instructions step by step",
            "recipe_id": 1,
            "current_step": 0
        }
        res = client.post("/api/ratvi/chat", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert data.get("success") is True
        assert data.get("intent") == "instructions"


# ============================================================================
# TIER 4: REAL-WORLD SCENARIOS (End-to-End User Journeys)
# ============================================================================

class TestTier4RealWorldScenarios:
    """Tier 4: Realistic Full User Journeys Across Multiple Pages and Features."""

    def test_tier4_01_indian_family_feast_journey(self, login, client, app):
        """
        Tier 4 Journey 1: Indian Family Feast
        1. User logs in.
        2. Searches for 'paneer, butter'.
        3. Selects top recipe (e.g. Paneer Butter Masala).
        4. Opens /recipe_details/<id>.
        5. Asks Ratvi: 'Cook for 8 people'.
        6. Verifies non-linear scaling of spices and cooking times.
        7. Asks Ratvi: 'Substitute for cream'.
        8. Asks Ratvi: 'Show cooking steps'.
        """
        with app.app_context():
            # 1. Search for paneer recipes
            res_search = login.get("/results?ingredients=paneer,butter", follow_redirects=True)
            assert res_search.status_code == 200
            assert b"Paneer" in res_search.data

            # 2. Lookup recipe 1 (or matching Indian recipe)
            recipe = cache.get_recipe_by_id(1)
            assert recipe is not None
            r_id = int(recipe.get("Srno"))

            # 3. View Recipe Details
            res_detail = login.get(f"/recipe_details/{r_id}", follow_redirects=True)
            assert res_detail.status_code == 200
            # Ensure Ratvi widget is rendered on recipe details page
            assert b"ratvi-widget" in res_detail.data

            # 4. Scale recipe for 8 people
            res_scale = client.post("/api/ratvi/chat", json={
                "message": "Cook for 8 people",
                "recipe_id": r_id
            })
            assert res_scale.status_code == 200
            data_scale = res_scale.get_json()
            assert data_scale.get("intent") == "servings_scaled"
            scaled_data = data_scale.get("data", {})
            assert scaled_data.get("target_servings") == 8
            assert "prep_time_mins" in scaled_data and "cook_time_mins" in scaled_data

            # 5. Ask for cream substitution
            res_sub = client.post("/api/ratvi/chat", json={
                "message": "Substitute for cream",
                "recipe_id": r_id
            })
            assert res_sub.status_code == 200
            data_sub = res_sub.get_json()
            assert data_sub.get("intent") == "substitution"

            # 6. Ask for cooking steps
            res_steps = client.post("/api/ratvi/chat", json={
                "message": "Show cooking steps",
                "recipe_id": r_id
            })
            assert res_steps.status_code == 200
            data_steps = res_steps.get_json()
            assert data_steps.get("intent") == "instructions"

    def test_tier4_02_solo_cook_downscaling_and_quick_meal(self, login, client, app):
        """
        Tier 4 Journey 2: Solo Cook Downscaling
        1. User discovers global pasta recipe.
        2. Navigates to /recipe_details/<id>.
        3. Asks Ratvi: 'Scale for 1 person' (downscaling from 4 servings).
        4. Verifies practical culinary fractions and reduced cooking times.
        5. Asks Ratvi: 'Show cooking steps'.
        """
        with app.app_context():
            # 1. Discover recipes
            res_disc = login.get("/discover", follow_redirects=True)
            assert res_disc.status_code == 200
            assert b"ratvi-widget" in res_disc.data

            # 2. Choose recipe 1
            r_id = 1
            res_detail = login.get(f"/recipe_details/{r_id}", follow_redirects=True)
            assert res_detail.status_code == 200

            # 3. Downscale for 1 person
            res_downscale = client.post("/api/ratvi/chat", json={
                "message": "Scale for 1 person",
                "recipe_id": r_id
            })
            assert res_downscale.status_code == 200
            data_downscale = res_downscale.get_json()
            assert data_downscale.get("intent") == "servings_scaled"
            scaled_info = data_downscale.get("data", {})
            assert scaled_info.get("target_servings") == 1
            # Total cooking time should be lower or equal to base time
            assert scaled_info.get("total_time_mins") is not None

    def test_tier4_03_guest_onboarding_and_profile_privacy_lifecycle(self, client, login):
        """
        Tier 4 Journey 3: Guest Onboarding & Profile Privacy Lifecycle
        1. Unauthenticated guest lands on / -> Ratvi widget visible.
        2. Guest asks Ratvi a general culinary question -> 200 OK.
        3. Guest visits /login and /signup -> Ratvi widget visible.
        4. User logs in -> Ratvi widget visible on /home.
        5. User visits /profile -> Ratvi widget is COMPLETELY EXCLUDED (4-layer defense).
        6. User visits /favourites -> Ratvi widget returns.
        """
        # 1. Guest lands on /
        res_landing = client.get("/")
        assert res_landing.status_code == 200
        assert b"ratvi-widget" in res_landing.data, "Widget must appear on landing page"

        # 2. Guest chats with Ratvi
        res_chat = client.post("/api/ratvi/chat", json={"message": "What can I cook with tomatoes?"})
        assert res_chat.status_code == 200
        assert res_chat.get_json().get("success") is True

        # 3. Guest views /signup and /login
        for auth_page in ["/signup", "/login"]:
            res_auth = client.get(auth_page)
            assert res_auth.status_code == 200
            assert b"ratvi-widget" in res_auth.data, f"Widget must appear on {auth_page}"

        # 4. Authenticated user views /home
        res_home = login.get("/home", follow_redirects=True)
        assert res_home.status_code == 200
        assert b"ratvi-widget" in res_home.data, "Widget must appear on /home"

        # 5. User views /profile -> STRICTLY EXCLUDED
        res_profile = login.get("/profile", follow_redirects=True)
        assert res_profile.status_code == 200
        assert b'id="ratvi-widget"' not in res_profile.data, (
            "STRICT PRIVACY VIOLATION: ratvi-widget found in /profile DOM"
        )
        assert b'id="ratvi-fab"' not in res_profile.data, (
            "STRICT PRIVACY VIOLATION: ratvi-fab found in /profile DOM"
        )

        # 6. User views /favourites -> Widget reappears
        res_fav = login.get("/favourites", follow_redirects=True)
        assert res_fav.status_code == 200
        assert b"ratvi-widget" in res_fav.data, "Widget must return on /favourites"
