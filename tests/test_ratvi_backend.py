"""
Automated Unit and Integration Test Suite for Milestone M2:
Dual-Engine Intelligence Backend (Ratvi), Non-Linear Ingredient Scaling,
Culinary Substitutions (35+ items), Step-by-Step Guidance, and Hybrid Fallback.
"""

import time
import pytest
import pandas as pd
from my_flask_app.utils.ratvi import (
    RatviEngine,
    ratvi,
    parse_culinary_fraction,
    format_culinary_fraction,
    EXPANDED_SUBSTITUTIONS
)
from my_flask_app.utils.cache import cache


# ============================================================================
# 1. FRACTION PARSING & FORMATTING TESTS
# ============================================================================

class TestFractionParsingAndFormatting:
    """Tests for parse_culinary_fraction and format_culinary_fraction."""

    def test_fraction_parsing_standard(self):
        """Standard fraction parsing as specified by Explorer 2 contract."""
        assert parse_culinary_fraction("1-1/2") == 1.5
        assert parse_culinary_fraction("1/2") == 0.5
        assert parse_culinary_fraction("2 1/4") == 2.25
        assert parse_culinary_fraction("½") == 0.5
        assert parse_culinary_fraction("3") == 3.0

    def test_fraction_parsing_hyphenated(self):
        """Hyphenated fractions commonly found in Indian datasets."""
        assert parse_culinary_fraction("1-1/2") == 1.5
        assert parse_culinary_fraction("2-3/4") == 2.75
        assert parse_culinary_fraction("1-1/4") == 1.25
        assert parse_culinary_fraction("3-1/3") == pytest.approx(10.0 / 3.0)

    def test_fraction_parsing_unicode(self):
        """Unicode fractions (½, ¼, ¾, ⅓, ⅔, ⅛) standalone and with prefixes."""
        assert parse_culinary_fraction("½") == 0.5
        assert parse_culinary_fraction("¼") == 0.25
        assert parse_culinary_fraction("¾") == 0.75
        assert parse_culinary_fraction("⅓") == pytest.approx(1.0 / 3.0)
        assert parse_culinary_fraction("⅔") == pytest.approx(2.0 / 3.0)
        assert parse_culinary_fraction("⅛") == 0.125
        assert parse_culinary_fraction("1 ½") == 1.5
        assert parse_culinary_fraction("2 ¼") == 2.25
        assert parse_culinary_fraction("1½") == 1.5

    def test_fraction_parsing_decimals_and_integers(self):
        """Standard decimals and integer strings."""
        assert parse_culinary_fraction("1.5") == 1.5
        assert parse_culinary_fraction("0.75") == 0.75
        assert parse_culinary_fraction("4") == 4.0
        assert parse_culinary_fraction(2.5) == 2.5
        assert parse_culinary_fraction(3) == 3.0

    def test_fraction_parsing_ranges(self):
        """Numeric range strings like '1-2' or '2 to 3'."""
        assert parse_culinary_fraction("1-2") == 1.5
        assert parse_culinary_fraction("2 to 4") == 3.0

    def test_fraction_parsing_invalid(self):
        """Empty, None, or unparseable strings raise ValueError."""
        with pytest.raises(ValueError):
            parse_culinary_fraction("")
        with pytest.raises(ValueError):
            parse_culinary_fraction(None)
        with pytest.raises(ValueError):
            parse_culinary_fraction("to taste")

    def test_format_culinary_fraction_increments(self):
        """Format continuous floats into practical culinary fractions."""
        assert format_culinary_fraction(0.1) == "1/8"
        assert format_culinary_fraction(0.25) == "1/4"
        assert format_culinary_fraction(0.33) == "1/3"
        assert format_culinary_fraction(0.5) == "1/2"
        assert format_culinary_fraction(0.67) == "2/3"
        assert format_culinary_fraction(0.75) == "3/4"
        assert format_culinary_fraction(1.0) == "1"
        assert format_culinary_fraction(1.5) == "1 1/2"
        assert format_culinary_fraction(1.74) == "1 3/4"
        assert format_culinary_fraction(2.25) == "2 1/4"
        assert format_culinary_fraction(3.6) in ["3 2/3", "3.6"]
        assert format_culinary_fraction(12.0) == "12"
        assert format_culinary_fraction(50.0) == "50"


# ============================================================================
# 2. NON-LINEAR INGREDIENT SCALING TESTS
# ============================================================================

class TestNonLinearScaling:
    """Tests for power-law ingredient scaling Q_scaled = Q_base * (R ^ alpha)."""

    def test_nonlinear_scaling_spices_and_liquids(self):
        """Verification of dampening for pungent spices and liquids from Explorer 2 contract."""
        engine = RatviEngine()
        # Test spice damping when doubling recipe (4 -> 8 servings, R = 2.0)
        # R^0.80 = 2.0^0.80 = 1.74 -> ~1 3/4 tsp
        res_spice = engine.scale_ingredient("1 tsp Red chilli powder", base_servings=4, target_servings=8)
        assert "1 3/4" in res_spice or "1.7" in res_spice

        # Test liquid damping when scaling 4 -> 8 servings
        # 2 * 2.0^0.85 = 2 * 1.80 = 3.60 -> ~3 2/3 cups
        res_liquid = engine.scale_ingredient("2 cups Water", base_servings=4, target_servings=8)
        assert "3.6" in res_liquid or "3 1/2" in res_liquid or "3 2/3" in res_liquid

    def test_nonlinear_scaling_salt_and_fats(self):
        """Dampening for salt (alpha=0.80) and cooking oils/fats (alpha=0.75)."""
        engine = RatviEngine()
        # Salt: 1 tsp Salt, 4 -> 8 servings
        res_salt = engine.scale_ingredient("1 tsp Salt", base_servings=4, target_servings=8)
        assert "1 3/4" in res_salt or "1.7" in res_salt
        assert "2 tsp" not in res_salt  # Must not scale strictly linearly

        # Oil: 2 tbsp Olive oil, 4 -> 8 servings
        # 2 * 2.0^0.75 = 2 * 1.6818 = 3.36 -> ~3 1/3 tbsp
        res_oil = engine.scale_ingredient("2 tbsp Olive oil", base_servings=4, target_servings=8)
        assert "3 1/3" in res_oil or "3.4" in res_oil or "3 1/2" in res_oil
        assert "4 tbsp" not in res_oil

    def test_linear_scaling_bulk_ingredients(self):
        """Bulk ingredients scale linearly (alpha = 1.00)."""
        engine = RatviEngine()
        res_bulk = engine.scale_ingredient("2 cups Basmati Rice", base_servings=4, target_servings=8)
        assert "4 cups" in res_bulk

        res_paneer = engine.scale_ingredient("200 grams Paneer", base_servings=4, target_servings=8)
        assert "400 grams" in res_paneer

    def test_downscaling_sub_unitary_servings(self):
        """Downscaling from 4 servings to 1 person (R = 0.25)."""
        engine = RatviEngine()
        res_bulk = engine.scale_ingredient("2 cups Rice", base_servings=4, target_servings=1)
        assert "1/2 cup" in res_bulk or "0.5" in res_bulk

        res_salt = engine.scale_ingredient("1 tsp Salt", base_servings=4, target_servings=1)
        # 1 * 0.25^0.95 = 0.267 -> ~1/4 tsp
        assert "1/4 tsp" in res_salt
        assert "-0" not in res_salt

    def test_cooking_time_scaling(self):
        """Prep time scales as R^0.50, cook time scales as R^0.25."""
        engine = RatviEngine()
        recipe = {
            'RecipeName': 'Test Dal',
            'Servings': 4,
            'PrepTimeInMins': 16,
            'CookTimeInMins': 32,
            'Ingredients': '1 cup Toor Dal, 3 cups Water, 1 tsp Salt'
        }
        # Double to 8 servings: R = 2.0
        # prep: 16 * 2^0.5 = 16 * 1.414 = 22.6 -> ~23 mins
        # cook: 32 * 2^0.25 = 32 * 1.189 = 38.0 -> ~38 mins
        scaled = engine.scale_recipe_servings(recipe, 8)
        assert scaled['target_servings'] == 8
        assert scaled['prep_time_mins'] == 23
        assert scaled['cook_time_mins'] == 38
        assert scaled['total_time_mins'] == 61

    def test_qualitative_ingredient_preservation(self):
        """Qualitative ingredients like 'Salt - to taste' are safely preserved."""
        engine = RatviEngine()
        res = engine.scale_ingredient("Salt - to taste", base_servings=4, target_servings=8)
        assert "to taste" in res.lower()


# ============================================================================
# 3. 35+ CULINARY SUBSTITUTIONS TESTS
# ============================================================================

class TestCulinarySubstitutions:
    """Tests for 35+ item culinary substitutions knowledge base."""

    def test_substitution_database_size(self):
        """Knowledge base must contain >= 35 distinct ingredient keys."""
        assert len(EXPANDED_SUBSTITUTIONS) >= 35, f"Expected >= 35 substitutions, got {len(EXPANDED_SUBSTITUTIONS)}"

    def test_substitution_butter(self):
        """Butter substitution returns ghee and oil with advice."""
        engine = RatviEngine()
        res = engine.find_substitutions("butter")
        assert res['ingredient'] == "butter"
        assert len(res['substitutes']) >= 3
        assert any("ghee" in s.lower() for s in res['substitutes'])
        assert any("oil" in s.lower() for s in res['substitutes'])
        assert "ratio" in res
        assert "tip" in res

    def test_substitution_cream_and_dairy(self):
        """Dairy cream, sour cream, and milk substitutions."""
        engine = RatviEngine()
        res_cream = engine.find_substitutions("cream")
        assert any("yogurt" in s.lower() or "cashew" in s.lower() or "coconut" in s.lower() for s in res_cream['substitutes'])

        res_paneer = engine.find_substitutions("paneer")
        assert any("tofu" in s.lower() for s in res_paneer['substitutes'])

    def test_substitution_egg(self):
        """Egg substitution provides flaxseed and vegan alternatives."""
        engine = RatviEngine()
        res = engine.find_substitutions("egg")
        assert any("flax" in s.lower() or "chia" in s.lower() or "tofu" in s.lower() for s in res['substitutes'])

    def test_substitution_indian_spices(self):
        """Indian spices: garam masala, chaat masala, hing, amchur, kasuri methi."""
        engine = RatviEngine()
        res_garam = engine.find_substitutions("garam masala")
        assert any("cumin" in s.lower() or "curry" in s.lower() for s in res_garam['substitutes'])

        res_hing = engine.find_substitutions("asafoetida")
        assert any("garlic" in s.lower() for s in res_hing['substitutes'])

        res_amchur = engine.find_substitutions("amchur")
        assert any("lemon" in s.lower() for s in res_amchur['substitutes'])

    def test_substitution_unknown_fallback(self):
        """Unknown ingredient returns graceful generic substitution structure."""
        engine = RatviEngine()
        res = engine.find_substitutions("dragonfruit-truffle-oil")
        assert "substitutes" in res
        assert len(res['substitutes']) > 0
        assert "tip" in res


# ============================================================================
# 4. STEP-BY-STEP GUIDANCE TESTS
# ============================================================================

class TestStepGuidance:
    """Tests for instruction parsing, timer extraction, and step navigation."""

    def test_extract_steps_from_list_and_string(self):
        """Extracts steps from list representation or raw sentence strings."""
        engine = RatviEngine()
        # List of steps
        rec_list = {'Instructions': ['Chop the onions.', 'Heat the oil in pan.', 'Add spices and simmer for 15 mins.']}
        steps_list = engine.extract_steps(rec_list)
        assert len(steps_list) == 3
        assert "Chop the onions" in steps_list[0]

        # String of numbered steps
        rec_str = {'Instructions': '1. Heat ghee in a kadhai. 2. Add cumin seeds and let splutter. 3. Simmer for 10 minutes.'}
        steps_str = engine.extract_steps(rec_str)
        assert len(steps_str) == 3

    def test_extract_timers(self):
        """Extracts cooking timers accurately."""
        engine = RatviEngine()
        step = "Simmer the dal gently for 15 to 20 mins until soft, then rest for 5 minutes."
        timers = engine.extract_timers(step)
        assert len(timers) >= 1
        assert any("15 to 20" in t for t in timers)

    def test_step_navigation_next_back(self):
        """Step navigation increments or decrements current_step cleanly."""
        engine = RatviEngine()
        recipe = {
            'RecipeName': 'Khichdi',
            'Instructions': ['Wash rice and dal.', 'Pressure cook for 15 mins.', 'Temper with cumin and ghee.']
        }
        ctx = {'current_recipe': recipe}

        # Step 1
        res1 = engine._process_deterministic("step 1", context=ctx, current_step=0)
        assert res1['intent'] == 'instructions'
        assert res1['data']['step_number'] == 1

        # Next -> Step 2
        res2 = engine._process_deterministic("next", context=ctx, current_step=0)
        assert res2['data']['step_number'] == 2


# ============================================================================
# 5. HYBRID FALLBACK & DETERMINISTIC ENGINE TESTS
# ============================================================================

class TestHybridFallbackEngine:
    """Tests for offline deterministic execution and hybrid fallback safety."""

    def test_deterministic_offline_latency(self):
        """Deterministic engine executes in <15ms."""
        engine = RatviEngine()
        recipe = {
            'RecipeName': 'Quick Curry',
            'Servings': 4,
            'PrepTimeInMins': 10,
            'CookTimeInMins': 20,
            'Ingredients': '1 cup Dal, 2 cups Water, 1 tsp Salt'
        }
        ctx = {'current_recipe': recipe}

        start = time.perf_counter()
        res = engine.process_chat("Scale for 6 people", context=ctx, options={'engine': 'deterministic'})
        duration_ms = (time.perf_counter() - start) * 1000

        assert duration_ms < 15.0, f"Deterministic engine took {duration_ms:.2f}ms, expected <15ms"
        assert res['success'] is True
        assert res['intent'] == 'servings_scaled'
        assert res['engine_used'] == 'deterministic'

    def test_hybrid_fallback_without_api_key(self, monkeypatch):
        """Without API key in environment, hybrid mode safely returns deterministic result."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        engine = RatviEngine()
        res = engine.process_chat("How do I make dal?", options={'engine': 'hybrid'})
        assert res['success'] is True
        assert res['engine_used'] == 'deterministic'
        assert len(res['reply']) > 0

    def test_hybrid_fallback_on_simulated_network_error(self, monkeypatch):
        """When LLM provider raises an exception, engine gracefully falls back to deterministic."""
        monkeypatch.setenv("GEMINI_API_KEY", "dummy_invalid_key_12345")

        def mock_raise(*args, **kwargs):
            raise ConnectionError("Simulated network drop")

        monkeypatch.setattr(RatviEngine, "_process_llm", mock_raise)

        engine = RatviEngine()
        res = engine.process_chat("Substitute for butter", options={'engine': 'hybrid'})
        assert res['success'] is True
        assert res['engine_used'] == 'deterministic'
        assert res['intent'] == 'substitution'


# ============================================================================
# 6. REST API ENDPOINT /api/ratvi/chat TESTS
# ============================================================================

class TestApiRatviChatEndpoint:
    """Flask client tests for POST /api/ratvi/chat."""

    def test_api_ratvi_chat_unauthenticated_guest(self, client):
        """Unauthenticated guest calls to /api/ratvi/chat must return 200 OK (no 302 redirect)."""
        res = client.post("/api/ratvi/chat", json={"message": "Hello Ratvi"})
        assert res.status_code == 200
        data = res.get_json()
        assert data['success'] is True
        assert len(data['reply']) > 0
        assert data['intent'] in ['general_chat', 'welcome']
        assert 'suggestions' in data

    def test_api_ratvi_chat_authenticated_parity(self, client, login):
        """Authenticated and guest users receive identical contract structures."""
        payload = {"message": "Substitute for egg"}
        res_guest = client.post("/api/ratvi/chat", json=payload)
        res_user = login.post("/api/ratvi/chat", json=payload)

        assert res_guest.status_code == 200
        assert res_user.status_code == 200
        assert res_guest.get_json()['intent'] == res_user.get_json()['intent'] == 'substitution'

    def test_api_ratvi_chat_serving_scaling(self, client, app):
        """Serving scaling request with recipe_id returns calculated portions and times."""
        with app.app_context():
            res = client.post("/api/ratvi/chat", json={
                "message": "Scale for 8 people",
                "recipe_id": 1
            })
            assert res.status_code == 200
            data = res.get_json()
            assert data['success'] is True
            assert data['intent'] == 'servings_scaled'
            assert data['data']['target_servings'] == 8
            assert len(data['data']['scaled_ingredients']) > 0
            assert "prep_time_mins" in data['data']
            assert "cook_time_mins" in data['data']

    def test_api_ratvi_chat_recipe_context_dict(self, client):
        """Client passes recipe_context dictionary directly in request payload."""
        payload = {
            "message": "Cook for 6 people",
            "recipe_context": {
                "title": "Custom Pasta",
                "servings": 2,
                "Ingredients": "200g Pasta, 1 cup Tomato sauce, 1 tsp Salt",
                "PrepTimeInMins": 10,
                "CookTimeInMins": 15
            }
        }
        res = client.post("/api/ratvi/chat", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert data['success'] is True
        assert data['intent'] == 'servings_scaled'
        assert data['data']['target_servings'] == 6

    def test_api_ratvi_chat_step_guidance(self, client, app):
        """Step-by-step cooking guide query returns structured instructions."""
        with app.app_context():
            res = client.post("/api/ratvi/chat", json={
                "message": "Show cooking steps",
                "recipe_id": 1
            })
            assert res.status_code == 200
            data = res.get_json()
            assert data['success'] is True
            assert data['intent'] == 'instructions'
            assert len(data['reply']) > 0

    def test_api_ratvi_chat_missing_message_400(self, client):
        """Missing or whitespace-only message returns 400 Bad Request."""
        res_empty = client.post("/api/ratvi/chat", json={})
        assert res_empty.status_code == 400
        assert res_empty.get_json()['success'] is False

        res_ws = client.post("/api/ratvi/chat", json={"message": "   "})
        assert res_ws.status_code == 400
        assert res_ws.get_json()['success'] is False


# ============================================================================
# 7. PD.ISNA LIST BUG REGRESSION TESTS
# ============================================================================

class TestPdIsnaListBugFix:
    """Verify pd.isna bug on list types is resolved in main.py."""

    def test_api_recipes_detail_no_pd_isna_crash(self, client, app):
        """GET /api/recipes/<id> must not raise ValueError when Instructions is a list."""
        with app.app_context():
            res = client.get("/api/recipes/1")
            assert res.status_code == 200
            data = res.get_json()
            assert data['success'] is True
            assert 'recipe' in data
            assert isinstance(data['recipe']['Instructions'], (list, str))

    def test_api_recommend_no_pd_isna_crash_and_integer_srno(self, client, app):
        """GET /api/recommend returns valid integer Srno and handles list instructions without 500 error."""
        with app.app_context():
            res = client.get("/api/recommend?ingredients=chicken,egg")
            assert res.status_code == 200
            data = res.get_json()
            assert data['success'] is True
            recipes = data.get('recipes', [])
            assert len(recipes) > 0
            # Ensure Srno is integer
            srno = recipes[0].get('Srno')
            assert isinstance(srno, int), f"Expected Srno to be integer, got {type(srno)}: {srno}"
