# Test Infrastructure Documentation: Harna Recipe System & Ratvi AI Companion

## Overview
This document specifies the end-to-end test infrastructure, execution environment, and test tier hierarchy for the Harna Recipe Recommendation System and the Ratvi AI Culinary Companion.

The test suite is structured as a **4-tier opaque-box test suite** located in `tests/test_e2e_ratvi.py`. It rigorously evaluates functionality against the interface contracts defined in `PROJECT.md` and user requirements in `ORIGINAL_REQUEST.md` without shortcuts or mocked tautologies.

---

## Environment & Runner Configuration

### 1. Python & Test Runner
- **Test Framework**: `pytest` (v7.4.4+)
- **Primary Test Runner Command**:
  ```powershell
  & ".\.venv\Scripts\python.exe" -m pytest -v tests/test_e2e_ratvi.py
  ```
- **Alternative (Anaconda Base with Python 3.12)**:
  ```powershell
  & "C:\Users\Mohith\anaconda3\python.exe" -m pytest -v tests/test_e2e_ratvi.py
  ```
- **Single Tier Execution**:
  ```powershell
  # Run Tier 1 only
  & ".\.venv\Scripts\python.exe" -m pytest -v tests/test_e2e_ratvi.py -k "TestTier1FeatureCoverage"

  # Run Tier 2 only
  & ".\.venv\Scripts\python.exe" -m pytest -v tests/test_e2e_ratvi.py -k "TestTier2BoundaryAndCorners"

  # Run Tier 3 only
  & ".\.venv\Scripts\python.exe" -m pytest -v tests/test_e2e_ratvi.py -k "TestTier3CrossFeatureCombinations"

  # Run Tier 4 only
  & ".\.venv\Scripts\python.exe" -m pytest -v tests/test_e2e_ratvi.py -k "TestTier4RealWorldScenarios"
  ```

### 2. Dependencies & Runtime Prerequisites
- `Flask` (3.0+)
- `Flask-Login` (0.6.3+)
- `Flask-SQLAlchemy` (3.1.1+)
- `pandas` (2.2.2+)
- `scikit-learn` (1.5.1+)
- `scipy` (1.13.1+)
- `nltk` (3.9.1+) with local tokenizers:
  - `tokenizers/punkt_tab` located in `C:\Users\Mohith\AppData\Roaming\nltk_data`
  - `tokenizers/punkt` located in `C:\Users\Mohith\AppData\Roaming\nltk_data`

---

## Test Architecture (4-Tier Structure)

The test suite spans **40 test cases** organized across 4 distinct verification tiers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        4-Tier Test Architecture                        │
├────────────────────────────────────────────────────────────────────────┤
│  Tier 1: Feature Coverage (Isolation / Happy Path) [16 Tests]          │
│  - R1: Model Artifacts, Schema, ModelCache, Recommendations (5 tests)  │
│  - R2: Floating UI Markup, Page Presence, /profile Exclusion (6 tests) │
│  - R3: /api/ratvi/chat Public Access, Scaling, Substitutions (5 tests) │
├────────────────────────────────────────────────────────────────────────┤
│  Tier 2: Boundary & Corner Cases (Limits / Fault Tolerance) [16 Tests] │
│  - R1: Nonexistent/Negative IDs, Empty Search, Special Chars (5 tests) │
│  - R2: Trailing Slash, Mobile CSS, Profile CSS Guard (5 tests)         │
│  - R3: Hyphenated/Unicode Fractions, Downscaling, Bad Payload (6 tests)│
├────────────────────────────────────────────────────────────────────────┤
│  Tier 3: Cross-Feature Combinations (Pairwise / Modules) [5 Tests]     │
│  - Recipe Context + Scaling, Active Substitutions, Search Pipeline     │
├────────────────────────────────────────────────────────────────────────┤
│  Tier 4: Real-World Scenarios (End-to-End User Journeys) [3 Tests]     │
│  - Indian Family Feast, Solo Cook Downscaling, Guest Onboarding        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Test Inventory & Requirements Traceability

### Tier 1: Feature Coverage in Isolation (`TestTier1FeatureCoverage`)
| Test Identifier | Target Requirement | Scope & Verification Contract |
|---|---|---|
| `test_tier1_r1_01_pickle_artifacts_exist` | R1 | Verifies `TFidf_matrix.pkl`, `Vectorizer_names.pkl`, `Vectorizer_ingredients.pkl`, `Recipes.pkl` exist in `data/processed/` and have size > 0. |
| `test_tier1_r1_02_recipes_dataframe_contract` | R1 | Verifies `Recipes.pkl` is a DataFrame with required columns (`Srno`, `RecipeName`, `Ingredients`, `Instructions`, `Cuisine`, `Servings`) and non-null `Srno`. |
| `test_tier1_r1_03_model_cache_singleton_loading` | R1 | Verifies `ModelCache` singleton (`cache`) loads in memory with populated `recipe_df` and `tfidf_combined`. |
| `test_tier1_r1_04_recipe_lookup_by_id` | R1 | Verifies `cache.get_recipe_by_id(1)` returns valid dictionary with `Srno == 1` and non-empty `RecipeName`. |
| `test_tier1_r1_05_recommendation_engine_dual_collection` | R1 | Verifies recommendation queries for Indian (`paneer, butter`) and Global (`pasta, cheese`) return dishes with `success: true`. |
| `test_tier1_r2_01_widget_markup_on_landing_page` | R2 | Verifies floating widget container, FAB pill, and panel markup on `/`. |
| `test_tier1_r2_02_widget_markup_on_auth_pages` | R2 | Verifies floating widget container and FAB pill on unauthenticated views `/login` and `/signup`. |
| `test_tier1_r2_03_widget_markup_on_core_app_views` | R2 | Verifies floating widget markup on authenticated app pages `/home`, `/results`, `/discover`, `/favourites`. |
| `test_tier1_r2_04_strict_exclusion_on_profile_view` | R2 | Verifies floating widget is strictly excluded from rendered HTML on `/profile` via Jinja conditional and block override. |
| `test_tier1_r2_05_theme_color_tokens_and_styling` | R2 | Verifies design system tokens Deep Teal (`#063238`) and Electric Lime (`#D4F620`) exist in stylesheets. |
| `test_tier1_r2_06_recipe_details_context_metadata` | R2 | Verifies recipe context hooks (e.g. data attributes, context pill) are present on `/recipe_details/<id>`. |
| `test_tier1_r3_01_public_unauthenticated_chat_access` | R3 | Verifies `POST /api/ratvi/chat` succeeds with `200 OK` (no `302 Redirect`) for guests, returning `success: true` and a reply. |
| `test_tier1_r3_02_serving_scaling_happy_path` | R3 | Verifies scaling request returns intent `servings_scaled` with `scaled_ingredients` and `target_servings == 6`. |
| `test_tier1_r3_03_culinary_substitution_happy_path` | R3 | Verifies substitution request returns intent `substitution` with valid alternatives and tips. |
| `test_tier1_r3_04_step_guidance_happy_path` | R3 | Verifies instructions query returns intent `instructions` with structured preparation steps. |
| `test_tier1_r3_05_hybrid_fallback_deterministic_offline` | R3 | Verifies offline deterministic engine runs smoothly without external LLM keys. |

### Tier 2: Boundary & Corner Cases (`TestTier2BoundaryAndCorners`)
| Test Identifier | Target Requirement | Scope & Verification Contract |
|---|---|---|
| `test_tier2_r1_01_nonexistent_recipe_id` | R1 | Verifies non-existent recipe ID (`9999999`) returns `None` or `404` without 500 server crash. |
| `test_tier2_r1_02_negative_and_zero_recipe_ids` | R1 | Verifies negative (`-1`) and zero (`0`) IDs handle cleanly. |
| `test_tier2_r1_03_empty_and_whitespace_recommendation_query` | R1 | Verifies queries with `""` and `"   "` return graceful 200 responses. |
| `test_tier2_r1_04_special_characters_search_sanitization` | R1 | Verifies search inputs with SQL injection, XSS tags, and symbols do not raise unhandled exceptions. |
| `test_tier2_r1_05_srno_uniqueness_and_no_id_collisions` | R1 | Verifies all `Srno` values in `Recipes.pkl` are unique with 0 duplicate keys. |
| `test_tier2_r2_01_profile_trailing_slash_and_query_params` | R2 | Verifies `/profile/` and `/profile?view=security` maintain widget exclusion. |
| `test_tier2_r2_02_recipe_title_special_characters_escaping` | R2 | Verifies quotes and ampersands in titles do not break HTML attributes. |
| `test_tier2_r2_03_mobile_css_drawer_media_query` | R2 | Verifies `@media (max-width: 768px)` exists in `ratvi-chat.css` for responsive drawer styling. |
| `test_tier2_r2_04_css_profile_display_none_guard` | R2 | Verifies CSS 4th-layer defense `body.page-profile #ratvi-widget { display: none !important; }`. |
| `test_tier2_r2_05_suggestion_chips_tray_structure` | R2 | Verifies `#ratvi-chips` container is present in the widget markup. |
| `test_tier2_r3_01_hyphenated_fractions_parsing` | R3 | Verifies fractions with hyphens (e.g. `1-1/2 cups`) are parsed as `1.5` without `ValueError`. |
| `test_tier2_r3_02_unicode_fractions_parsing` | R3 | Verifies unicode fractions (`½`, `¼`, `¾`) are parsed and scaled accurately. |
| `test_tier2_r3_03_downscaling_sub_unitary_ratio` | R3 | Verifies downscaling (ratio $0.25$) yields positive, reduced quantities and times. |
| `test_tier2_r3_04_extreme_upscaling_stress` | R3 | Verifies massive upscaling ($50$ servings) executes without numerical overflow. |
| `test_tier2_r3_05_missing_or_empty_message_payload` | R3 | Verifies empty JSON or blank message payload returns `400 Bad Request`. |
| `test_tier2_r3_06_non_linear_damping_comparison` | R3 | Verifies salt/spices damp significantly more than bulk base ingredients when scaled. |

### Tier 3: Cross-Feature Combinations (`TestTier3CrossFeatureCombinations`)
| Test Identifier | Scope & Verification Contract |
|---|---|
| `test_tier3_01_recipe_context_injection_into_portion_scaling` | Injects recipe metadata from `ModelCache` into chat scaling request, verifying returned adjustments and time estimations. |
| `test_tier3_02_recipe_context_ingredient_substitution` | Requests ingredient substitution for a component in the active recipe context. |
| `test_tier3_03_unauthenticated_vs_authenticated_api_parity` | Verifies identical chat API behavior between guest and authenticated sessions. |
| `test_tier3_04_search_to_details_to_chat_pipeline` | Complete module pipeline: Search recommendations -> Select recipe -> View recipe details -> Scale via chat API. |
| `test_tier3_05_step_guidance_with_interactive_progression` | Verifies step-by-step guidance returns ordered steps. |

### Tier 4: Real-World Scenarios (`TestTier4RealWorldScenarios`)
| Test Identifier | Scope & Verification Contract |
|---|---|
| `test_tier4_01_indian_family_feast_journey` | Comprehensive journey: Search Indian dish -> view details -> scale for 8 people -> ask dairy-free substitution -> request cooking steps. |
| `test_tier4_02_solo_cook_downscaling_and_quick_meal` | Solo cook journey: Discover global recipe -> view details -> scale down for 1 person -> verify practical cooking fractions and reduced time -> get steps. |
| `test_tier4_03_guest_onboarding_and_profile_privacy_lifecycle` | Multi-view privacy lifecycle: Guest interacts on landing `/` -> views auth pages -> logs in -> visits `/profile` (verifies 100% absence) -> visits `/favourites` (verifies widget reappears). |

---

## Test Fixture Infrastructure

The test suite leverages modular fixtures from `tests/conftest.py`:
- `app`: Instantiates the Flask application with testing configuration and an isolated SQLite in-memory database (`sqlite:///:memory:`).
- `client`: Provides an unauthenticated HTTP test client with session cookies.
- `login`: Creates a unique temporary test user and authenticates the client session for protected route access.
- `cache`: Directly accesses the in-memory `ModelCache` singleton for model assertions.
