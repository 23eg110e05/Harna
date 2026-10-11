# Project: Harna Recipe System & Ratvi AI Culinary Companion

## Architecture
Harna is a Flask-based recipe recommendation and culinary guidance web application. It combines:
1. **Machine Learning Recommendation Engine**: TF-IDF cosine similarity over recipe titles and ingredients, loaded in-memory via `ModelCache` singleton (`my_flask_app/utils/cache.py`).
2. **Dual-Engine Intelligence Backend**: `/api/ratvi/chat` providing serving-based non-linear ingredient calculations, culinary substitutions, and step-by-step guidance, backed by a deterministic culinary engine with optional external LLM support.
3. **Floating Ratvi AI Companion Chatbot Interface**: Responsive floating widget styled with Harna's Deep Teal (`#063238`) and Electric Lime (`#D4F620`), rendered on all required views (`/`, `/home`, `/results`, `/recipe_details/<id>`, `/discover`, `/favourites`, `/login`, `/signup`), strictly excluded on `/profile`, with recipe context awareness on recipe detail views.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Frontend Client (Browser)                         │
│  - Standalone Pages: /, /login, /signup                                     │
│  - App Pages (extends base.html): /home, /results, /discover, /favourites   │
│  - Recipe Details: /recipe_details/<id> (Active Recipe Context)             │
│  - Profile: /profile (Widget STRICTLY EXCLUDED via 4-layer defense)         │
│  - Floating Ratvi Widget: FAB pill, modal, dynamic suggestion chips         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ POST /api/ratvi/chat
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Flask Backend Application                          │
│  - App Factory: my_flask_app/__init__.py                                    │
│  - Blueprint: main.py (/api/ratvi/chat, /api/recommend, /api/recipes/<id>)  │
│  - ModelCache Singleton: my_flask_app/utils/cache.py                         │
└──────────────────────┬───────────────────────────────┬──────────────────────┘
                       │                               │
                       ▼                               ▼
       ┌───────────────────────────────┐   ┌───────────────────────────────┐
       │   Dual-Engine Ratvi Core      │   │  Pre-trained ML Artifacts     │
       │   (my_flask_app/utils/ratvi.py)│   │  (data/processed/)            │
       │  - Fraction Parser (1-1/2)    │   │  - TFidf_matrix.pkl           │
       │  - Non-Linear Scaler (R^alpha)│   │  - Vectorizer_names.pkl       │
       │  - 35+ Culinary Substitutions │   │  - Vectorizer_ingredients.pkl │
       │  - Step-by-Step Guide         │   │  - Recipes.pkl (25,655 rows)  │
       │  - Deterministic & LLM Engine │   └───────────────────────────────┘
       └───────────────────────────────┘
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Model Training Path Fix | Fix `os.path.dirname` double invocation in `train_harna_models.py` line 72 so `raw_dir` and `processed_dir` resolve correctly to repository paths | M1 | Survey E1 |
| 2 | Dataset Aggregation & Schema Alignment | Aggregate 5,655 Indian recipes from `Final_Indian_Dataset.csv` with 20,000 curated global recipes from `recipes_data.csv`, assigning non-colliding `Srno` from 14,168 and nutrition defaults | M1 | Survey E1 |
| 3 | Model Artifact Serialization | Serialize `TFidf_matrix.pkl` (25,655 rows), `Vectorizer_names.pkl`, `Vectorizer_ingredients.pkl`, and `Recipes.pkl` into `data/processed/` using Anaconda Python | M1 | Survey E1 |
| 4 | ModelCache In-Memory Loading | Verify `ModelCache.load_models()` and `get_recipe_by_id()` correctly load and serve the updated combined dataset | M1 | Survey E1 |
| 5 | Indian & International Recommendations | Ensure recommendation queries (e.g. `paneer, butter`, `beef, mushroom`, `chicken, egg`, `pasta, cheese`) return relevant recipes from both collections | M1 | Survey E1 |
| 6 | Robust Culinary Fraction Parser | Parse fractions, mixed numbers, decimals, unicode fractions, and hyphenated fractions (`1-1/2`) without crashing | M2 | Survey E2 |
| 7 | Non-Linear Ingredient Scaling | Scale ingredients by factor $R$ using power-law damping for bulk ($1.0$), liquids ($0.85$), salt/acids ($0.80$), pungent spices ($0.80$), and cooking fats ($0.75$), with cooking time adjustments | M2 | Survey E2 |
| 8 | Expanded Culinary Substitutions | 35+ items across Indian and Global cuisines with exact substitution ratios, conversion tips, and dietary tags | M2 | Survey E2 |
| 9 | Step-by-Step Cooking Guidance | Structured preparation instructions breakdown with interactive progression and timer extraction | M2 | Survey E2 |
| 10 | Dual-Engine Hybrid Fallback | Deterministic offline engine (<5ms) with optional external LLM API support via `.env` and automatic graceful fallback | M2 | Survey E2 |
| 11 | Persistent Chat Endpoint `/api/ratvi/chat` | Public unauthenticated POST endpoint in `my_flask_app/main.py` handling chat requests, context injection, and returning structured responses | M2 | Survey E2 |
| 12 | Floating Action Pill (FAB) Widget | Bottom-right collapsible action pill matching Deep Teal (`#063238`) and Electric Lime (`#D4F620`) theme with pulsing status dot | M3 | Survey E3 |
| 13 | Chat Message Interface & History | Expandable modal panel (380x560px) with message history, bot/user bubbles, typing indicator, clear chat, and smooth transitions | M3 | Survey E3 |
| 14 | Dynamic Suggestion Chips | Suggestion chips ("Scale for 5 people", "Substitute for butter", "Show cooking steps") dynamically adapting to recipe context | M3 | Survey E3 |
| 15 | Recipe Context-Awareness Integration | Automatically detect `/recipe_details/<id>` via URL and DOM metadata, display context header pill, and pass `recipe_id` in API payload | M3 | Survey E3 |
| 16 | Universal View Integration | Render floating widget on `/`, `/home`, `/results`, `/recipe_details/<id>`, `/discover`, `/favourites`, `/login`, `/signup` | M3 | Survey E3 |
| 17 | Strict `/profile` Exclusion | 4-layer defense-in-depth exclusion (Jinja condition in `base.html`, block override in `profile.html`, JS route guard, CSS display none) | M3 | Survey E3 |
| 18 | Mobile Responsive UI | Fully responsive layout adjusting to full-width drawer on screens $\le 768px$ with accessible touch targets | M3 | Survey E3 |
| 19 | E2E Test Suite (Tiers 1-4) | Comprehensive opaque-box test suite verifying all features across model pipeline, backend API, and UI integration | M4 / Test Track | Requirements |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Model Training Pipeline & Serialization | Fix path bug in `train_harna_models.py`, aggregate Indian & global datasets, serialize all 4 pickle files to `data/processed/`, verify `ModelCache` and dual-collection recommendations | none | PLANNED |
| M2 | Dual-Engine Intelligence Backend | Implement fraction parser, non-linear scaling, substitutions, step guidance, hybrid fallback in `my_flask_app/utils/ratvi.py`, and expose `/api/ratvi/chat` in `my_flask_app/main.py` | none | PLANNED |
| M3 | Floating Ratvi UI & Template Integration | Implement `components/ratvi_chat.html`, `ratvi-chat.css`, `ratvi-chat.js`, inject across target views, strictly exclude on `/profile`, wire recipe context awareness | M2 | PLANNED |
| M4 | Final Milestone: 100% E2E Pass & Hardening | Pass 100% of E2E tests (Tiers 1-4), execute Tier 5 adversarial coverage hardening with challenger-worker-reviewer cycle | M1, M2, M3, TEST_READY | PLANNED |

## Interface Contracts

### 1. Model Artifact Contract (`data/processed/`)
- `TFidf_matrix.pkl`: `scipy.sparse.csr_matrix` of shape `(25655, N_features)`
- `Vectorizer_names.pkl`: `sklearn.feature_extraction.text.TfidfVectorizer`
- `Vectorizer_ingredients.pkl`: `sklearn.feature_extraction.text.TfidfVectorizer`
- `Recipes.pkl`: `pandas.DataFrame` with 25,655 rows containing columns:
  * `Srno` (`int`, 1 to 34,167)
  * `RecipeName` (`str`), `TranslatedRecipeName` (`str`)
  * `Ingredients` (`str`), `TranslatedIngredients` (`str`)
  * `Instructions` (`list` or `str`), `TranslatedInstructions` (`str`)
  * `Cuisine` (`str`), `Course` (`str`), `Diet` (`str`)
  * `PrepTimeInMins` (`float`), `CookTimeInMins` (`float`), `TotalTimeInMins` (`float`), `Servings` (`float`)
  * Nutrition columns: `Calories (kcal)`, `Protein (g)`, `Carbohydrates (g)`, `Fats (g)`, `Fiber (g)` and their `.1` counterparts

### 2. Dual-Engine Chat API Contract (`/api/ratvi/chat`)
- **Method**: `POST`
- **Auth**: Public (no login required)
- **Request Body**:
  ```json
  {
    "message": "string (required)",
    "recipe_id": "integer (optional)",
    "recipe_context": {"title": "string", "servings": "number"} (optional),
    "history": [{"role": "user|assistant", "content": "string"}] (optional),
    "current_step": "integer (optional)",
    "options": {"engine": "hybrid|deterministic|llm"} (optional)
  }
  ```
- **Response Body**:
  ```json
  {
    "success": true,
    "reply": "string (formatted markdown)",
    "intent": "servings_scaled|substitution|instructions|general_chat",
    "engine_used": "deterministic|llm",
    "data": {
      "recipe_id": 12,
      "recipe_name": "string",
      "base_servings": 4,
      "target_servings": 5,
      "ratio": 1.25,
      "scaled_ingredients": ["string"],
      "prep_time_mins": 15,
      "cook_time_mins": 25,
      "total_time_mins": 40
    },
    "suggestions": ["string"]
  }
  ```

### 3. Frontend Chat Widget Contract
- Element ID: `#ratvi-widget`
- Collapsed Button: `#ratvi-fab`
- Panel: `#ratvi-chat-panel`
- Context Bar: `#ratvi-context-bar`
- Message Stream: `#ratvi-messages`
- Chips Container: `#ratvi-chips`
- Form: `#ratvi-form`, Input: `#ratvi-input`

## Code Layout
- `train_harna_models.py`: Root level training script
- `my_flask_app/`:
  - `__init__.py`: App factory and extension setup
  - `main.py`: Main blueprint routes, includes `/api/ratvi/chat`
  - `auth.py`: Authentication routes
  - `utils/`:
    - `cache.py`: `ModelCache` singleton
    - `ratvi.py`: Dual-Engine Ratvi Core (Scaling, Substitutions, Guidance, Hybrid Engine)
    - `preprocess.py`: Text preprocessing utilities
  - `templates/`:
    - `base.html`: Common layout (includes ratvi widget with `/profile` exclusion)
    - `index.html`: Landing page (includes ratvi widget)
    - `login.html`: Login page (includes ratvi widget)
    - `signup.html`: Signup page (includes ratvi widget)
    - `profile.html`: Profile page (strictly excludes ratvi widget)
    - `recipe_details.html`: Recipe details page (data attributes for recipe context)
    - `components/ratvi_chat.html`: Reusable Ratvi widget markup
  - `static/`:
    - `css/design-system.css`: Harna design system tokens
    - `css/ratvi-chat.css`: Ratvi chatbot styles
    - `scripts/ratvi-chat.js`: Ratvi chatbot client controller
- `data/`:
  - `raw/`: Raw CSV datasets
  - `processed/`: Serialized pickle models and DataFrames
- `tests/`:
  - `test_e2e_ratvi.py`: End-to-end integration test suite
