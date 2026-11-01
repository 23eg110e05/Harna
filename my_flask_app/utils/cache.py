import os
import pickle
import pandas as pd
import re
from nltk.stem import PorterStemmer

_stemmer = PorterStemmer()

# All common staples, seasonings, aromatics, spices, cooking mediums, and common descriptor words
PANTRY_STAPLES = {
    'salt', 'water', 'oil', 'sunflow', 'cook', 'requir', 'tast', 'pinch', 'teaspoon', 'tablespoon', 'cup',
    'black', 'pepper', 'white', 'peppercorn', 'turmer', 'chilli', 'powder', 'cumin', 'seed', 'coriand',
    'mustard', 'ghee', 'butter', 'asafoetida', 'sugar', 'hing', 'garam', 'masala', 'curri', 'leav',
    'methi', 'fennel', 'clove', 'cinnamon', 'cardamom', 'ginger', 'garlic', 'fine', 'chopped', 'chop', 'peel',
    'sprig', 'dhania', 'jeera', 'mirchi', 'haldi', 'lemon', 'lime', 'flour', 'wheat', 'gram', 'besan',
    'maida', 'cornflour', 'fry', 'deep', 'roast', 'ground', 'past', 'slice', 'slit', 'fresh', 'dri',
    'baking', 'soda', 'powder', 'vinegar', 'warm', 'cold', 'hot', 'vanilla', 'extract', 'pinch',
    'whole', 'red', 'green', 'purpos', 'knead', 'roughli', 'need', 'piec', 'cut', 'wash', 'crush',
    'soak', 'drain', 'warm', 'cool', 'boil', 'small', 'medium', 'larg', 'halv', 'stick', 'grate',
    'mix', 'heat', 'pan', 'pot', 'bowl', 'spoon', 'serv', 'dish', 'prepar', 'add', 'use'
}

# Major Core Food Stems that CANNOT be assumed (must be present in user search if recipe needs them)
CORE_FOOD_STEMS = {
    # Meats, Poultry, Seafood
    'chicken', 'mutton', 'lamb', 'beef', 'pork', 'steak', 'bacon', 'ham', 'sausage', 'meatbal', 'hamburg',
    'meat', 'loaf', 'fish', 'prawn', 'shrimp', 'crab', 'salmon', 'tuna', 'lobster', 'turkey', 'duck', 'sirloin',
    # Dairy Mains
    'paneer', 'chees', 'cream', 'curd', 'dahi', 'yogurt', 'khoya', 'mava', 'tofu', 'milk', 'mayonnais',
    # Vegetables
    'potato', 'aloo', 'tomato', 'onion', 'spinach', 'palak', 'cabbag', 'cauliflow', 'gobi', 'pea', 'matar',
    'carrot', 'bean', 'mushroom', 'broccoli', 'brinjal', 'eggplant', 'baingan', 'okra', 'bhindi',
    'capsicum', 'bell', 'pumpkin', 'squash', 'corn', 'zucchini', 'cucumber', 'cucumb', 'karela', 'beetroot',
    'avocado', 'celeri', 'radish', 'mooli', 'lettuce', 'artichok', 'asparagus', 'mango',
    # Grains, Carbs, Noodles
    'rice', 'chawal', 'pasta', 'noodl', 'macaroni', 'bread', 'oat', 'poha', 'vermicelli', 'semiya', 'suji', 'rava',
    # Dals & Legumes
    'dal', 'lentil', 'rajma', 'chana', 'chickpea', 'urad', 'moong', 'toor', 'masoor', 'soyabean',
    # Eggs
    'egg'
}

class ModelCache:
    """
    In-memory singleton cache for Harna's pre-computed ML models,
    TF-IDF matrices, inverted ingredient indices, and recipe database.
    """
    _instance = None

    def __init__(self):
        self.vectorizer_name = None
        self.vectorizer_ing = None
        self.tfidf_combined = None
        self.recipe_df = None
        self.inverted_index = {}
        self.recipe_stems_map = []
        self.recipe_title_core = []
        self.PANTRY_STAPLES = PANTRY_STAPLES
        self.CORE_FOOD_STEMS = CORE_FOOD_STEMS
        self._loaded = False
        self._load_error = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelCache()
        return cls._instance

    def _resolve_data_dir(self):
        """
        Resolve data/processed/ directory robustly across local dev,
        Railway, Render, and any other deployment environment.
        """
        candidates = []

        # 1. From env variable (set in Railway/Render dashboard)
        env_path = os.environ.get("HARNA_DATA_DIR")
        if env_path:
            candidates.append(os.path.join(env_path, "processed"))

        # 2. Relative to this file: utils/ → my_flask_app/ → project root → data/processed
        this_file = os.path.abspath(__file__)
        pkg_root   = os.path.dirname(os.path.dirname(this_file))   # my_flask_app/
        proj_root  = os.path.dirname(pkg_root)                      # project root
        candidates.append(os.path.join(proj_root, "data", "processed"))

        # 3. Relative to cwd (gunicorn / flask run)
        candidates.append(os.path.join(os.getcwd(), "data", "processed"))

        # 4. Absolute fallback for Railway (project cloned to /app)
        candidates.append("/app/data/processed")

        for path in candidates:
            if os.path.isdir(path):
                pkls = [f for f in os.listdir(path) if f.endswith(".pkl")]
                if pkls:
                    print(f"[Harna Cache] Found {len(pkls)} pkl files in: {path}")
                    return path

        raise FileNotFoundError(
            f"[Harna Cache] Could not find data/processed/ with pkl files. Tried:\n"
            + "\n".join(f"  - {p}" for p in candidates)
        )

    def load_models(self, data_dir=None):
        if self._loaded:
            return

        try:
            processed_dir = data_dir or self._resolve_data_dir()
            if processed_dir and not os.path.exists(os.path.join(processed_dir, "Recipes.pkl")):
                sub = os.path.join(processed_dir, "processed")
                if os.path.exists(os.path.join(sub, "Recipes.pkl")):
                    processed_dir = sub

            with open(os.path.join(processed_dir, "Vectorizer_names.pkl"), 'rb') as f:
                self.vectorizer_name = pickle.load(f)

            with open(os.path.join(processed_dir, "Vectorizer_ingredients.pkl"), 'rb') as f:
                self.vectorizer_ing = pickle.load(f)

            with open(os.path.join(processed_dir, "TFidf_matrix.pkl"), 'rb') as f:
                self.tfidf_combined = pickle.load(f)

            with open(os.path.join(processed_dir, "Recipes.pkl"), 'rb') as f:
                self.recipe_df = pickle.load(f)

            # Pre-compute ingredient-only TF-IDF matrix for ultra-fast pantry matching (<10ms)
            try:
                self.tfidf_ingredients = self.vectorizer_ing.transform(
                    self.recipe_df['Ingredients_clean'].fillna('')
                )
            except Exception as ex:
                self.tfidf_ingredients = None
                print(f"[Harna Cache] Note: Deferred tfidf_ingredients transform: {ex}")

            # Build Inverted Index & Core Stems map for strict available-ingredients matching
            try:
                inv = {}
                stems_map = []
                title_core_map = []
                for idx, row in self.recipe_df.iterrows():
                    ing_clean = str(row.get('Ingredients_clean', ''))
                    stems = set(ing_clean.split())
                    stems_map.append(stems)
                    for s in stems:
                        if s not in inv:
                            inv[s] = set()
                        inv[s].add(idx)

                    title_words = re.findall(r'[a-zA-Z]+', str(row.get('RecipeName', '')).lower())
                    t_core = {_stemmer.stem(w) for w in title_words}.intersection(self.CORE_FOOD_STEMS)
                    title_core_map.append(t_core)

                self.inverted_index = inv
                self.recipe_stems_map = stems_map
                self.recipe_title_core = title_core_map
                print(f"[Harna Cache] Inverted index built for {len(inv)} ingredient stems across {len(self.recipe_df)} recipes.")
            except Exception as ex:
                print(f"[Harna Cache] Note: Failed to build inverted index: {ex}")

            self._loaded = True
            self._load_error = None
            print("[Harna Backend] ML Models & Recipe Dataframe successfully cached into memory.")

        except Exception as e:
            self._load_error = str(e)
            print(f"[Harna Backend] CRITICAL: Failed to load ML models — {e}")
            raise

    def get_recipe_by_id(self, recipe_id):
        if not self._loaded:
            self.load_models()
        if not isinstance(recipe_id, (int, float)) or isinstance(recipe_id, bool):
            return None
        try:
            val = float(recipe_id)
            if not val.is_integer() or pd.isna(val) or val <= 0:
                return None
            int_id = int(val)
        except (ValueError, TypeError, OverflowError):
            return None

        matches = self.recipe_df[self.recipe_df['Srno'] == int_id]
        if matches.empty:
            matches = self.recipe_df[self.recipe_df['Srno'] == float(int_id)]
        if matches.empty:
            return None
        return matches.iloc[0].to_dict()

# Convenience accessor
cache = ModelCache.get_instance()
