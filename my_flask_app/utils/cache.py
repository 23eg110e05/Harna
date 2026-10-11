import os
import pickle
import pandas as pd

class ModelCache:
    """
    In-memory singleton cache for Harna's pre-computed ML models,
    TF-IDF matrices, and recipe database to achieve <15ms recommendation latency.
    """
    _instance = None

    def __init__(self):
        self.vectorizer_name = None
        self.vectorizer_ing = None
        self.tfidf_combined = None
        self.recipe_df = None
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

            with open(os.path.join(processed_dir, "Vectorizer_names.pkl"), 'rb') as f:
                self.vectorizer_name = pickle.load(f)

            with open(os.path.join(processed_dir, "Vectorizer_ingredients.pkl"), 'rb') as f:
                self.vectorizer_ing = pickle.load(f)

            with open(os.path.join(processed_dir, "TFidf_matrix.pkl"), 'rb') as f:
                self.tfidf_combined = pickle.load(f)

            with open(os.path.join(processed_dir, "Recipes.pkl"), 'rb') as f:
                self.recipe_df = pickle.load(f)

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
        matches = self.recipe_df[
            (self.recipe_df['Srno'] == recipe_id) |
            (self.recipe_df['Srno'] == float(recipe_id))
        ]
        if matches.empty:
            try:
                matches = self.recipe_df[
                    self.recipe_df['Srno'].astype(str).str.split('.').str[0] ==
                    str(recipe_id).split('.')[0]
                ]
            except Exception:
                pass
        if matches.empty:
            return None
        return matches.iloc[0].to_dict()

# Convenience accessor
cache = ModelCache.get_instance()
