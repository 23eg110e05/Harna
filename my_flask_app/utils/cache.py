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

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelCache()
        return cls._instance

    def load_models(self, data_dir=None):
        if self._loaded:
            return

        if data_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "data"))

        processed_dir = os.path.join(data_dir, "processed")

        with open(os.path.join(processed_dir, "Vectorizer_names.pkl"), 'rb') as f:
            self.vectorizer_name = pickle.load(f)

        with open(os.path.join(processed_dir, "Vectorizer_ingredients.pkl"), 'rb') as f:
            self.vectorizer_ing = pickle.load(f)

        with open(os.path.join(processed_dir, "TFidf_matrix.pkl"), 'rb') as f:
            self.tfidf_combined = pickle.load(f)

        with open(os.path.join(processed_dir, "Recipes.pkl"), 'rb') as f:
            self.recipe_df = pickle.load(f)

        self._loaded = True
        print("[Harna Backend] ML Models & Recipe Dataframe successfully cached into memory.")

    def get_recipe_by_id(self, recipe_id):
        if not self._loaded:
            self.load_models()
        matches = self.recipe_df[self.recipe_df['Srno'] == recipe_id]
        if matches.empty:
            return None
        return matches.iloc[0].to_dict()

# Convenience accessor
cache = ModelCache.get_instance()
