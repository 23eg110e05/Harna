import os
import time
import psutil
import pytest
import threading
import concurrent.futures
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity
from my_flask_app.utils.cache import ModelCache

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def get_fresh_cache():
    """Create a completely uninitialized ModelCache instance for testing."""
    cache = ModelCache()
    return cache


# =====================================================================
# 1. BOUNDARY ID & DATA DISTRIBUTION TESTS
# =====================================================================

class TestBoundaryAndDistribution:
    @classmethod
    def setup_class(cls):
        cls.cache = ModelCache.get_instance()
        cls.cache.load_models(DATA_DIR)
        cls.df = cls.cache.recipe_df

    def test_boundary_id_1_indian_lower(self):
        recipe = self.cache.get_recipe_by_id(1)
        assert recipe is not None, "Recipe ID 1 must exist"
        assert recipe["Srno"] == 1
        assert recipe["RecipeName"] == "Masala Karela Recipe"
        assert recipe["Cuisine"] != "Global / Fusion"
        assert "Ingredients" in recipe and len(str(recipe["Ingredients"])) > 0

    def test_boundary_id_14167_indian_upper(self):
        recipe = self.cache.get_recipe_by_id(14167)
        assert recipe is not None, "Recipe ID 14167 must exist"
        assert recipe["Srno"] == 14167
        assert recipe["Cuisine"] != "Global / Fusion"

    def test_boundary_id_14168_global_lower(self):
        recipe = self.cache.get_recipe_by_id(14168)
        assert recipe is not None, "Recipe ID 14168 must exist"
        assert recipe["Srno"] == 14168
        assert recipe["Cuisine"] == "Global / Fusion"
        assert recipe["Calories (kcal)"] == 320.0
        assert recipe["Calories (kcal).1"] == 80.0

    def test_boundary_id_34167_global_upper(self):
        recipe = self.cache.get_recipe_by_id(34167)
        assert recipe is not None, "Recipe ID 34167 must exist"
        assert recipe["Srno"] == 34167
        assert recipe["Cuisine"] == "Global / Fusion"
        assert recipe["Calories (kcal)"] == 320.0
        assert recipe["Protein (g).1"] == 3.5

    def test_boundary_id_0_out_of_bounds_lower(self):
        recipe = self.cache.get_recipe_by_id(0)
        assert recipe is None, "Recipe ID 0 must return None"

    def test_boundary_id_999999_out_of_bounds_upper(self):
        recipe = self.cache.get_recipe_by_id(999999)
        assert recipe is None, "Recipe ID 999999 must return None"

    @pytest.mark.parametrize("invalid_id", [-1, -999, 34168, 100000, "1", "abc", None, 1.5])
    def test_invalid_and_edge_ids_handled_safely(self, invalid_id):
        # Should gracefully return None without raising unhandled exceptions
        try:
            result = self.cache.get_recipe_by_id(invalid_id)
            assert result is None, f"Expected None for invalid ID {invalid_id}, got {result}"
        except Exception as e:
            pytest.fail(f"get_recipe_by_id({invalid_id}) raised unexpected exception: {e}")

    def test_distribution_consistency(self):
        indian_df = self.df[self.df["Cuisine"] != "Global / Fusion"]
        global_df = self.df[self.df["Cuisine"] == "Global / Fusion"]

        assert len(indian_df) == 5655, f"Expected 5,655 Indian recipes, got {len(indian_df)}"
        assert len(global_df) == 20000, f"Expected 20,000 Global recipes, got {len(global_df)}"
        assert len(self.df) == 25655, f"Expected 25,655 total recipes, got {len(self.df)}"

        # Verify Indian ID range
        assert indian_df["Srno"].min() == 1
        assert indian_df["Srno"].max() == 14167

        # Verify Global ID range
        assert global_df["Srno"].min() == 14168
        assert global_df["Srno"].max() == 34167

        # Strict disjointness
        overlap = set(indian_df["Srno"]).intersection(set(global_df["Srno"]))
        assert len(overlap) == 0, f"Detected colliding IDs between Indian and Global slices: {overlap}"


# =====================================================================
# 2. CONCURRENCY & RACE CONDITION STRESS TESTS
# =====================================================================

class TestConcurrency:
    @classmethod
    def setup_class(cls):
        cls.cache = ModelCache.get_instance()
        cls.cache.load_models(DATA_DIR)

    def test_concurrent_warm_lookups_100_threads(self):
        """100 threads concurrently executing 50 lookups each (5,000 total operations)."""
        valid_ids = [1, 14167, 14168, 34167, 20000]
        invalid_ids = [0, 999999, 5000]  # 5000 is an unmapped ID gap in the Indian dataset
        target_ids = valid_ids + invalid_ids
        results = []
        errors = []

        def worker(thread_idx):
            for i in range(50):
                tid = target_ids[(thread_idx + i) % len(target_ids)]
                try:
                    res = self.cache.get_recipe_by_id(tid)
                    if tid in invalid_ids:
                        assert res is None, f"Expected None for invalid ID {tid}"
                    else:
                        assert res is not None, f"Expected recipe for valid ID {tid}"
                        assert res["Srno"] == tid
                    results.append(1)
                except Exception as e:
                    errors.append((thread_idx, tid, str(e)))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Encountered concurrency errors: {errors[:5]}"
        assert len(results) == 5000, f"Expected 5000 completed lookups, got {len(results)}"

    def test_cold_start_concurrent_load_race_condition(self):
        """Stress-test uninitialized ModelCache accessed simultaneously by 20 threads."""
        fresh_cache = ModelCache()
        assert not fresh_cache._loaded

        errors = []
        retrieved_recipes = []

        def cold_worker(thread_idx):
            # Each worker attempts to get a recipe immediately when cache is not loaded
            recipe_id = 1 if thread_idx % 2 == 0 else 14168
            try:
                rec = fresh_cache.get_recipe_by_id(recipe_id)
                assert rec is not None
                assert rec["Srno"] == recipe_id
                retrieved_recipes.append(rec["Srno"])
            except Exception as e:
                errors.append((thread_idx, str(e)))

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(cold_worker, i) for i in range(20)]
            concurrent.futures.wait(futures)

        assert len(errors) == 0, f"Cold start race condition triggered exceptions: {errors}"
        assert len(retrieved_recipes) == 20
        assert fresh_cache._loaded
        assert fresh_cache.recipe_df is not None

    def test_concurrent_recommendation_computations(self):
        """20 concurrent threads running vectorization and cosine similarity."""
        queries = ["paneer, butter", "beef, mushroom", "chicken, egg", "pasta, cheese"]
        errors = []
        outputs = []

        def rec_worker(thread_idx):
            q = queries[thread_idx % len(queries)]
            try:
                m = self.cache.tfidf_combined
                vn = self.cache.vectorizer_name
                vi = self.cache.vectorizer_ing
                df = self.cache.recipe_df

                uv = hstack([0.5 * vn.transform([q]), 0.5 * vi.transform([q])])
                sim = cosine_similarity(uv, m).flatten()
                top_idx = sim.argmax()
                top_recipe = df.iloc[top_idx]
                outputs.append((q, top_recipe["RecipeName"], sim[top_idx]))
            except Exception as e:
                errors.append((thread_idx, str(e)))

        threads = [threading.Thread(target=rec_worker, args=(i,)) for i in range(40)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent recommendation queries failed: {errors}"
        assert len(outputs) == 40


# =====================================================================
# 3. MEMORY STABILITY & COLD START BENCHMARK TESTS
# =====================================================================

class TestMemoryAndLatency:
    def test_cold_start_load_latency_and_memory(self):
        process = psutil.Process()
        mem_before_mb = process.memory_info().rss / (1024 * 1024)

        t0 = time.perf_counter()
        fresh_cache = ModelCache()
        fresh_cache.load_models(DATA_DIR)
        t_load = time.perf_counter() - t0

        mem_after_mb = process.memory_info().rss / (1024 * 1024)
        mem_delta_mb = mem_after_mb - mem_before_mb

        print(f"\n[Cold Start Benchmark] Load time: {t_load:.3f}s | Delta RSS: {mem_delta_mb:.1f} MB")
        # Cold start must complete in reasonable time (< 15s) and memory delta < 500 MB
        assert t_load < 15.0, f"Cold start too slow: {t_load:.2f}s"
        assert fresh_cache._loaded

    def test_lookup_latency_benchmark(self):
        cache = ModelCache.get_instance()
        cache.load_models(DATA_DIR)

        # Warmup
        cache.get_recipe_by_id(1)

        # Benchmark 500 lookups
        t0 = time.perf_counter()
        for i in range(500):
            tid = (i % 25655) + 1
            cache.get_recipe_by_id(tid)
        t_total = time.perf_counter() - t0
        avg_ms = (t_total / 500) * 1000

        print(f"\n[Lookup Benchmark] 500 lookups in {t_total:.3f}s ({avg_ms:.3f} ms/lookup)")
        # Average lookup time should be under 5ms
        assert avg_ms < 5.0, f"Single lookup too slow: {avg_ms:.2f} ms"

    def test_memory_stability_under_repeated_queries(self):
        cache = ModelCache.get_instance()
        cache.load_models(DATA_DIR)
        process = psutil.Process()

        mem_initial_mb = process.memory_info().rss / (1024 * 1024)

        # Perform 2,000 lookups and 50 recommendations
        for i in range(2000):
            cache.get_recipe_by_id((i % 34167) + 1)

        vn = cache.vectorizer_name
        vi = cache.vectorizer_ing
        m = cache.tfidf_combined
        for i in range(50):
            uv = hstack([0.5 * vn.transform(["paneer butter chicken pasta"]), 0.5 * vi.transform(["garlic ginger onion"])])
            _ = cosine_similarity(uv, m).flatten()

        mem_final_mb = process.memory_info().rss / (1024 * 1024)
        mem_growth_mb = mem_final_mb - mem_initial_mb

        print(f"\n[Memory Stability] Initial: {mem_initial_mb:.1f} MB | Final: {mem_final_mb:.1f} MB | Growth: {mem_growth_mb:.1f} MB")
        # Growth should be minimal (< 60MB across lookups and calculations)
        assert mem_growth_mb < 60.0, f"Potential memory leak detected: {mem_growth_mb:.1f} MB growth"
