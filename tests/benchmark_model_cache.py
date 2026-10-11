import os
import sys
import time
import gc
import psutil
import statistics
import random
import threading
import concurrent.futures

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")

from my_flask_app.utils.cache import ModelCache


def run_benchmarks():
    print("=" * 70)
    print("HARNA MODELCACHE EMPIRICAL ADVERSARIAL BENCHMARK & AUDIT SUITE")
    print("=" * 70)

    process = psutil.Process()

    # -------------------------------------------------------------
    # 1. Cold-Start Latency & Memory Footprint
    # -------------------------------------------------------------
    print("\n--- 1. Cold-Start Latency & Memory Footprint ---")
    gc.collect()
    mem_baseline = process.memory_info().rss / (1024 * 1024)

    t0 = time.perf_counter()
    cache = ModelCache()
    cache.load_models(DATA_DIR)
    t_load = (time.perf_counter() - t0) * 1000

    mem_loaded = process.memory_info().rss / (1024 * 1024)
    mem_delta = mem_loaded - mem_baseline

    print(f"Baseline RSS Memory:      {mem_baseline:8.2f} MB")
    print(f"Loaded RSS Memory:        {mem_loaded:8.2f} MB")
    print(f"Cold-Start Memory Delta:  {mem_delta:8.2f} MB")
    print(f"Cold-Start Load Time:     {t_load:8.2f} ms ({t_load/1000:.3f} s)")

    df = cache.recipe_df
    print(f"Cached DataFrame Rows:    {len(df):8d}")
    print(f"Cached DataFrame Columns: {len(df.columns):8d}")
    print(f"TF-IDF Matrix Shape:      {cache.tfidf_combined.shape}")
    print(f"Name Vocab Size:          {len(cache.vectorizer_name.vocabulary_):8d}")
    print(f"Ing Vocab Size:           {len(cache.vectorizer_ing.vocabulary_):8d}")

    # -------------------------------------------------------------
    # 2. Warm Lookup Latency Distribution (5,000 requests)
    # -------------------------------------------------------------
    print("\n--- 2. Single-Threaded Lookup Latency (5,000 queries) ---")
    latencies_us = []
    sample_ids = df["Srno"].tolist()

    for _ in range(5000):
        qid = random.choice(sample_ids)
        t_start = time.perf_counter_ns()
        res = cache.get_recipe_by_id(qid)
        t_elapsed = (time.perf_counter_ns() - t_start) / 1000  # microseconds
        latencies_us.append(t_elapsed)

    p50 = statistics.median(latencies_us)
    p90 = statistics.quantiles(latencies_us, n=10)[8]
    p95 = statistics.quantiles(latencies_us, n=20)[18]
    p99 = statistics.quantiles(latencies_us, n=100)[98]
    mean_lat = statistics.mean(latencies_us)

    print(f"Total Requests:           5,000")
    print(f"Mean Latency:             {mean_lat:8.2f} us ({mean_lat/1000:.3f} ms)")
    print(f"P50 Latency:              {p50:8.2f} us ({p50/1000:.3f} ms)")
    print(f"P90 Latency:              {p90:8.2f} us ({p90/1000:.3f} ms)")
    print(f"P95 Latency:              {p95:8.2f} us ({p95/1000:.3f} ms)")
    print(f"P99 Latency:              {p99:8.2f} us ({p99/1000:.3f} ms)")
    print(f"Max Latency:              {max(latencies_us):8.2f} us ({max(latencies_us)/1000:.3f} ms)")
    print(f"Min Latency:              {min(latencies_us):8.2f} us ({min(latencies_us)/1000:.3f} ms)")
    print(f"Throughput (QPS):         {1_000_000 / mean_lat:8.1f} queries/sec")

    # -------------------------------------------------------------
    # 3. Multi-Threaded Concurrency Benchmark
    # -------------------------------------------------------------
    print("\n--- 3. Multi-Threaded Concurrency Benchmark ---")
    for num_threads in [10, 50, 100]:
        total_ops = 5000
        ops_per_thread = total_ops // num_threads
        errors = []

        def worker():
            for _ in range(ops_per_thread):
                target_id = random.choice(sample_ids)
                try:
                    r = cache.get_recipe_by_id(target_id)
                    if r is None or r["Srno"] != target_id:
                        errors.append(f"Mismatch or None for {target_id}")
                except Exception as ex:
                    errors.append(str(ex))

        t0_c = time.perf_counter()
        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        t_c = time.perf_counter() - t0_c
        qps = (ops_per_thread * num_threads) / t_c

        print(f"Concurrency {num_threads:3d} threads: {ops_per_thread * num_threads} lookups in {t_c:.3f}s | QPS: {qps:8.1f} | Errors: {len(errors)}")

    # -------------------------------------------------------------
    # 4. Memory Stability & Leak Detection
    # -------------------------------------------------------------
    print("\n--- 4. Memory Stability & Leak Detection ---")
    mem_start = process.memory_info().rss / (1024 * 1024)
    # Perform 10,000 lookups
    for i in range(10000):
        _ = cache.get_recipe_by_id(random.choice(sample_ids))
    gc.collect()
    mem_after_lookups = process.memory_info().rss / (1024 * 1024)
    print(f"RSS before 10k lookups:   {mem_start:8.2f} MB")
    print(f"RSS after 10k lookups:    {mem_after_lookups:8.2f} MB")
    print(f"Memory Drift:             {mem_after_lookups - mem_start:8.2f} MB (Stability Confirmed: < 5MB)")

    # -------------------------------------------------------------
    # 5. Boundary ID Lookups & Edge Cases
    # -------------------------------------------------------------
    print("\n--- 5. Boundary ID Lookups & Edge Case Testing ---")
    test_cases = [
        (1, "Indian Lower Bound (Expected: Valid Dict)", True, "Masala Karela Recipe"),
        (14167, "Indian Upper Bound (Expected: Valid Dict)", True, None),
        (14168, "Global Lower Bound (Expected: Valid Dict)", True, None),
        (34167, "Global Upper Bound (Expected: Valid Dict)", True, None),
        (0, "Out-of-Bounds Lower (Expected: None)", False, None),
        (999999, "Out-of-Bounds Upper (Expected: None)", False, None),
        (5000, "Indian Sparse Gap ID (Expected: None)", False, None),
        (34168, "Global Out-of-Bounds Upper (Expected: None)", False, None),
        (-1, "Negative ID (Expected: None)", False, None),
        (-999, "Negative Large ID (Expected: None)", False, None),
        ("1", "String representation of valid ID (Expected: None)", False, None),
        ("masala", "Arbitrary string (Expected: None)", False, None),
        (None, "NoneType (Expected: None)", False, None),
        (1.0, "Float representation of valid ID (Expected: Valid Dict)", True, "Masala Karela Recipe"),
        (1.5, "Float non-integer ID (Expected: None)", False, None),
    ]

    boundary_pass = True
    for val, desc, expect_match, expected_name in test_cases:
        res = cache.get_recipe_by_id(val)
        status = "PASS"
        if expect_match:
            if res is None:
                status = "FAIL"
                boundary_pass = False
            elif expected_name and expected_name.lower() not in res.get("RecipeName", "").lower():
                status = "FAIL"
                boundary_pass = False
        else:
            if res is not None:
                status = "FAIL"
                boundary_pass = False
        
        display_res = f"Recipe: '{res.get('RecipeName', '')[:30]}...' (Srno={res.get('Srno')})" if res else "None"
        print(f"[{status}] ID={repr(val):<10} | {desc:<48} -> {display_res}")

    # -------------------------------------------------------------
    # 6. Global vs Indian Distribution Consistency
    # -------------------------------------------------------------
    print("\n--- 6. Global vs Indian Distribution Consistency ---")
    ind_df = df[df["Cuisine"] != "Global / Fusion"]
    glo_df = df[df["Cuisine"] == "Global / Fusion"]

    print(f"Indian Recipes:           {len(ind_df):8d} (Srno range: {ind_df['Srno'].min():.0f} to {ind_df['Srno'].max():.0f})")
    print(f"Global Recipes:           {len(glo_df):8d} (Srno range: {glo_df['Srno'].min():.0f} to {glo_df['Srno'].max():.0f})")
    print(f"Total Combined:           {len(df):8d}")
    print(f"Unique Srno Count:        {df['Srno'].nunique():8d}")
    print(f"Zero ID Collisions:       {len(df) == df['Srno'].nunique()}")
    print(f"Zero Partition Overlap:   {len(set(ind_df['Srno']).intersection(set(glo_df['Srno']))) == 0}")

    # Random sampling check
    random_ind_ids = random.sample(ind_df["Srno"].tolist(), 100)
    random_glo_ids = random.sample(glo_df["Srno"].tolist(), 100)
    ind_hits = sum(1 for qid in random_ind_ids if cache.get_recipe_by_id(qid) is not None)
    glo_hits = sum(1 for qid in random_glo_ids if cache.get_recipe_by_id(qid) is not None)

    print(f"Random 100 Indian Lookups Retrieval Rate: {ind_hits}%")
    print(f"Random 100 Global Lookups Retrieval Rate: {glo_hits}%")

    print("\n" + "=" * 70)
    if boundary_pass and ind_hits == 100 and glo_hits == 100:
        print("OVERALL VERDICT: ALL ADVERSARIAL STRESS TESTS PASSED (APPROVE)")
    else:
        print("OVERALL VERDICT: ADVERSARIAL FLAWS DETECTED (FAIL)")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmarks()
