"""
LLM Memory Framework Benchmark Suite
Produces quantitative metrics for comparison with other memory systems.

Metrics measured:
- Latency (p50, p95, p99)
- Throughput (ops/sec)
- Memory capacity limits
- Embedding quality
- Context building efficiency
- LLM management overhead
"""

import os
import sys
import gc
import json
import time
import shutil
import tempfile
import statistics
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from memory import MemoryManager
from memory.managers.policies import ShortTermPolicy
from memory.stores.short_term import ShortTermMemory
from memory.stores.long_term import LongTermMemory
from memory.stores.feature_memory import FeatureMemory
from memory.utils.helpers import count_tokens


def get_llm_client():
    """Get LLM client if available."""
    try:
        from server.local_client import LocalLLMClient
        from config import MODEL_SERVER_URL
        client = LocalLLMClient(MODEL_SERVER_URL)
        client.embed("test")  # Verify connection
        return client
    except Exception as e:
        print(f"[ERROR] LLM not available: {e}")
        return None


def measure_latency(func, iterations: int = 100) -> Dict[str, float]:
    """Measure latency statistics for a function."""
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        latencies.append((time.perf_counter() - start) * 1000)  # ms
    
    latencies.sort()
    return {
        "min_ms": round(min(latencies), 3),
        "max_ms": round(max(latencies), 3),
        "mean_ms": round(statistics.mean(latencies), 3),
        "p50_ms": round(latencies[len(latencies) // 2], 3),
        "p95_ms": round(latencies[int(len(latencies) * 0.95)], 3),
        "p99_ms": round(latencies[int(len(latencies) * 0.99)], 3),
        "iterations": iterations
    }


def benchmark_short_term_write(iterations: int = 1000) -> Dict:
    """Benchmark short-term memory write operations."""
    print(f"\n[BENCHMARK] Short-Term Memory Write ({iterations} ops)")
    
    stm = ShortTermMemory(
        scope="user:bench",
        policy=ShortTermPolicy(max_tokens=1000000, max_entries=100000)
    )
    
    # Warmup
    for i in range(10):
        stm.add(f"warmup {i}", "user")
    stm.clear()
    
    # Benchmark
    start = time.perf_counter()
    for i in range(iterations):
        stm.add(f"Benchmark message number {i} with some content", "user")
    total_time = time.perf_counter() - start
    
    ops_per_sec = iterations / total_time
    
    # Latency measurement
    def write_op():
        stm.add("test message", "user")
    
    latency = measure_latency(write_op, min(100, iterations))
    
    return {
        "operation": "short_term_write",
        "total_ops": iterations,
        "total_time_s": round(total_time, 4),
        "ops_per_sec": round(ops_per_sec, 1),
        "latency": latency
    }


def benchmark_short_term_read(iterations: int = 1000) -> Dict:
    """Benchmark short-term memory read operations."""
    print(f"\n[BENCHMARK] Short-Term Memory Read ({iterations} ops)")
    
    stm = ShortTermMemory(
        scope="user:bench",
        policy=ShortTermPolicy(max_tokens=100000, max_entries=10000)
    )
    
    # Populate
    for i in range(100):
        stm.add(f"Message {i} with content", "user")
    
    # Benchmark
    start = time.perf_counter()
    for i in range(iterations):
        stm.get_context(max_tokens=500)
    total_time = time.perf_counter() - start
    
    ops_per_sec = iterations / max(total_time, 0.0001)
    
    # Latency
    def read_op():
        stm.get_context(max_tokens=500)
    
    latency = measure_latency(read_op, min(100, iterations))
    
    return {
        "operation": "short_term_read",
        "total_ops": iterations,
        "total_time_s": round(total_time, 4),
        "ops_per_sec": round(ops_per_sec, 1),
        "latency": latency
    }


def benchmark_feature_memory(iterations: int = 500) -> Dict:
    """Benchmark feature memory operations."""
    print(f"\n[BENCHMARK] Feature Memory ({iterations} ops)")
    
    temp_dir = tempfile.mkdtemp()
    try:
        fm = FeatureMemory(scope="user:bench", storage_dir=temp_dir)
        
        # Write benchmark
        start = time.perf_counter()
        for i in range(iterations):
            fm.set(f"key_{i}", f"value_{i}")
        write_time = time.perf_counter() - start
        
        # Read benchmark
        start = time.perf_counter()
        for i in range(iterations):
            fm.get(f"key_{i % 100}")
        read_time = time.perf_counter() - start
        
        # History benchmark
        for i in range(100):
            fm.set("changing", i)
        
        start = time.perf_counter()
        for i in range(iterations):
            fm.get_with_history("changing")
        history_time = time.perf_counter() - start
        
        return {
            "operation": "feature_memory",
            "total_ops": iterations,
            "write_ops_per_sec": round(iterations / max(write_time, 0.0001), 1),
            "read_ops_per_sec": round(iterations / max(read_time, 0.0001), 1),
            "history_ops_per_sec": round(iterations / max(history_time, 0.0001), 1),
            "write_time_s": round(write_time, 4),
            "read_time_s": round(read_time, 4)
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def benchmark_embedding_latency(client, iterations: int = 50) -> Dict:
    """Benchmark embedding generation latency."""
    print(f"\n[BENCHMARK] Embedding Latency ({iterations} ops)")
    
    test_texts = [
        "Short text",
        "Medium length text with more words and content",
        "A longer text passage that contains multiple sentences and covers various topics to test embedding performance with different input sizes.",
    ]
    
    results = {}
    for i, text in enumerate(test_texts):
        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            client.embed(text)
            latencies.append((time.perf_counter() - start) * 1000)
        
        latencies.sort()
        results[f"text_{len(text)}_chars"] = {
            "mean_ms": round(statistics.mean(latencies), 2),
            "p50_ms": round(latencies[len(latencies) // 2], 2),
            "p95_ms": round(latencies[int(len(latencies) * 0.95)], 2),
        }
    
    return {
        "operation": "embedding",
        "iterations": iterations,
        "by_text_length": results
    }


def benchmark_semantic_search(client, iterations: int = 50) -> Dict:
    """Benchmark semantic search latency."""
    print(f"\n[BENCHMARK] Semantic Search ({iterations} ops)")
    
    temp_dir = tempfile.mkdtemp()
    try:
        ltm = LongTermMemory(
            scope="user:bench",
            qdrant_path=os.path.join(temp_dir, "qdrant"),
            llm_client=client
        )
        
        # Populate with varied content
        topics = ["Python programming", "Machine learning", "Web development",
                  "Database design", "API development", "Testing strategies",
                  "Cloud computing", "DevOps practices", "Security best practices",
                  "Performance optimization"]
        
        for i, topic in enumerate(topics):
            ltm.store_conversation(f"Tell me about {topic}", f"Here's information about {topic}...")
        
        # Benchmark search
        queries = ["how to code", "database", "testing", "security", "performance"]
        latencies = []
        
        for _ in range(iterations):
            query = queries[_ % len(queries)]
            start = time.perf_counter()
            ltm.search(query, limit=5)
            latencies.append((time.perf_counter() - start) * 1000)
        
        latencies.sort()
        
        return {
            "operation": "semantic_search",
            "corpus_size": len(topics),
            "iterations": iterations,
            "mean_ms": round(statistics.mean(latencies), 2),
            "p50_ms": round(latencies[len(latencies) // 2], 2),
            "p95_ms": round(latencies[int(len(latencies) * 0.95)], 2),
            "p99_ms": round(latencies[int(len(latencies) * 0.99)], 2),
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def benchmark_context_building(client, iterations: int = 50) -> Dict:
    """Benchmark full context building with all memory types."""
    print(f"\n[BENCHMARK] Context Building ({iterations} ops)")
    
    temp_dir = tempfile.mkdtemp()
    try:
        mm = MemoryManager(
            scope="user:bench",
            llm_client=client,
            data_dir=temp_dir,
            use_llm_management=False
        )
        
        # Populate memories
        for i in range(20):
            mm.add_turn(f"Question {i}", f"Answer {i}", extract_facts=False)
        for i in range(10):
            mm.set_fact(f"fact_{i}", f"value_{i}")
        
        # Benchmark context building at different token limits
        results = {}
        for max_tokens in [500, 1000, 2000, 4000]:
            latencies = []
            for _ in range(iterations):
                start = time.perf_counter()
                mm.get_context("test query", max_tokens=max_tokens)
                latencies.append((time.perf_counter() - start) * 1000)
            
            latencies.sort()
            results[f"max_{max_tokens}_tokens"] = {
                "mean_ms": round(statistics.mean(latencies), 2),
                "p50_ms": round(latencies[len(latencies) // 2], 2),
                "p95_ms": round(latencies[int(len(latencies) * 0.95)], 2),
            }
        
        return {
            "operation": "context_building",
            "iterations": iterations,
            "by_token_limit": results
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def benchmark_llm_management(client, iterations: int = 10) -> Dict:
    """Benchmark LLM-driven memory management."""
    print(f"\n[BENCHMARK] LLM Memory Management ({iterations} ops)")
    
    from memory.managers.llm_manager import LLMMemoryManager
    
    manager = LLMMemoryManager(client)
    
    # Create test entries
    entries = [
        {"id": f"entry_{i}", "role": "user", "content": f"Message {i} content",
         "tokens": 10, "age_minutes": i * 5, "access_count": 1}
        for i in range(20)
    ]
    
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        manager.get_management_actions(entries, total_tokens=200, max_tokens=500)
        latencies.append((time.perf_counter() - start) * 1000)
    
    latencies.sort()
    
    return {
        "operation": "llm_management",
        "entries_per_call": len(entries),
        "iterations": iterations,
        "mean_ms": round(statistics.mean(latencies), 2),
        "p50_ms": round(latencies[len(latencies) // 2], 2),
        "p95_ms": round(latencies[int(len(latencies) * 0.95)], 2),
    }


def benchmark_capacity_limits() -> Dict:
    """Test capacity limits of memory systems."""
    print("\n[BENCHMARK] Capacity Limits")
    
    results = {}
    
    # Short-term capacity
    stm = ShortTermMemory(
        scope="user:capacity",
        policy=ShortTermPolicy(max_tokens=10000000, max_entries=1000000)
    )
    
    start = time.perf_counter()
    for i in range(10000):
        stm.add(f"Message {i}", "user")
        if i % 1000 == 0:
            print(f"  Short-term: {i} entries...")
    stm_time = time.perf_counter() - start
    
    results["short_term"] = {
        "entries": len(stm),
        "tokens": stm.total_tokens,
        "time_s": round(stm_time, 2),
        "entries_per_sec": round(10000 / stm_time, 1)
    }
    
    # Feature memory capacity
    temp_dir = tempfile.mkdtemp()
    try:
        fm = FeatureMemory(scope="user:capacity", storage_dir=temp_dir)
        
        start = time.perf_counter()
        for i in range(1000):
            fm.set(f"fact_{i}", f"value_{i}")
        fm_time = time.perf_counter() - start
        
        # History depth
        for i in range(500):
            fm.set("deep_history", i)
        fact = fm.get_with_history("deep_history")
        
        results["feature_memory"] = {
            "facts": len(fm),
            "max_history_depth": len(fact.history),
            "time_s": round(fm_time, 2),
            "facts_per_sec": round(1000 / max(fm_time, 0.0001), 1)
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return {
        "operation": "capacity_limits",
        "results": results
    }


def benchmark_end_to_end(client, iterations: int = 20) -> Dict:
    """Benchmark complete end-to-end conversation flow."""
    print(f"\n[BENCHMARK] End-to-End Flow ({iterations} ops)")
    
    temp_dir = tempfile.mkdtemp()
    try:
        mm = MemoryManager(
            scope="user:e2e",
            llm_client=client,
            data_dir=temp_dir,
            use_llm_management=True
        )
        
        latencies = []
        for i in range(iterations):
            start = time.perf_counter()
            
            # Simulate full conversation turn
            mm.add_turn(
                f"User question {i} about programming",
                f"Here's a helpful response {i} about programming",
                extract_facts=True
            )
            context = mm.get_context(f"follow up {i}")
            
            latencies.append((time.perf_counter() - start) * 1000)
        
        latencies.sort()
        
        return {
            "operation": "end_to_end_turn",
            "iterations": iterations,
            "includes": ["add_turn", "fact_extraction", "context_building"],
            "mean_ms": round(statistics.mean(latencies), 2),
            "p50_ms": round(latencies[len(latencies) // 2], 2),
            "p95_ms": round(latencies[int(len(latencies) * 0.95)], 2),
            "p99_ms": round(latencies[int(len(latencies) * 0.99)], 2),
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_all_benchmarks(client) -> Dict:
    """Run all benchmarks and return results."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
        },
        "benchmarks": {}
    }
    
    benchmarks = [
        ("short_term_write", lambda: benchmark_short_term_write(1000)),
        ("short_term_read", lambda: benchmark_short_term_read(1000)),
        ("feature_memory", lambda: benchmark_feature_memory(500)),
        ("embedding_latency", lambda: benchmark_embedding_latency(client, 30)),
        ("semantic_search", lambda: benchmark_semantic_search(client, 30)),
        ("context_building", lambda: benchmark_context_building(client, 30)),
        ("llm_management", lambda: benchmark_llm_management(client, 5)),
        ("capacity_limits", benchmark_capacity_limits),
        ("end_to_end", lambda: benchmark_end_to_end(client, 15)),
    ]
    
    for name, func in benchmarks:
        try:
            results["benchmarks"][name] = func()
            print(f"  -> {name}: OK")
        except Exception as e:
            print(f"  -> {name}: FAILED ({e})")
            results["benchmarks"][name] = {"error": str(e)}
    
    return results


def print_summary(results: Dict):
    """Print human-readable summary."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    
    b = results["benchmarks"]
    
    print("\n### THROUGHPUT (ops/sec)")
    print("-" * 50)
    if "short_term_write" in b and "ops_per_sec" in b["short_term_write"]:
        print(f"  Short-term write:     {b['short_term_write']['ops_per_sec']:>12,} ops/sec")
    if "short_term_read" in b and "ops_per_sec" in b["short_term_read"]:
        print(f"  Short-term read:      {b['short_term_read']['ops_per_sec']:>12,} ops/sec")
    if "feature_memory" in b and "write_ops_per_sec" in b["feature_memory"]:
        print(f"  Feature memory write: {b['feature_memory']['write_ops_per_sec']:>12,} ops/sec")
        print(f"  Feature memory read:  {b['feature_memory']['read_ops_per_sec']:>12,} ops/sec")
    
    print("\n### LATENCY (milliseconds)")
    print("-" * 50)
    print(f"{'Operation':<25} {'p50':>10} {'p95':>10} {'p99':>10}")
    print("-" * 50)
    
    if "short_term_write" in b and "latency" in b["short_term_write"]:
        lat = b["short_term_write"]["latency"]
        print(f"{'Short-term write':<25} {lat['p50_ms']:>10.3f} {lat['p95_ms']:>10.3f} {lat['p99_ms']:>10.3f}")
    
    if "short_term_read" in b and "latency" in b["short_term_read"]:
        lat = b["short_term_read"]["latency"]
        print(f"{'Short-term read':<25} {lat['p50_ms']:>10.3f} {lat['p95_ms']:>10.3f} {lat['p99_ms']:>10.3f}")
    
    if "embedding_latency" in b and "by_text_length" in b["embedding_latency"]:
        for text_len, lat in b["embedding_latency"]["by_text_length"].items():
            print(f"{'Embedding (' + text_len.split('_')[1] + ' chars)':<25} {lat['p50_ms']:>10.2f} {lat['p95_ms']:>10.2f} {'N/A':>10}")
    
    if "semantic_search" in b and "p50_ms" in b["semantic_search"]:
        ss = b["semantic_search"]
        print(f"{'Semantic search':<25} {ss['p50_ms']:>10.2f} {ss['p95_ms']:>10.2f} {ss['p99_ms']:>10.2f}")
    
    if "context_building" in b and "by_token_limit" in b["context_building"]:
        for limit, lat in b["context_building"]["by_token_limit"].items():
            tokens = limit.split('_')[1]
            print(f"{'Context (' + tokens + ' tokens)':<25} {lat['p50_ms']:>10.2f} {lat['p95_ms']:>10.2f} {'N/A':>10}")
    
    if "llm_management" in b and "p50_ms" in b["llm_management"]:
        lm = b["llm_management"]
        print(f"{'LLM management':<25} {lm['p50_ms']:>10.2f} {lm['p95_ms']:>10.2f} {'N/A':>10}")
    
    if "end_to_end" in b and "p50_ms" in b["end_to_end"]:
        e2e = b["end_to_end"]
        print(f"{'End-to-end turn':<25} {e2e['p50_ms']:>10.2f} {e2e['p95_ms']:>10.2f} {e2e['p99_ms']:>10.2f}")
    
    print("\n### CAPACITY LIMITS")
    print("-" * 50)
    if "capacity_limits" in b and "results" in b["capacity_limits"]:
        cap = b["capacity_limits"]["results"]
        if "short_term" in cap:
            st = cap["short_term"]
            print(f"  Short-term: {st['entries']:,} entries, {st['tokens']:,} tokens @ {st['entries_per_sec']:,.0f}/sec")
        if "feature_memory" in cap:
            fm = cap["feature_memory"]
            print(f"  Features:   {fm['facts']:,} facts, {fm['max_history_depth']} history depth @ {fm['facts_per_sec']:,.0f}/sec")
    
    print("\n" + "=" * 70)


def main():
    print("=" * 70)
    print("LLM MEMORY FRAMEWORK BENCHMARK SUITE")
    print("=" * 70)
    
    client = get_llm_client()
    if not client:
        print("\n[ERROR] LLM server not running!")
        print("Start with: .\\venv\\Scripts\\python.exe server\\model_server.py")
        return None
    
    print("\n[OK] LLM server connected")
    print("Running benchmarks... (this may take a few minutes)\n")
    
    gc.collect()  # Clean up before benchmarking
    
    results = run_all_benchmarks(client)
    
    # Print summary
    print_summary(results)
    
    # Save to file
    output_file = "memory/benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    main()
