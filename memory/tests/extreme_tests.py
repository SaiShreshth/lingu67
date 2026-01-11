"""
Extreme Limit Tests - Find breaking points of the memory framework.

Tests designed to push the system beyond normal limits and identify
edge behaviors, potential failures, and edge cases.
"""

import os
import sys
import gc
import time
import shutil
import tempfile
import traceback
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from memory.utils.scopes import MemoryScope
from memory.managers.policies import ShortTermPolicy
from memory.stores.feature_memory import FeatureMemory
from memory.stores.short_term import ShortTermMemory
from memory.core import MemoryManager
from memory.utils.helpers import count_tokens


def run_test(name, func):
    """Run a single test with error handling."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print('='*60)
    
    try:
        start = time.time()
        result = func()
        elapsed = time.time() - start
        
        print(f"Result: {result}")
        print(f"Time: {elapsed:.2f}s")
        print("[PASS]")
        return True, result
        
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        traceback.print_exc()
        return False, str(e)


# ==================== EXTREME TESTS ====================

def test_massive_content():
    """Test with extremely large content (10MB+)."""
    stm = ShortTermMemory(
        scope="user:extreme",
        policy=ShortTermPolicy(max_tokens=1000000, max_entries=10000)
    )
    
    # Generate 10MB content
    massive_content = "x" * (10 * 1024 * 1024)  # 10MB
    
    entry = stm.add(massive_content, "user")
    
    return {
        "content_size_mb": len(massive_content) / (1024 * 1024),
        "tokens_estimated": entry.tokens,
        "stored": True
    }


def test_thousands_of_entries():
    """Test with 10,000+ short-term entries."""
    stm = ShortTermMemory(
        scope="user:extreme",
        policy=ShortTermPolicy(max_tokens=10000000, max_entries=100000)
    )
    
    start = time.time()
    for i in range(10000):
        stm.add(f"Message number {i} with some content", "user")
    
    add_time = time.time() - start
    
    start = time.time()
    context = stm.get_context(max_tokens=5000)
    get_time = time.time() - start
    
    return {
        "entries_added": 10000,
        "add_time_s": round(add_time, 2),
        "context_size": len(context),
        "get_time_s": round(get_time, 4),
        "total_tokens": stm.total_tokens
    }


def test_deep_history():
    """Test fact with 1000+ historical changes."""
    temp_dir = tempfile.mkdtemp()
    try:
        fm = FeatureMemory(scope="user:extreme", storage_dir=temp_dir)
        
        start = time.time()
        for i in range(1000):
            fm.set("deep_fact", f"value_{i}")
        
        set_time = time.time() - start
        
        fact = fm.get_with_history("deep_fact")
        
        # Get prompt with history
        start = time.time()
        prompt = fm.to_prompt(include_history=True)
        prompt_time = time.time() - start
        
        return {
            "history_length": len(fact.history),
            "set_time_s": round(set_time, 2),
            "prompt_time_s": round(prompt_time, 4),
            "prompt_size_kb": len(prompt) / 1024,
            "current_value": fact.current
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_concurrent_rapid_access():
    """Test rapid-fire access patterns."""
    import threading
    
    temp_dir = tempfile.mkdtemp()
    errors = []
    
    try:
        fm = FeatureMemory(scope="user:concurrent", storage_dir=temp_dir)
        
        def writer():
            for i in range(100):
                try:
                    fm.set(f"key_{threading.current_thread().name}_{i}", f"value_{i}")
                except Exception as e:
                    errors.append(str(e))
        
        threads = [threading.Thread(target=writer, name=f"thread_{i}") for i in range(5)]
        
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        elapsed = time.time() - start
        
        return {
            "threads": 5,
            "writes_per_thread": 100,
            "total_writes": 500,
            "elapsed_s": round(elapsed, 2),
            "errors": len(errors),
            "final_fact_count": len(fm)
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_memory_pressure():
    """Test memory usage under pressure."""
    entries = []
    stm = ShortTermMemory(
        scope="user:memory",
        policy=ShortTermPolicy(max_tokens=100000000, max_entries=1000000)
    )
    
    # Track memory before
    gc.collect()
    
    # Add entries until we hit 100MB of content
    target_mb = 50
    content = "x" * 1000  # 1KB per entry
    start = time.time()
    
    while stm.total_tokens * 4 < target_mb * 1024 * 1024:  # rough token to bytes
        stm.add(content, "user")
        if len(stm) % 10000 == 0:
            print(f"  Added {len(stm)} entries, ~{stm.total_tokens * 4 / (1024*1024):.1f}MB")
    
    elapsed = time.time() - start
    
    gc.collect()
    
    return {
        "target_mb": target_mb,
        "entries": len(stm),
        "total_tokens": stm.total_tokens,
        "elapsed_s": round(elapsed, 2),
        "entries_per_second": int(len(stm) / elapsed)
    }


def test_token_budget_extreme():
    """Test context building with extreme token budgets."""
    stm = ShortTermMemory(
        scope="user:budget",
        policy=ShortTermPolicy(max_tokens=1000000, max_entries=50000)
    )
    
    # Add many entries
    for i in range(1000):
        stm.add(f"Message {i}: " + "content " * 50, "user")
    
    # Test various budget sizes
    results = {}
    for budget in [10, 100, 1000, 10000, 100000]:
        start = time.time()
        context = stm.get_context(max_tokens=budget)
        elapsed = time.time() - start
        
        actual_tokens = sum(count_tokens(m["content"]) for m in context)
        
        results[f"budget_{budget}"] = {
            "messages": len(context),
            "actual_tokens": actual_tokens,
            "time_ms": round(elapsed * 1000, 2)
        }
    
    return results


def test_scope_isolation():
    """Test that scopes are properly isolated."""
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create multiple scopes
        fm_user1 = FeatureMemory(scope="user:user1", storage_dir=temp_dir)
        fm_user2 = FeatureMemory(scope="user:user2", storage_dir=temp_dir)
        fm_global = FeatureMemory(scope="global", storage_dir=temp_dir)
        
        # Set different values
        fm_user1.set("secret", "user1_secret")
        fm_user2.set("secret", "user2_secret")
        fm_global.set("shared", "global_value")
        
        # Verify isolation
        isolated = (
            fm_user1.get("secret") == "user1_secret" and
            fm_user2.get("secret") == "user2_secret" and
            fm_user1.get("shared") is None and  # User can't see global
            fm_global.get("secret") is None     # Global can't see user
        )
        
        return {
            "user1_secret": fm_user1.get("secret"),
            "user2_secret": fm_user2.get("secret"),
            "isolation_correct": isolated
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_corrupted_storage_recovery():
    """Test recovery from corrupted storage files."""
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create and populate
        fm1 = FeatureMemory(scope="user:corrupt", storage_dir=temp_dir)
        fm1.set("key", "value")
        
        # Corrupt the file
        filepath = fm1._get_filepath()
        with open(filepath, 'w') as f:
            f.write("{{{{not valid json}}}")
        
        # Try to load corrupted file
        fm2 = FeatureMemory(scope="user:corrupt", storage_dir=temp_dir)
        
        # Should recover gracefully (empty state)
        return {
            "recovered": True,
            "fact_count": len(fm2),
            "can_set_new": True if fm2.set("new_key", "new_value") is None else False
        }
    except Exception as e:
        return {"recovered": False, "error": str(e)}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_special_content():
    """Test with special/problematic content."""
    stm = ShortTermMemory(scope="user:special")
    
    test_cases = {
        "empty": "",
        "null_bytes": "text\x00with\x00nulls",
        "unicode": "日本語 テスト 🎉 émojis",
        "newlines": "line1\nline2\r\nline3",
        "json_like": '{"key": "value"}',
        "sql_injection": "'; DROP TABLE users; --",
        "html": "<script>alert('xss')</script>",
        "very_long_word": "a" * 10000,
    }
    
    results = {}
    for name, content in test_cases.items():
        try:
            entry = stm.add(content, "user")
            results[name] = {
                "stored": True,
                "tokens": entry.tokens,
                "retrieved": entry.content == content
            }
        except Exception as e:
            results[name] = {"stored": False, "error": str(e)}
    
    return results


def test_auto_prune_behavior():
    """Test automatic pruning behavior at limits."""
    policy = ShortTermPolicy(max_tokens=100, max_entries=5, min_entries=2)
    stm = ShortTermMemory(scope="user:prune", policy=policy)
    
    history = []
    
    # Add entries and track pruning
    for i in range(20):
        stm.add(f"Message {i}", "user")
        history.append({
            "added": i,
            "entry_count": len(stm),
            "total_tokens": stm.total_tokens
        })
    
    return {
        "final_entry_count": len(stm),
        "final_tokens": stm.total_tokens,
        "min_entries_respected": len(stm) >= policy.min_entries,
        "max_tokens_respected": stm.total_tokens <= policy.max_tokens,
        "history_sample": history[-5:]
    }


# ==================== MAIN ====================

def main():
    print("=" * 60)
    print("EXTREME LIMIT TESTS")
    print("Finding breaking points of the memory framework")
    print("=" * 60)
    
    tests = [
        ("Massive Content (10MB)", test_massive_content),
        ("10,000 Entries", test_thousands_of_entries),
        ("Deep History (1000 changes)", test_deep_history),
        ("Concurrent Rapid Access", test_concurrent_rapid_access),
        ("Memory Pressure (50MB)", test_memory_pressure),
        ("Token Budget Extremes", test_token_budget_extreme),
        ("Scope Isolation", test_scope_isolation),
        ("Corrupted Storage Recovery", test_corrupted_storage_recovery),
        ("Special Content Handling", test_special_content),
        ("Auto-Prune Behavior", test_auto_prune_behavior),
    ]
    
    results = {}
    passed = 0
    failed = 0
    
    for name, func in tests:
        success, result = run_test(name, func)
        results[name] = {"success": success, "result": result}
        if success:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed > 0:
        print("\nFailed tests:")
        for name, data in results.items():
            if not data["success"]:
                print(f"  - {name}: {data['result']}")
    
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
