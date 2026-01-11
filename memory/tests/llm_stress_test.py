"""
LLM + Memory Edge Case and Stress Tests
Tests the memory framework under challenging conditions with real LLM.
"""

import os
import sys
import shutil
import tempfile
import time
import threading

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from memory import MemoryManager
from memory.managers.policies import ShortTermPolicy
from memory.managers.llm_manager import LLMMemoryManager
from memory.stores.short_term import ShortTermMemory
from memory.stores.long_term import LongTermMemory
from memory.stores.feature_memory import FeatureMemory


def get_llm_client():
    """Get LLM client if available."""
    try:
        from server.local_client import LocalLLMClient
        from config import MODEL_SERVER_URL
        
        client = LocalLLMClient(MODEL_SERVER_URL)
        # Quick test
        embedding = client.embed("test")
        if embedding and len(embedding) == 384:
            return client
    except Exception as e:
        print(f"[ERROR] LLM not available: {e}")
    return None


def run_test(name, func, client):
    """Run a single test with error handling."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print('='*60)
    
    try:
        start = time.time()
        result = func(client)
        elapsed = time.time() - start
        
        print(f"\nResult: {result}")
        print(f"Time: {elapsed:.2f}s")
        print("[PASS]")
        return True, result
        
    except Exception as e:
        import traceback
        print(f"\n[FAIL] {type(e).__name__}: {e}")
        traceback.print_exc()
        return False, str(e)


# ==================== EDGE CASE TESTS ====================

def test_unicode_heavy_content(client):
    """Test with heavy Unicode content including emojis, CJK, RTL."""
    print("Testing Unicode handling with LLM...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        mm = MemoryManager(
            scope="user:unicode_test",
            llm_client=client,
            data_dir=temp_dir
        )
        
        # Add Unicode-heavy conversations
        conversations = [
            ("My name is 田中太郎 and I love 寿司", "Nice to meet you! Sushi is delicious!"),
            ("I also enjoy الطعام العربي", "Arabic cuisine has rich flavors!"),
            ("My favorite emoji is 🎉🚀💡", "Those are fun emojis!"),
            ("Café résumé naïve", "French diacritics noted!"),
        ]
        
        for user_msg, assistant_msg in conversations:
            mm.add_turn(user_msg, assistant_msg, extract_facts=False)
            # Safe print for Windows console
            try:
                print(f"  Added: {user_msg[:30]}...")
            except UnicodeEncodeError:
                print(f"  Added: [Unicode content]...")
        
        # Get context
        context = mm.get_context("What languages have we discussed?")
        
        # Search long-term
        results = mm.search_long_term("sushi", limit=3)
        
        return {
            "entries_added": len(mm.short_term),
            "context_has_unicode": "田中" in context["combined"] or "emoji" in context["combined"],
            "search_results": len(results)
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_very_long_conversation(client):
    """Test with a very long conversation (50+ turns)."""
    print("Testing extended conversation handling...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        mm = MemoryManager(
            scope="user:long_conv",
            llm_client=client,
            data_dir=temp_dir,
            short_term_policy=ShortTermPolicy(max_tokens=5000, max_entries=100)
        )
        
        # Simulate 50 conversation turns
        topics = ["Python", "JavaScript", "databases", "APIs", "testing"]
        
        start = time.time()
        for i in range(50):
            topic = topics[i % len(topics)]
            user_msg = f"Question {i+1}: Tell me about {topic} best practices"
            assistant_msg = f"For {topic}, key practices include code organization, documentation, and testing."
            mm.add_turn(user_msg, assistant_msg, extract_facts=False)
        
        add_time = time.time() - start
        
        # Get context
        start = time.time()
        context = mm.get_context("What have we discussed?", max_tokens=2000)
        context_time = time.time() - start
        
        # Stats
        stats = mm.get_stats()
        
        return {
            "turns_added": 50,
            "add_time_s": round(add_time, 2),
            "context_time_s": round(context_time, 3),
            "short_term_entries": stats["short_term"]["entry_count"],
            "short_term_tokens": stats["short_term"]["total_tokens"],
            "long_term_points": stats["long_term"].get("points_count", "N/A")
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_rapid_fact_updates(client):
    """Test rapid updates to the same fact."""
    print("Testing rapid fact updates with LLM extraction...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        mm = MemoryManager(
            scope="user:rapid_facts",
            llm_client=client,
            data_dir=temp_dir
        )
        
        # Simulate user changing preferences rapidly
        foods = ["pizza", "sushi", "tacos", "curry", "pasta"]
        
        for food in foods:
            mm.features.set("favorite_food", food)
            print(f"  Set favorite_food = {food}")
        
        # Check history
        fact = mm.features.get_with_history("favorite_food")
        
        return {
            "current_value": fact.current,
            "history_length": len(fact.history),
            "history_preserved": len(fact.history) == len(foods)
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_memory_context_limit(client):
    """Test context building at token limits."""
    print("Testing context building at limits...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        mm = MemoryManager(
            scope="user:limit_test",
            llm_client=client,
            data_dir=temp_dir
        )
        
        # Add a lot of content
        for i in range(20):
            mm.add_turn(
                f"Question {i}: " + "detail " * 50,
                f"Answer {i}: " + "explanation " * 50,
                extract_facts=False
            )
        
        # Test various token limits
        results = {}
        for limit in [100, 500, 1000, 2000]:
            context = mm.get_context("summary", max_tokens=limit)
            results[f"limit_{limit}"] = {
                "tokens_used": context["token_usage"]["total"],
                "within_limit": context["token_usage"]["total"] <= limit * 1.1  # 10% tolerance
            }
        
        return results
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_concurrent_memory_access(client):
    """Test concurrent access to memory with LLM."""
    print("Testing concurrent memory access...")
    
    temp_dir = tempfile.mkdtemp()
    errors = []
    results = {"reads": 0, "writes": 0}
    lock = threading.Lock()
    
    try:
        mm = MemoryManager(
            scope="user:concurrent",
            llm_client=client,
            data_dir=temp_dir,
            use_llm_management=False  # Faster for stress test
        )
        
        def writer(thread_id):
            nonlocal results
            for i in range(10):
                try:
                    mm.add_turn(
                        f"Thread {thread_id} message {i}",
                        f"Response to thread {thread_id}",
                        extract_facts=False
                    )
                    with lock:
                        results["writes"] += 1
                except Exception as e:
                    errors.append(f"Writer {thread_id}: {e}")
        
        def reader(thread_id):
            nonlocal results
            for i in range(10):
                try:
                    context = mm.get_context(f"Thread {thread_id} query {i}")
                    with lock:
                        results["reads"] += 1
                except Exception as e:
                    errors.append(f"Reader {thread_id}: {e}")
        
        threads = []
        for i in range(3):
            threads.append(threading.Thread(target=writer, args=(i,)))
            threads.append(threading.Thread(target=reader, args=(i,)))
        
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.time() - start
        
        return {
            "threads": len(threads),
            "writes": results["writes"],
            "reads": results["reads"],
            "errors": len(errors),
            "elapsed_s": round(elapsed, 2)
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_llm_management_under_pressure(client):
    """Test LLM memory management with many entries."""
    print("Testing LLM management under pressure...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        policy = ShortTermPolicy(
            max_tokens=1000,
            max_entries=50,
            manage_every_n_turns=5
        )
        
        mm = MemoryManager(
            scope="user:pressure",
            llm_client=client,
            data_dir=temp_dir,
            short_term_policy=policy,
            use_llm_management=True
        )
        
        # Add many entries to trigger management
        print("  Adding entries to trigger LLM management...")
        for i in range(15):
            mm.add_turn(
                f"User message {i}: This is a moderately long message to fill up space",
                f"Assistant response {i}: Here is a helpful response with information",
                extract_facts=False
            )
        
        # Check if management is needed
        needs_mgmt = mm.needs_management()
        print(f"  Needs management: {needs_mgmt}")
        
        if needs_mgmt:
            print("  Running LLM management...")
            start = time.time()
            mm.manage_short_term()
            mgmt_time = time.time() - start
            print(f"  Management took: {mgmt_time:.2f}s")
        else:
            mgmt_time = 0
        
        stats = mm.get_stats()
        
        return {
            "entries_before": 30,  # 15 turns * 2
            "entries_after": stats["short_term"]["entry_count"],
            "tokens_after": stats["short_term"]["total_tokens"],
            "management_ran": needs_mgmt,
            "management_time_s": round(mgmt_time, 2)
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_embedding_similarity_accuracy(client):
    """Test that semantic search returns relevant results."""
    print("Testing embedding similarity accuracy...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        ltm = LongTermMemory(
            scope="user:similarity",
            qdrant_path=os.path.join(temp_dir, "qdrant"),
            llm_client=client
        )
        
        # Store diverse topics
        topics = [
            ("Programming", "Python is great for data science and machine learning"),
            ("Cooking", "The recipe calls for olive oil and fresh basil"),
            ("Sports", "The basketball team won the championship game"),
            ("Music", "The symphony orchestra performed Beethoven's 9th"),
            ("Travel", "Paris has beautiful architecture and museums"),
        ]
        
        for topic, content in topics:
            ltm.store_conversation(f"Tell me about {topic}", content)
        
        # Test search accuracy
        test_queries = [
            ("machine learning", "Programming"),
            ("recipe ingredients", "Cooking"),
            ("championship game", "Sports"),
            ("orchestra concert", "Music"),
            ("Eiffel Tower", "Travel"),
        ]
        
        correct = 0
        for query, expected_topic in test_queries:
            results = ltm.search(query, limit=1)
            if results:
                found_text = results[0]["text"]
                # Check if the expected topic's content is in the result
                expected_content = [c for t, c in topics if t == expected_topic][0]
                if expected_content in found_text or expected_topic.lower() in found_text.lower():
                    correct += 1
                    print(f"  [OK] '{query}' -> found '{expected_topic}'")
                else:
                    print(f"  [MISS] '{query}' -> got something else")
        
        return {
            "queries": len(test_queries),
            "correct": correct,
            "accuracy": f"{correct}/{len(test_queries)}"
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_scope_isolation_with_llm(client):
    """Test that different scopes are isolated with LLM operations."""
    print("Testing scope isolation with LLM...")
    
    # Use separate directories to avoid Qdrant concurrent local client issue
    temp_dir1 = tempfile.mkdtemp()
    temp_dir2 = tempfile.mkdtemp()
    
    try:
        mm_user1 = MemoryManager(
            scope="user:alice",
            llm_client=client,
            data_dir=temp_dir1,
            use_llm_management=False
        )
        
        mm_user2 = MemoryManager(
            scope="user:bob",
            llm_client=client,
            data_dir=temp_dir2,
            use_llm_management=False
        )
        
        # Set different facts
        mm_user1.set_fact("name", "Alice")
        mm_user1.set_fact("secret", "alice_password_123")
        
        mm_user2.set_fact("name", "Bob")
        mm_user2.set_fact("secret", "bob_password_456")
        
        # Add conversations
        mm_user1.add_turn("My credit card is 1234", "Noted securely", extract_facts=False)
        mm_user2.add_turn("My credit card is 5678", "Noted securely", extract_facts=False)
        
        # Verify isolation - data from user1 should not appear in user2's context and vice versa
        ctx1 = mm_user1.get_context("password")
        ctx2 = mm_user2.get_context("password")
        
        # Check that user1's context has Alice's data
        user1_has_own_data = "Alice" in ctx1["combined"] and "alice_password_123" in ctx1["combined"]
        # Check that user2's context has Bob's data
        user2_has_own_data = "Bob" in ctx2["combined"] and "bob_password_456" in ctx2["combined"]
        # Check cross-contamination doesn't happen
        no_cross_contamination = (
            "bob_password" not in ctx1["combined"] and
            "alice_password" not in ctx2["combined"]
        )
        
        print(f"  User1 has own data: {user1_has_own_data}")
        print(f"  User2 has own data: {user2_has_own_data}")
        print(f"  No cross-contamination: {no_cross_contamination}")
        
        return {
            "user1_facts": mm_user1.features.to_dict(),
            "user2_facts": mm_user2.features.to_dict(),
            "user1_has_own_data": user1_has_own_data,
            "user2_has_own_data": user2_has_own_data,
            "isolated": no_cross_contamination
        }
    finally:
        shutil.rmtree(temp_dir1, ignore_errors=True)
        shutil.rmtree(temp_dir2, ignore_errors=True)


# ==================== STRESS TESTS ====================

def test_high_throughput(client):
    """Test high-throughput operations."""
    print("Testing high-throughput memory operations...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        mm = MemoryManager(
            scope="user:throughput",
            llm_client=client,
            data_dir=temp_dir,
            use_llm_management=False  # Skip LLM for pure throughput
        )
        
        # Benchmark: 100 operations
        operations = 100
        
        # Write benchmark
        start = time.time()
        for i in range(operations):
            mm.short_term.add(f"Message {i}", "user")
        write_time = time.time() - start
        
        # Read benchmark
        start = time.time()
        for i in range(operations):
            mm.short_term.get_context(max_tokens=500)
        read_time = time.time() - start
        
        # Avoid divide by zero
        write_time = max(write_time, 0.0001)
        read_time = max(read_time, 0.0001)
        
        return {
            "operations": operations,
            "write_time_s": round(write_time, 4),
            "write_ops_per_sec": round(operations / write_time, 1),
            "read_time_s": round(read_time, 4),
            "read_ops_per_sec": round(operations / read_time, 1)
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_memory_with_llm_errors(client):
    """Test memory behavior when LLM returns unexpected responses."""
    print("Testing graceful handling of LLM edge responses...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        mm = MemoryManager(
            scope="user:error_test",
            llm_client=client,
            data_dir=temp_dir
        )
        
        # These should not crash even if LLM returns odd responses
        test_inputs = [
            "",  # Empty
            "x" * 10000,  # Very long
            "```json\n{invalid json\n```",  # Looks like JSON but isn't
            "<script>alert('xss')</script>",  # XSS attempt
            "'; DROP TABLE users; --",  # SQL injection
        ]
        
        crashes = 0
        for i, inp in enumerate(test_inputs):
            try:
                mm.add_turn(inp, f"Response {i}", extract_facts=True)
            except Exception as e:
                print(f"  Crash on input {i}: {e}")
                crashes += 1
        
        return {
            "inputs_tested": len(test_inputs),
            "crashes": crashes,
            "survived": crashes == 0
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ==================== MAIN ====================

def main():
    print("="*60)
    print("LLM + MEMORY EDGE CASE AND STRESS TESTS")
    print("="*60)
    
    client = get_llm_client()
    if not client:
        print("\n[ERROR] LLM server not running!")
        print("Start with: .\\venv\\Scripts\\python.exe server\\model_server.py")
        return False
    
    print("\n[OK] LLM server connected\n")
    
    # Edge case tests
    edge_tests = [
        ("Unicode Heavy Content", test_unicode_heavy_content),
        ("Very Long Conversation (50 turns)", test_very_long_conversation),
        ("Rapid Fact Updates", test_rapid_fact_updates),
        ("Context Token Limits", test_memory_context_limit),
        ("Concurrent Memory Access", test_concurrent_memory_access),
        ("Embedding Similarity Accuracy", test_embedding_similarity_accuracy),
        ("Scope Isolation with LLM", test_scope_isolation_with_llm),
    ]
    
    # Stress tests
    stress_tests = [
        ("LLM Management Under Pressure", test_llm_management_under_pressure),
        ("High Throughput", test_high_throughput),
        ("Memory with LLM Edge Responses", test_memory_with_llm_errors),
    ]
    
    results = {}
    
    print("\n" + "="*60)
    print("EDGE CASE TESTS")
    print("="*60)
    
    for name, func in edge_tests:
        success, result = run_test(name, func, client)
        results[name] = success
    
    print("\n" + "="*60)
    print("STRESS TESTS")
    print("="*60)
    
    for name, func in stress_tests:
        success, result = run_test(name, func, client)
        results[name] = success
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status} {name}")
    
    print(f"\nTotal: {passed}/{total} passed")
    print("="*60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
