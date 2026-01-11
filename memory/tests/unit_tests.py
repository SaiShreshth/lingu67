"""
Memory Framework Tests
Comprehensive test suite: Unit, Component, Edge, and Stress tests.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from memory.utils.scopes import MemoryScope, ScopeType, parse_scope, validate_scope
from memory.managers.policies import ShortTermPolicy, PolicyEnforcer
from memory.utils.helpers import (
    count_tokens, truncate_to_tokens, TokenBudget,
    merge_similar_texts, extract_json_from_text, hash_content
)
from memory.stores.feature_memory import FeatureMemory, Fact, FactValue
from memory.stores.short_term import ShortTermMemory, MemoryEntry
from memory.stores.long_term import LongTermMemory
from memory.managers.llm_manager import LLMMemoryManager, SimpleMemoryManager
from memory.core import MemoryManager, get_memory_manager, clear_manager_cache


class TestScopes(unittest.TestCase):
    """Unit tests for memory scopes."""
    
    def test_global_scope(self):
        scope = MemoryScope.global_scope()
        self.assertTrue(scope.is_global())
        self.assertEqual(str(scope), "global")
    
    def test_user_scope(self):
        scope = MemoryScope.user("123")
        self.assertTrue(scope.is_user())
        self.assertEqual(str(scope), "user:123")
        self.assertEqual(scope.identifier, "123")
    
    def test_session_scope(self):
        scope = MemoryScope.session("abc")
        self.assertTrue(scope.is_session())
        self.assertEqual(str(scope), "session:abc")
    
    def test_parse_global(self):
        scope = MemoryScope.parse("global")
        self.assertTrue(scope.is_global())
    
    def test_parse_user(self):
        scope = MemoryScope.parse("user:test_user")
        self.assertTrue(scope.is_user())
        self.assertEqual(scope.identifier, "test_user")
    
    def test_parse_invalid(self):
        with self.assertRaises(ValueError):
            MemoryScope.parse("invalid:format:too:many")
    
    def test_storage_key(self):
        scope = MemoryScope.user("user@email.com")
        key = scope.get_storage_key()
        self.assertIn("user_", key)
        self.assertNotIn("@", key)  # Should be sanitized
    
    def test_validate_scope(self):
        self.assertTrue(validate_scope("global"))
        self.assertTrue(validate_scope("user:123"))
        self.assertFalse(validate_scope("bad"))


class TestPolicies(unittest.TestCase):
    """Unit tests for memory policies."""
    
    def setUp(self):
        self.policy = ShortTermPolicy(
            max_tokens=1000,
            target_tokens=800,
            max_entries=20,
            max_age_minutes=60
        )
    
    def test_over_capacity(self):
        self.assertFalse(self.policy.is_over_capacity(500))
        self.assertTrue(self.policy.is_over_capacity(1500))
    
    def test_needs_compression(self):
        # Default threshold is 80%
        self.assertFalse(self.policy.needs_compression(700))
        self.assertTrue(self.policy.needs_compression(900))
    
    def test_entry_expired(self):
        recent = datetime.now() - timedelta(minutes=30)
        old = datetime.now() - timedelta(minutes=120)
        
        self.assertFalse(self.policy.is_entry_expired(recent))
        self.assertTrue(self.policy.is_entry_expired(old))
    
    def test_decay_calculation(self):
        now = datetime.now()
        
        # No decay for recently accessed
        decay = self.policy.calculate_decay(now, now)
        self.assertAlmostEqual(decay, 1.0, places=2)
        
        # Half decay after half-life
        half_life_ago = now - timedelta(minutes=self.policy.decay_half_life_minutes)
        decay = self.policy.calculate_decay(half_life_ago, half_life_ago)
        self.assertAlmostEqual(decay, 0.5, places=1)
    
    def test_policy_enforcer(self):
        enforcer = PolicyEnforcer(self.policy)
        
        # Should not manage initially
        self.assertFalse(enforcer.should_manage())
        
        # After enough turns
        for _ in range(self.policy.manage_every_n_turns):
            enforcer.record_turn()
        
        self.assertTrue(enforcer.should_manage())
        
        enforcer.reset_manage_counter()
        self.assertFalse(enforcer.should_manage())


class TestUtils(unittest.TestCase):
    """Unit tests for utility functions."""
    
    def test_count_tokens(self):
        # Rough estimate test
        short_text = "Hello"
        long_text = "This is a much longer text that should have more tokens"
        
        short_tokens = count_tokens(short_text)
        long_tokens = count_tokens(long_text)
        
        self.assertGreater(long_tokens, short_tokens)
        self.assertGreater(short_tokens, 0)
    
    def test_count_tokens_empty(self):
        self.assertEqual(count_tokens(""), 0)
        self.assertEqual(count_tokens(None), 0)
    
    def test_truncate_to_tokens(self):
        text = "A" * 1000
        truncated = truncate_to_tokens(text, 10)
        
        self.assertLess(len(truncated), len(text))
        self.assertTrue(truncated.endswith("..."))
    
    def test_token_budget(self):
        budget = TokenBudget(100)
        
        self.assertTrue(budget.allocate("a", 50))
        self.assertEqual(budget.remaining(), 50)
        
        self.assertFalse(budget.allocate("b", 60))  # Over budget
        self.assertTrue(budget.allocate("b", 40))
        
        self.assertEqual(budget.remaining(), 10)
    
    def test_extract_json(self):
        text = 'Some text {"key": "value"} more text'
        result = extract_json_from_text(text)
        
        self.assertIsNotNone(result)
        self.assertEqual(result["key"], "value")
    
    def test_merge_similar_texts(self):
        texts = [
            "The cat sat on the mat",
            "The cat sat on a mat",  # Very similar
            "The dog ran in the park"  # Different
        ]
        
        merged = merge_similar_texts(texts, threshold=0.7)
        self.assertLessEqual(len(merged), len(texts))
    
    def test_hash_content(self):
        h1 = hash_content("test content")
        h2 = hash_content("test content")
        h3 = hash_content("different content")
        
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)


class TestFeatureMemory(unittest.TestCase):
    """Unit tests for feature memory."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.fm = FeatureMemory(scope="user:test", storage_dir=self.temp_dir)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_set_and_get(self):
        self.fm.set("name", "John")
        self.assertEqual(self.fm.get("name"), "John")
    
    def test_get_default(self):
        self.assertIsNone(self.fm.get("nonexistent"))
        self.assertEqual(self.fm.get("nonexistent", "default"), "default")
    
    def test_history_tracking(self):
        self.fm.set("color", "red")
        self.fm.set("color", "blue")
        self.fm.set("color", "green")
        
        fact = self.fm.get_with_history("color")
        self.assertEqual(fact.current, "green")
        self.assertEqual(len(fact.history), 3)
        
        # Check history order
        self.assertEqual(fact.history[0].value, "red")
        self.assertEqual(fact.history[1].value, "blue")
        self.assertEqual(fact.history[2].value, "green")
    
    def test_no_history_for_same_value(self):
        self.fm.set("name", "John")
        self.fm.set("name", "John")  # Same value
        
        fact = self.fm.get_with_history("name")
        self.assertEqual(len(fact.history), 1)  # No duplicate
    
    def test_to_prompt(self):
        self.fm.set("name", "John")
        self.fm.set("age", 30)
        
        prompt = self.fm.to_prompt()
        data = json.loads(prompt)
        
        self.assertEqual(data["name"], "John")
        self.assertEqual(data["age"], 30)
    
    def test_to_prompt_with_history(self):
        self.fm.set("fruit", "apple")
        self.fm.set("fruit", "orange")
        
        prompt = self.fm.to_prompt(include_history=True)
        data = json.loads(prompt)
        
        # Should have history note
        self.assertIn("current", data["fruit"])
    
    def test_persistence(self):
        self.fm.set("test_key", "test_value")
        
        # Create new instance
        fm2 = FeatureMemory(scope="user:test", storage_dir=self.temp_dir)
        self.assertEqual(fm2.get("test_key"), "test_value")
    
    def test_delete(self):
        self.fm.set("key", "value")
        self.assertTrue("key" in self.fm)
        
        self.fm.delete("key")
        self.assertFalse("key" in self.fm)


class TestShortTermMemory(unittest.TestCase):
    """Unit tests for short-term memory."""
    
    def setUp(self):
        self.policy = ShortTermPolicy(max_tokens=500, max_entries=10)
        self.stm = ShortTermMemory(scope="user:test", policy=self.policy)
    
    def test_add_entry(self):
        entry = self.stm.add("Hello", "user")
        
        self.assertEqual(len(self.stm), 1)
        self.assertEqual(entry.content, "Hello")
        self.assertEqual(entry.role, "user")
    
    def test_add_turn(self):
        user_entry, assistant_entry = self.stm.add_turn("Hi", "Hello!")
        
        self.assertEqual(len(self.stm), 2)
        self.assertEqual(user_entry.role, "user")
        self.assertEqual(assistant_entry.role, "assistant")
    
    def test_get_context(self):
        self.stm.add("Message 1", "user")
        self.stm.add("Response 1", "assistant")
        
        context = self.stm.get_context()
        
        self.assertEqual(len(context), 2)
        self.assertEqual(context[0]["role"], "user")
        self.assertEqual(context[1]["role"], "assistant")
    
    def test_token_limit(self):
        # Add many entries to exceed limit
        for i in range(20):
            self.stm.add(f"Message {i} " * 20, "user")
        
        # Should auto-prune
        self.assertLessEqual(self.stm.total_tokens, self.policy.max_tokens)
    
    def test_delete_entry(self):
        entry = self.stm.add("Test", "user")
        initial_count = len(self.stm)
        
        self.stm.delete(entry.id)
        self.assertEqual(len(self.stm), initial_count - 1)
    
    def test_compress_entry(self):
        entry = self.stm.add("This is a very long message " * 10, "user")
        original_tokens = entry.tokens
        
        self.stm.compress_entry(entry.id, "Short summary")
        
        # Find the entry again
        entries = self.stm.get_entries()
        compressed = [e for e in entries if e.id == entry.id][0]
        
        self.assertTrue(compressed.compressed)
        self.assertLess(compressed.tokens, original_tokens)
    
    def test_get_stats(self):
        self.stm.add("Test", "user")
        
        stats = self.stm.get_stats()
        
        self.assertEqual(stats["entry_count"], 1)
        self.assertIn("total_tokens", stats)
        self.assertIn("usage_percent", stats)


class TestLongTermMemory(unittest.TestCase):
    """Unit tests for long-term memory."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock LLM client
        self.mock_llm = Mock()
        self.mock_llm.embed.return_value = [0.1] * 384
        
        self.ltm = LongTermMemory(
            scope="user:test",
            qdrant_path=os.path.join(self.temp_dir, "qdrant"),
            llm_client=self.mock_llm
        )
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_store_conversation(self):
        point_id = self.ltm.store_conversation("Hello", "Hi there!")
        self.assertIsNotNone(point_id)
        self.assertNotEqual(point_id, "")
    
    def test_store_file_summary(self):
        point_id = self.ltm.store_file_summary(
            "doc.txt",
            "This document contains information about..."
        )
        self.assertIsNotNone(point_id)
    
    def test_search(self):
        self.ltm.store_conversation("What is Python?", "Python is a programming language.")
        
        results = self.ltm.search("programming", limit=5)
        
        # Note: With mock embeddings, results may vary
        self.assertIsInstance(results, list)
    
    def test_get_stats(self):
        stats = self.ltm.get_stats()
        self.assertIn("collection", stats)


class TestSimpleMemoryManager(unittest.TestCase):
    """Unit tests for simple (rule-based) memory manager."""
    
    def setUp(self):
        self.manager = SimpleMemoryManager(min_keep=2)
    
    def test_keep_recent(self):
        entries = [
            {"id": "1", "age_minutes": 60, "content": "old"},
            {"id": "2", "age_minutes": 30, "content": "medium"},
            {"id": "3", "age_minutes": 5, "content": "recent"},
            {"id": "4", "age_minutes": 1, "content": "very recent"},
        ]
        
        actions = self.manager.get_management_actions(entries, 500, 1000)
        
        # Should not delete the last 2 (min_keep)
        deleted_ids = [a["id"] for a in actions if a["action"] == "DELETE"]
        self.assertNotIn("3", deleted_ids)
        self.assertNotIn("4", deleted_ids)
    
    def test_delete_old_greetings(self):
        entries = [
            {"id": "1", "age_minutes": 15, "content": "Hello there!"},
            {"id": "2", "age_minutes": 5, "content": "New message"},
            {"id": "3", "age_minutes": 1, "content": "Recent"},
        ]
        
        actions = self.manager.get_management_actions(entries, 500, 1000)
        
        deleted_ids = [a["id"] for a in actions if a["action"] == "DELETE"]
        self.assertIn("1", deleted_ids)  # Old greeting deleted


class TestMemoryManagerIntegration(unittest.TestCase):
    """Component tests for MemoryManager integration."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock LLM
        self.mock_llm = Mock()
        self.mock_llm.embed.return_value = [0.1] * 384
        self.mock_llm.complete.return_value = '{"facts": []}'
        
        self.mm = MemoryManager(
            scope="user:integration_test",
            llm_client=self.mock_llm,
            data_dir=self.temp_dir,
            use_llm_management=False  # Use simple manager
        )
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        clear_manager_cache()
    
    def test_add_turn(self):
        self.mm.add_turn("Hello", "Hi there!", extract_facts=False)
        
        # Check short-term
        self.assertEqual(len(self.mm.short_term), 2)
    
    def test_get_context(self):
        self.mm.add_turn("Hello", "Hi!", extract_facts=False)
        self.mm.features.set("name", "John")
        
        context = self.mm.get_context("test query")
        
        self.assertIn("short_term", context)
        self.assertIn("features", context)
        self.assertIn("combined", context)
    
    def test_set_and_get_fact(self):
        self.mm.set_fact("language", "Python")
        self.assertEqual(self.mm.get_fact("language"), "Python")
    
    def test_get_stats(self):
        self.mm.add_turn("Test", "Response", extract_facts=False)
        
        stats = self.mm.get_stats()
        
        self.assertIn("short_term", stats)
        self.assertIn("long_term", stats)
        self.assertIn("features", stats)
    
    def test_clear_session(self):
        self.mm.add_turn("Test", "Response", extract_facts=False)
        self.mm.clear_session()
        
        self.assertEqual(len(self.mm.short_term), 0)


# ==================== EDGE CASE TESTS ====================

class TestEdgeCases(unittest.TestCase):
    """Edge case tests."""
    
    def test_empty_content(self):
        stm = ShortTermMemory(scope="user:test")
        entry = stm.add("", "user")
        self.assertEqual(entry.tokens, 0)
    
    def test_unicode_content(self):
        temp_dir = tempfile.mkdtemp()
        fm = FeatureMemory(scope="user:test", storage_dir=temp_dir)
        fm.set("emoji", "🎉🎊🎈")
        self.assertEqual(fm.get("emoji"), "🎉🎊🎈")
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_very_long_content(self):
        stm = ShortTermMemory(scope="user:test")
        long_content = "x" * 100000
        entry = stm.add(long_content, "user")
        
        self.assertGreater(entry.tokens, 0)
    
    def test_null_values_in_facts(self):
        temp_dir = tempfile.mkdtemp()
        fm = FeatureMemory(scope="user:test", storage_dir=temp_dir)
        fm.set("nullable", None)
        self.assertIsNone(fm.get("nullable"))
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_special_characters_in_scope(self):
        scope = MemoryScope.user("user/with\\special:chars")
        key = scope.get_storage_key()
        # Should be sanitized
        self.assertNotIn("/", key)
        self.assertNotIn("\\", key)
    
    def test_concurrent_access_simulation(self):
        """Simulate concurrent access patterns."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            fm1 = FeatureMemory(scope="user:test", storage_dir=temp_dir)
            fm2 = FeatureMemory(scope="user:test", storage_dir=temp_dir)
            
            fm1.set("key", "value1")
            fm2.set("key", "value2")  # Overwrites
            
            # Reload and check
            fm3 = FeatureMemory(scope="user:test", storage_dir=temp_dir)
            self.assertEqual(fm3.get("key"), "value2")
        finally:
            shutil.rmtree(temp_dir)


# ==================== STRESS TESTS ====================

class TestStress(unittest.TestCase):
    """Stress tests for memory limits."""
    
    def test_many_short_term_entries(self):
        """Test with many short-term entries."""
        stm = ShortTermMemory(
            scope="user:stress",
            policy=ShortTermPolicy(max_tokens=10000, max_entries=1000)
        )
        
        for i in range(500):
            stm.add(f"Message {i}", "user")
        
        # Should handle without error
        context = stm.get_context()
        self.assertIsInstance(context, list)
    
    def test_many_facts(self):
        """Test with many feature facts."""
        temp_dir = tempfile.mkdtemp()
        try:
            fm = FeatureMemory(scope="user:stress", storage_dir=temp_dir)
            
            for i in range(200):
                fm.set(f"fact_{i}", f"value_{i}")
            
            self.assertEqual(len(fm), 200)
            
            # Test serialization
            prompt = fm.to_prompt()
            self.assertIsNotNone(prompt)
        finally:
            shutil.rmtree(temp_dir)
    
    def test_rapid_updates(self):
        """Test rapid fact updates."""
        temp_dir = tempfile.mkdtemp()
        try:
            fm = FeatureMemory(scope="user:stress", storage_dir=temp_dir)
            
            for i in range(50):
                fm.set("changing_value", i)
            
            fact = fm.get_with_history("changing_value")
            self.assertEqual(fact.current, 49)
            self.assertEqual(len(fact.history), 50)
        finally:
            shutil.rmtree(temp_dir)


# ==================== TEST RUNNER ====================

def run_all_tests():
    """Run all tests with verbose output."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestScopes,
        TestPolicies,
        TestUtils,
        TestFeatureMemory,
        TestShortTermMemory,
        TestLongTermMemory,
        TestSimpleMemoryManager,
        TestMemoryManagerIntegration,
        TestEdgeCases,
        TestStress,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("=" * 60)
    print("MEMORY FRAMEWORK TEST SUITE")
    print("=" * 60)
    
    success = run_all_tests()
    
    print("\n" + "=" * 60)
    if success:
        print("[PASS] ALL TESTS PASSED")
    else:
        print("[FAIL] SOME TESTS FAILED")
    print("=" * 60)
    
    sys.exit(0 if success else 1)
