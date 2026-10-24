"""
Tests for centralized configuration
"""
import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import Config


class TestConfig(unittest.TestCase):
    """Test centralized configuration parameters"""
    
    def test_generation_temperature_exists(self):
        """Test that GENERATION_TEMPERATURE is defined"""
        self.assertTrue(hasattr(Config, 'GENERATION_TEMPERATURE'))
    
    def test_generation_temperature_is_zero(self):
        """Test that temperature is set to 0.0 for reproducibility"""
        self.assertEqual(Config.GENERATION_TEMPERATURE, 0.0)
    
    def test_generation_max_tokens_exists(self):
        """Test that GENERATION_MAX_TOKENS is defined"""
        self.assertTrue(hasattr(Config, 'GENERATION_MAX_TOKENS'))
    
    def test_generation_max_tokens_value(self):
        """Test that max tokens is reasonable"""
        self.assertEqual(Config.GENERATION_MAX_TOKENS, 8000)
        self.assertGreater(Config.GENERATION_MAX_TOKENS, 1000)
    
    def test_generation_top_p_exists(self):
        """Test that GENERATION_TOP_P is defined"""
        self.assertTrue(hasattr(Config, 'GENERATION_TOP_P'))
    
    def test_generation_top_p_value(self):
        """Test that top_p is in valid range"""
        self.assertGreaterEqual(Config.GENERATION_TOP_P, 0.0)
        self.assertLessEqual(Config.GENERATION_TOP_P, 1.0)
    
    def test_generation_top_k_exists(self):
        """Test that GENERATION_TOP_K is defined"""
        self.assertTrue(hasattr(Config, 'GENERATION_TOP_K'))
    
    def test_generation_top_k_value(self):
        """Test that top_k is positive"""
        self.assertGreater(Config.GENERATION_TOP_K, 0)
    
    def test_generation_repetition_penalty_exists(self):
        """Test that GENERATION_REPETITION_PENALTY is defined"""
        self.assertTrue(hasattr(Config, 'GENERATION_REPETITION_PENALTY'))
    
    def test_model_names_exist(self):
        """Test that all model names are defined"""
        self.assertTrue(hasattr(Config, 'MODEL_GOOGLE'))
        self.assertTrue(hasattr(Config, 'MODEL_OPENAI'))
        self.assertTrue(hasattr(Config, 'MODEL_TOGETHER'))
        self.assertTrue(hasattr(Config, 'MODEL_LOCAL'))
    
    def test_model_names_not_empty(self):
        """Test that model names are not empty strings"""
        self.assertIsInstance(Config.MODEL_GOOGLE, str)
        self.assertIsInstance(Config.MODEL_OPENAI, str)
        self.assertIsInstance(Config.MODEL_TOGETHER, str)
        self.assertIsInstance(Config.MODEL_LOCAL, str)
        
        self.assertGreater(len(Config.MODEL_GOOGLE), 0)
        self.assertGreater(len(Config.MODEL_OPENAI), 0)
        self.assertGreater(len(Config.MODEL_TOGETHER), 0)
        self.assertGreater(len(Config.MODEL_LOCAL), 0)


if __name__ == '__main__':
    unittest.main()
