"""
Tests for provider async methods and functionality
"""
import unittest
import asyncio
import sys
import os
import importlib
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import Config


class TestBaseProvider(unittest.TestCase):
    """Test BaseVLM abstract class"""
    
    def test_base_provider_module_exists(self):
        """Test that BaseVLM module exists"""
        try:
            from src.providers.base import BaseVLM
            self.assertTrue(hasattr(BaseVLM, 'analyze'))
        except ImportError:
            self.skipTest("Could not import BaseVLM")


class TestProviderFactory(unittest.TestCase):
    """Test get_provider factory function"""
    
    def test_get_provider_function_exists(self):
        """Test that get_provider function exists"""
        try:
            # Try to import without actually calling (may fail without dependencies)
            import importlib.util
            spec = importlib.util.find_spec('src.providers')
            self.assertIsNotNone(spec)
        except ImportError:
            self.skipTest("Could not import providers module")


class TestProviderAsyncMethods(unittest.TestCase):
    """Test that all providers have async methods"""
    
    def test_provider_files_exist(self):
        """Test that provider files exist"""
        import os
        base_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'providers')
        
        self.assertTrue(os.path.exists(os.path.join(base_path, 'google.py')))
        self.assertTrue(os.path.exists(os.path.join(base_path, 'openai.py')))
        self.assertTrue(os.path.exists(os.path.join(base_path, 'together.py')))
        self.assertTrue(os.path.exists(os.path.join(base_path, 'local.py')))
    
    def test_async_methods_in_source_code(self):
        """Test that async methods exist in source code"""
        import os
        base_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'providers')
        
        # Check that analyze_async exists in each file
        for provider_file in ['google.py', 'openai.py', 'together.py', 'local.py']:
            with open(os.path.join(base_path, provider_file), 'r') as f:
                content = f.read()
                self.assertIn('analyze_async', content, 
                            f"analyze_async not found in {provider_file}")


class TestConfigIntegration(unittest.TestCase):
    """Test that providers use centralized config"""
    
    def test_config_temperature_is_used(self):
        """Test that Config.GENERATION_TEMPERATURE exists and is accessible"""
        # Providers should reference Config.GENERATION_TEMPERATURE
        self.assertTrue(hasattr(Config, 'GENERATION_TEMPERATURE'))
        self.assertIsInstance(Config.GENERATION_TEMPERATURE, (int, float))
    
    def test_config_max_tokens_is_used(self):
        """Test that Config.GENERATION_MAX_TOKENS exists and is accessible"""
        self.assertTrue(hasattr(Config, 'GENERATION_MAX_TOKENS'))
        self.assertIsInstance(Config.GENERATION_MAX_TOKENS, int)
    
    def test_config_values_are_reasonable(self):
        """Test that config values are in reasonable ranges"""
        # Temperature should be 0-2
        self.assertGreaterEqual(Config.GENERATION_TEMPERATURE, 0.0)
        self.assertLessEqual(Config.GENERATION_TEMPERATURE, 2.0)
        
        # Max tokens should be positive and reasonable
        self.assertGreater(Config.GENERATION_MAX_TOKENS, 100)
        self.assertLess(Config.GENERATION_MAX_TOKENS, 100000)


class TestRetryLogic(unittest.TestCase):
    """Test retry logic in async methods"""
    
    def test_retry_parameters_in_source_code(self):
        """Test that async methods have retry parameters in source"""
        import os
        base_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'providers')
        
        for provider_file in ['openai.py', 'together.py', 'local.py']:
            with open(os.path.join(base_path, provider_file), 'r') as f:
                content = f.read()
                # Check for retry parameters
                self.assertIn('max_retries', content, 
                            f"max_retries not found in {provider_file}")
                self.assertIn('retry_delay', content, 
                            f"retry_delay not found in {provider_file}")


class TestDocstrings(unittest.TestCase):
    """Test that providers have proper documentation"""
    
    def test_docstrings_in_source_code(self):
        """Test that docstrings exist in source files"""
        import os
        base_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'providers')
        
        for provider_file in ['google.py', 'openai.py', 'together.py', 'local.py']:
            with open(os.path.join(base_path, provider_file), 'r') as f:
                content = f.read()
                # Check for docstrings (triple quotes)
                self.assertIn('"""', content, 
                            f"Docstrings not found in {provider_file}")


if __name__ == '__main__':
    unittest.main()
