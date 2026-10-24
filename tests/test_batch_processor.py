"""
Tests for batch processor functionality
"""
import unittest
import asyncio
import sys
import os
from unittest.mock import Mock, AsyncMock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.providers.batch_processor import (
    BatchProcessor,
    StatusTracker,
    ProcessingTask
)


class MockProvider:
    """Mock provider for testing"""
    def __init__(self, model_name="mock-model"):
        self.model_name = model_name
    
    async def analyze_async(self, image_path, prompt, max_retries=3):
        """Mock async analyze"""
        await asyncio.sleep(0.01)  # Simulate API call
        return f"Mock analysis of {image_path}"


class FailingProvider:
    """Mock provider that fails for testing retry logic"""
    def __init__(self, model_name="failing-model"):
        self.model_name = model_name
        self.call_count = 0
    
    async def analyze_async(self, image_path, prompt, max_retries=3):
        """Mock async analyze that fails"""
        self.call_count += 1
        await asyncio.sleep(0.01)
        raise Exception("Mock API error")


class TestStatusTracker(unittest.TestCase):
    """Test StatusTracker functionality"""
    
    def test_status_tracker_initialization(self):
        """Test that StatusTracker initializes correctly"""
        tracker = StatusTracker()
        self.assertEqual(tracker.num_tasks_started, 0)
        self.assertEqual(tracker.num_tasks_succeeded, 0)
        self.assertEqual(tracker.num_tasks_failed, 0)
    
    def test_status_tracker_summary(self):
        """Test that get_summary returns correct dict"""
        tracker = StatusTracker()
        tracker.num_tasks_started = 10
        tracker.num_tasks_succeeded = 8
        tracker.num_tasks_failed = 2
        
        summary = tracker.get_summary()
        self.assertEqual(summary['total_tasks'], 10)
        self.assertEqual(summary['succeeded'], 8)
        self.assertEqual(summary['failed'], 2)
        self.assertEqual(summary['success_rate'], 80.0)


class TestProcessingTask(unittest.TestCase):
    """Test ProcessingTask functionality"""
    
    def test_processing_task_creation(self):
        """Test that ProcessingTask creates correctly"""
        task = ProcessingTask(
            task_id=1,
            image_path="test.jpg",
            prompt="Test prompt",
            metadata={"key": "value"}
        )
        self.assertEqual(task.task_id, 1)
        self.assertEqual(task.image_path, "test.jpg")
        self.assertEqual(task.prompt, "Test prompt")
        self.assertEqual(task.metadata, {"key": "value"})
        self.assertIsNone(task.result)
        self.assertIsNone(task.error)


class TestBatchProcessor(unittest.TestCase):
    """Test BatchProcessor functionality"""
    
    def test_batch_processor_initialization(self):
        """Test that BatchProcessor initializes correctly"""
        provider = MockProvider()
        processor = BatchProcessor(
            provider=provider,
            max_concurrent_requests=5,
            max_requests_per_minute=30
        )
        self.assertEqual(processor.provider.model_name, "mock-model")
        self.assertEqual(processor.max_concurrent_requests, 5)
        self.assertEqual(processor.max_requests_per_minute, 30)
    
    def test_batch_processor_process_single_image(self):
        """Test processing a single image"""
        async def run_test():
            provider = MockProvider()
            processor = BatchProcessor(provider=provider)
            
            results = await processor.process_batch(
                image_paths=["test1.jpg"],
                prompt="Analyze this"
            )
            
            self.assertEqual(len(results), 1)
            self.assertIsNotNone(results[0].result)
            self.assertIn("test1.jpg", results[0].result)
            
            # Check status tracker
            status = processor.get_status()
            self.assertEqual(status['total_tasks'], 1)
            self.assertEqual(status['succeeded'], 1)
            self.assertEqual(status['failed'], 0)
        
        asyncio.run(run_test())
    
    def test_batch_processor_process_multiple_images(self):
        """Test processing multiple images"""
        async def run_test():
            provider = MockProvider()
            processor = BatchProcessor(
                provider=provider,
                max_concurrent_requests=2
            )
            
            image_paths = ["test1.jpg", "test2.jpg", "test3.jpg"]
            results = await processor.process_batch(
                image_paths=image_paths,
                prompt="Analyze this"
            )
            
            self.assertEqual(len(results), 3)
            
            # All should succeed
            for result in results:
                self.assertIsNotNone(result.result)
                self.assertIsNone(result.error)
            
            # Check status
            status = processor.get_status()
            self.assertEqual(status['total_tasks'], 3)
            self.assertEqual(status['succeeded'], 3)
            self.assertEqual(status['failed'], 0)
            self.assertEqual(status['success_rate'], 100.0)
        
        asyncio.run(run_test())
    
    def test_batch_processor_handles_failures(self):
        """Test that processor handles failures correctly"""
        async def run_test():
            provider = FailingProvider()
            processor = BatchProcessor(
                provider=provider,
                max_retries=2,
                retry_delay=0.01  # Fast for testing
            )
            
            results = await processor.process_batch(
                image_paths=["test1.jpg"],
                prompt="Analyze this"
            )
            
            self.assertEqual(len(results), 1)
            self.assertIsNone(results[0].result)
            self.assertIsNotNone(results[0].error)
            
            # Check status
            status = processor.get_status()
            self.assertEqual(status['total_tasks'], 1)
            self.assertEqual(status['succeeded'], 0)
            self.assertEqual(status['failed'], 1)
            self.assertEqual(status['success_rate'], 0.0)
        
        asyncio.run(run_test())
    
    def test_batch_processor_with_metadata(self):
        """Test processing with metadata"""
        async def run_test():
            provider = MockProvider()
            processor = BatchProcessor(provider=provider)
            
            metadata = [
                {"filename": "test1.jpg", "category": "A"},
                {"filename": "test2.jpg", "category": "B"}
            ]
            
            results = await processor.process_batch(
                image_paths=["test1.jpg", "test2.jpg"],
                prompt="Analyze",
                metadata=metadata
            )
            
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].metadata["category"], "A")
            self.assertEqual(results[1].metadata["category"], "B")
        
        asyncio.run(run_test())


if __name__ == '__main__':
    unittest.main()
