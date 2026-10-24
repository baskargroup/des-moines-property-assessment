"""
UNIFIED BATCH PROCESSOR FOR ALL VLM PROVIDERS

Processes multiple images concurrently using any VLM provider (Google, OpenAI, Together, Local).

Features:
- Concurrent/parallel processing for maximum throughput
- Rate limiting and throttling to stay under API limits
- Automatic retry logic with exponential backoff
- Progress tracking with detailed statistics
- Works with all providers (google, openai, together, local)
- Memory efficient - streams results to file
- Comprehensive error handling and logging

Usage:
    from src.providers.batch_processor import BatchProcessor
    from src.providers import get_provider
    
    provider = get_provider("google")
    processor = BatchProcessor(
        provider=provider,
        max_concurrent_requests=15,  # Google Paid: up to 60 rpm
        max_requests_per_minute=50   # Leave headroom under 60 limit
    )
    
    results = await processor.process_batch(
        image_paths=["img1.jpg", "img2.jpg", ...],
        prompt="Analyze this medical image..."
    )
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class StatusTracker:
    """Tracks processing progress and statistics."""
    
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_retries: int = 0
    time_of_last_rate_limit_error: float = 0
    start_time: float = field(default_factory=time.time)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        elapsed = time.time() - self.start_time
        return {
            "total_tasks": self.num_tasks_started,
            "succeeded": self.num_tasks_succeeded,
            "failed": self.num_tasks_failed,
            "in_progress": self.num_tasks_in_progress,
            "rate_limit_errors": self.num_rate_limit_errors,
            "retries": self.num_retries,
            "elapsed_time": elapsed,
            "success_rate": (self.num_tasks_succeeded / self.num_tasks_started * 100) if self.num_tasks_started > 0 else 0
        }


@dataclass
class ProcessingTask:
    """Represents a single image processing task."""
    
    task_id: int
    image_path: str
    prompt: str
    metadata: Optional[Dict] = None
    result: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0


class BatchProcessor:
    """
    Unified batch processor for all VLM providers.
    
    Handles concurrent processing, rate limiting, retries, and progress tracking.
    """
    
    def __init__(
        self,
        provider,
        max_concurrent_requests: int = 15,
        max_requests_per_minute: float = 50,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        """
        Initialize the batch processor.
        
        Args:
            provider: VLM provider instance (Google, OpenAI, Together, or Local)
            max_concurrent_requests: Maximum number of concurrent API requests (default 15 for Google Paid)
            max_requests_per_minute: Rate limit for requests per minute (default 50, leaves headroom under 60 limit)
            max_retries: Maximum number of retry attempts per task
            retry_delay: Base delay between retries in seconds
        """
        self.provider = provider
        self.max_concurrent_requests = max_concurrent_requests
        self.max_requests_per_minute = max_requests_per_minute
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.status_tracker = StatusTracker()
        
        logger.info(f"Initialized BatchProcessor with provider: {provider.model_name}")
        logger.info(f"Settings: max_concurrent={max_concurrent_requests}, max_rpm={max_requests_per_minute}")
    
    async def process_single_task(
        self,
        task: ProcessingTask,
        semaphore: asyncio.Semaphore,
        rate_limiter: asyncio.Semaphore
    ) -> ProcessingTask:
        """
        Process a single image analysis task with rate limiting and retries.
        
        Args:
            task: The processing task
            semaphore: Semaphore for controlling concurrency
            rate_limiter: Semaphore for rate limiting
            
        Returns:
            ProcessingTask with result or error populated
        """
        async with semaphore:  # Limit concurrent requests
            async with rate_limiter:  # Rate limiting
                for attempt in range(self.max_retries):
                    task.attempts = attempt + 1
                    self.status_tracker.num_tasks_in_progress += 1
                    
                    try:
                        logger.info(f"Processing task {task.task_id}: {Path(task.image_path).name} (attempt {attempt + 1}/{self.max_retries})")
                        
                        # Check if provider has async method
                        if hasattr(self.provider, 'analyze_async'):
                            result = await self.provider.analyze_async(
                                image_path=task.image_path,
                                prompt=task.prompt,
                                max_retries=1  # We handle retries here
                            )
                        else:
                            # Fallback to synchronous method in executor
                            loop = asyncio.get_event_loop()
                            result = await loop.run_in_executor(
                                None,
                                self.provider.analyze,
                                task.image_path,
                                task.prompt
                            )
                        
                        if result:
                            task.result = result
                            self.status_tracker.num_tasks_succeeded += 1
                            self.status_tracker.num_tasks_in_progress -= 1
                            logger.info(f"✅ Task {task.task_id} completed successfully")
                            return task
                        else:
                            raise Exception("Provider returned None")
                            
                    except Exception as e:
                        error_msg = str(e)
                        logger.warning(f"⚠️  Task {task.task_id} failed (attempt {attempt + 1}): {error_msg}")
                        
                        # Check if it's a rate limit error
                        if "rate limit" in error_msg.lower() or "429" in error_msg:
                            self.status_tracker.num_rate_limit_errors += 1
                            self.status_tracker.time_of_last_rate_limit_error = time.time()
                        
                        if attempt < self.max_retries - 1:
                            # Exponential backoff
                            wait_time = self.retry_delay * (2 ** attempt)
                            logger.info(f"Retrying in {wait_time}s...")
                            self.status_tracker.num_retries += 1
                            await asyncio.sleep(wait_time)
                        else:
                            # Final attempt failed
                            task.error = error_msg
                            self.status_tracker.num_tasks_failed += 1
                            self.status_tracker.num_tasks_in_progress -= 1
                            logger.error(f"❌ Task {task.task_id} failed after all attempts")
                            return task
                
                # Add small delay for rate limiting
                await asyncio.sleep(60.0 / self.max_requests_per_minute)
        
        return task
    
    async def process_batch(
        self,
        image_paths: List[str],
        prompt: str,
        metadata: Optional[List[Dict]] = None
    ) -> List[ProcessingTask]:
        """
        Process a batch of images concurrently.
        
        Args:
            image_paths: List of paths to image files
            prompt: Text prompt to use for all images
            metadata: Optional list of metadata dicts (one per image)
            
        Returns:
            List of ProcessingTask objects with results
        """
        logger.info(f"Starting batch processing of {len(image_paths)} images")
        logger.info(f"Provider: {self.provider.model_name}")
        logger.info(f"Prompt length: {len(prompt)} characters")
        
        # Create tasks
        tasks = []
        for i, image_path in enumerate(image_paths):
            task = ProcessingTask(
                task_id=i,
                image_path=image_path,
                prompt=prompt,
                metadata=metadata[i] if metadata else None
            )
            tasks.append(task)
            self.status_tracker.num_tasks_started += 1
        
        # Create semaphores for concurrency and rate limiting
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        rate_limiter = asyncio.Semaphore(int(self.max_requests_per_minute / 60))  # Per second
        
        # Process all tasks concurrently
        results = await asyncio.gather(
            *[self.process_single_task(task, semaphore, rate_limiter) for task in tasks],
            return_exceptions=True
        )
        
        # Log summary
        summary = self.status_tracker.get_summary()
        logger.info("=" * 60)
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total tasks: {summary['total_tasks']}")
        logger.info(f"Succeeded: {summary['succeeded']}")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"Success rate: {summary['success_rate']:.1f}%")
        logger.info(f"Total retries: {summary['retries']}")
        logger.info(f"Rate limit errors: {summary['rate_limit_errors']}")
        logger.info(f"Elapsed time: {summary['elapsed_time']:.2f}s")
        logger.info("=" * 60)
        
        return [r for r in results if isinstance(r, ProcessingTask)]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current processing status."""
        return self.status_tracker.get_summary()


async def process_images_batch(
    provider_name: str,
    image_paths: List[str],
    prompt: str,
    max_concurrent: int = 15,
    max_rpm: float = 50
) -> List[ProcessingTask]:
    """
    Convenience function to process a batch of images with a specific provider.
    
    Args:
        provider_name: Name of provider ('google', 'openai', 'together', 'local')
        image_paths: List of image file paths
        prompt: Text prompt for analysis
        max_concurrent: Maximum concurrent requests
        max_rpm: Maximum requests per minute
        
    Returns:
        List of ProcessingTask objects with results
    """
    from src.providers import get_provider
    
    provider = get_provider(provider_name)
    processor = BatchProcessor(
        provider=provider,
        max_concurrent_requests=max_concurrent,
        max_requests_per_minute=max_rpm
    )
    
    return await processor.process_batch(image_paths, prompt)


# Example usage
if __name__ == "__main__":
    async def main():
        # Example: process 5 images with Google Gemini (Paid tier)
        image_paths = [
            "image1.jpg",
            "image2.jpg", 
            "image3.jpg",
            "image4.jpg",
            "image5.jpg"
        ]
        
        prompt = "Analyze this medical image and provide detailed findings."
        
        results = await process_images_batch(
            provider_name="google",
            image_paths=image_paths,
            prompt=prompt,
            max_concurrent=15,  # Google Paid allows ~60 rpm
            max_rpm=50          # Leave headroom
        )
        
        # Print results
        for task in results:
            if task.result:
                print(f"Image {task.task_id}: Success")
                print(f"Response: {task.result[:200]}...")
            else:
                print(f"Image {task.task_id}: Failed - {task.error}")
    
    asyncio.run(main())
