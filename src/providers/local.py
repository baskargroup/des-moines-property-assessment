import requests
import aiohttp
import asyncio
import json
import base64
import logging
from src.providers.base import BaseVLM
from src.config import Config

logger = logging.getLogger(__name__)

class LocalVLM(BaseVLM):
    """Local Ollama Vision Language Model provider."""
    
    def __init__(self, model_name=Config.MODEL_LOCAL):
        """
        Initialize the local Ollama VLM provider.
        
        Args:
            model_name (str): Name of the Ollama model to use (default from config)
        """
        super().__init__(model_name)
        self.api_url = f"{Config.OLLAMA_BASE_URL}/api/generate"
        logger.info(f"Initialized LocalVLM with model: {model_name}")
        logger.info(f"API URL: {self.api_url}")

    def analyze(self, image_path, prompt):
        """
        Analyze an image using a local Ollama vision model.
        
        Args:
            image_path (str): Path to the image file
            prompt (str): Text prompt for the model
            
        Returns:
            str: Model response text, or None if request failed
        """
        logger.info(f"LocalVLM.analyze() called with model: {self.model_name}")
        logger.info(f"Image path: {image_path}")
        logger.info(f"Prompt length: {len(prompt)} characters")
        
        # Encode image
        try:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            logger.info(f"Base64 image length: {len(img_b64)} characters")
        except Exception as e:
            logger.error(f"Failed to read/encode image: {e}")
            return None

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "format": "json"  # Enforce JSON output for structured data
        }

        try:
            logger.info("Sending API request to Ollama...")
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            
            response_content = response.json().get("response", "")
            logger.info(f"✅ API call successful! Response length: {len(response_content)} characters")
            logger.info(f"Response preview (first 200 chars): {response_content[:200]}")
            
            return response_content
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            logger.error(f"❌ Error calling Local VLM: {error_msg}", exc_info=True)
            print(f"Error calling Local VLM: {error_msg}")
            return None

    async def analyze_async(self, image_path, prompt, max_retries=3, retry_delay=2):
        """
        Async version of analyze for concurrent processing with retry logic.
        
        Args:
            image_path (str): Path to the image file
            prompt (str): Text prompt for the model
            max_retries (int): Maximum number of retry attempts
            retry_delay (int): Base delay between retries in seconds
            
        Returns:
            str: Model response text, or None if all retries failed
        """
        logger.info(f"LocalVLM.analyze_async() called with model: {self.model_name}")
        logger.info(f"Image path: {image_path}")
        
        # Encode image
        try:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            logger.info(f"Base64 image length: {len(img_b64)} characters")
        except Exception as e:
            logger.error(f"Failed to read/encode image: {e}")
            return None

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "format": "json"
        }

        for attempt in range(max_retries):
            try:
                logger.info(f"Sending async API request to Ollama (attempt {attempt + 1}/{max_retries})...")
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.api_url, json=payload) as response:
                        response.raise_for_status()
                        response_json = await response.json()
                        response_content = response_json.get("response", "")
                        
                logger.info(f"✅ Async API call successful! Response length: {len(response_content)} characters")
                logger.info(f"Response preview (first 200 chars): {response_content[:200]}")
                
                return response_content
                
            except aiohttp.ClientResponseError as e:
                error_msg = str(e)
                logger.error(f"❌ Local VLM async error (attempt {attempt + 1}/{max_retries}): {error_msg}")
                
                if attempt < max_retries - 1:
                    wait_time = retry_delay
                    logger.warning(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    return None
                    
            except Exception as e:
                error_msg = str(e)
                logger.error(f"❌ Local VLM async error (attempt {attempt + 1}/{max_retries}): {error_msg}")
                
                if attempt < max_retries - 1:
                    wait_time = retry_delay
                    logger.warning(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    return None
        
        return None

