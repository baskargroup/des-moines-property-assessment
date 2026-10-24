import os
import base64
import requests
import aiohttp
import asyncio
import logging
from src.providers.base import BaseVLM
from src.config import Config
from src.data_loader import DataLoader

logger = logging.getLogger(__name__)

class TogetherVLM(BaseVLM):
    """Together AI Vision Language Model provider using Llama Vision models."""

    def __init__(self, model_name=Config.MODEL_TOGETHER):
        """
        Initialize the Together AI VLM provider.
        
        Args:
            model_name (str): Name of the Together AI model to use (default from config)
        """
        super().__init__(model_name)
        self.api_key = Config.TOGETHER_API_KEY
        self.url = "https://api.together.xyz/v1/chat/completions"
        logger.info(f"Initialized TogetherVLM with model: {model_name}")

    def analyze(self, image_path, prompt):
        """
        Analyze an image using Together AI's vision model.
        
        Together AI Llama Vision uses OpenAI-compatible format for vision models.
        
        Args:
            image_path (str): Path to the image file
            prompt (str): Text prompt for the model
            
        Returns:
            str: Model response text, or None if request failed
        """
        logger.info(f"TogetherVLM.analyze() called with model: {self.model_name}")
        logger.info(f"Image path: {image_path}")
        logger.info(f"Prompt length: {len(prompt)} characters")
        
        base64_image = DataLoader.encode_image(image_path)
        if not base64_image:
            logger.error(f"Failed to encode image: {image_path}")
            return "Error: Image not found"

        logger.info(f"Base64 image length: {len(base64_image)} characters")

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": Config.GENERATION_MAX_TOKENS,
            "temperature": Config.GENERATION_TEMPERATURE,
            "top_p": Config.GENERATION_TOP_P,
            "top_k": Config.GENERATION_TOP_K,
            "repetition_penalty": Config.GENERATION_REPETITION_PENALTY,
            "stop": ["<|eot_id|>"]
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            logger.info("Sending API request to Together AI...")
            response = requests.post(self.url, json=payload, headers=headers)
            response.raise_for_status()
            
            response_content = response.json()['choices'][0]['message']['content']
            logger.info(f"✅ API call successful! Response length: {len(response_content)} characters")
            logger.info(f"Response preview (first 200 chars): {response_content[:200]}")
            
            return response_content
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Error calling Together AI: {error_msg}", exc_info=True)
            print(f"Error calling Together AI: {error_msg}")
            # Print detailed error if available
            try:
                error_detail = response.text
                logger.error(f"API response: {error_detail}")
                print(f"API response: {error_detail}")
            except:
                pass
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
        logger.info(f"TogetherVLM.analyze_async() called with model: {self.model_name}")
        logger.info(f"Image path: {image_path}")
        
        base64_image = DataLoader.encode_image(image_path)
        if not base64_image:
            logger.error(f"Failed to encode image: {image_path}")
            return "Error: Image not found"

        logger.info(f"Base64 image length: {len(base64_image)} characters")

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": Config.GENERATION_MAX_TOKENS,
            "temperature": Config.GENERATION_TEMPERATURE,
            "top_p": Config.GENERATION_TOP_P,
            "top_k": Config.GENERATION_TOP_K,
            "repetition_penalty": Config.GENERATION_REPETITION_PENALTY,
            "stop": ["<|eot_id|>"]
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        for attempt in range(max_retries):
            try:
                logger.info(f"Sending async API request (attempt {attempt + 1}/{max_retries})...")
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.url, json=payload, headers=headers) as response:
                        response.raise_for_status()
                        response_json = await response.json()
                        response_content = response_json['choices'][0]['message']['content']
                        
                logger.info(f"✅ Async API call successful! Response length: {len(response_content)} characters")
                logger.info(f"Response preview (first 200 chars): {response_content[:200]}")
                
                return response_content
                
            except aiohttp.ClientResponseError as e:
                error_msg = str(e)
                
                # Check for rate limit errors
                is_rate_limit = e.status == 429 or "rate limit" in error_msg.lower()
                
                if is_rate_limit and attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(f"⚠️  Together AI rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                elif attempt < max_retries - 1:
                    wait_time = retry_delay
                    logger.warning(f"⚠️  Together AI error (attempt {attempt + 1}/{max_retries}): {error_msg}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"❌ Together AI async call failed after all attempts: {error_msg}")
                    return None
                    
            except Exception as e:
                error_msg = str(e)
                logger.error(f"❌ Together AI async error (attempt {attempt + 1}/{max_retries}): {error_msg}")
                
                if attempt < max_retries - 1:
                    wait_time = retry_delay
                    logger.warning(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    return None
        
        return None
