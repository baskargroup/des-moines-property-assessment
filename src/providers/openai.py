from openai import OpenAI, AsyncOpenAI
from src.providers.base import BaseVLM
from src.config import Config
from src.data_loader import DataLoader
import logging
import asyncio
import time

logger = logging.getLogger(__name__)

class OpenAIVLM(BaseVLM):
    """OpenAI Vision Language Model provider using GPT-4 Vision or GPT-5 models."""
    
    def __init__(self, model_name=Config.MODEL_OPENAI):
        """
        Initialize the OpenAI VLM provider.
        
        Args:
            model_name (str): Name of the OpenAI model to use (default from config)
        """
        super().__init__(model_name)
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.async_client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)

    def analyze(self, image_path, prompt):
        """
        Analyze an image using OpenAI's vision models (GPT-4 Vision or GPT-5).
        
        Args:
            image_path (str): Path to the image file
            prompt (str): Text prompt for the model
            
        Returns:
            str: Model response text, or None if request failed
        """
        logger.info(f"OpenAI.analyze() called with model: {self.model_name}")
        logger.info(f"Image path: {image_path}")
        logger.info(f"Prompt length: {len(prompt)} characters")
        
        base64_image = DataLoader.encode_image(image_path)
        if not base64_image:
            logger.error(f"Failed to encode image: {image_path}")
            return "Error: Image not found"

        try:
            # GPT-5 uses max_completion_tokens instead of max_tokens
            is_gpt5 = "gpt-5" in self.model_name.lower()
            logger.info(f"Is GPT-5: {is_gpt5}")
            
            api_params = {
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
                                },
                            },
                        ],
                    }
                ],
            }
            
            logger.info(f"Base64 image length: {len(base64_image)} characters (first 50: {base64_image[:50]}...)")
            
            if is_gpt5:
                api_params["max_completion_tokens"] = Config.GENERATION_MAX_TOKENS
                logger.info("Using max_completion_tokens for GPT-5")
                
                # Enable reasoning for GPT-5.2+
                if "5.2" in self.model_name:
                    api_params["reasoning_effort"] = "low"
                    logger.info("Enabled reasoning_effort: low for GPT-5.2")
            else:
                api_params["max_tokens"] = Config.GENERATION_MAX_TOKENS
                logger.info("Using max_tokens for non-GPT-5 model")
            
            # Add temperature from centralized config
            # Note: GPT-5/5.1/5.2 models ONLY support temperature=1.0 (default)
            # Other models can use custom temperature
            if "gpt-5" in self.model_name.lower():
                # GPT-5 models only support temperature=1.0, so don't set it
                # (or set to 1.0 explicitly)
                api_params["temperature"] = 1.0
                logger.info("Using temperature: 1.0 (GPT-5 only supports default temperature)")
            else:
                api_params["temperature"] = Config.GENERATION_TEMPERATURE
                logger.info(f"Using temperature: {Config.GENERATION_TEMPERATURE}")
            
            logger.info(f"Sending API request to OpenAI...")
            logger.info(f"API params: model={api_params['model']}, token_param={'max_completion_tokens' if is_gpt5 else 'max_tokens'}, reasoning_effort={api_params.get('reasoning_effort', 'N/A')}")
            
            response = self.client.chat.completions.create(**api_params)
            
            response_content = response.choices[0].message.content
            logger.info(f"✅ API call successful! Response length: {len(response_content)} characters")
            logger.info(f"Response preview (first 200 chars): {response_content[:200]}")
            
            return response_content
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Error calling OpenAI API: {error_msg}", exc_info=True)
            print(f"Error calling OpenAI: {error_msg}")
            # Check if it's a model not found error
            if "model" in error_msg.lower() and ("not found" in error_msg.lower() or "does not exist" in error_msg.lower()):
                logger.warning(f"Model '{self.model_name}' may not be available")
                print(f"⚠️  Model '{self.model_name}' may not be available. Try 'gpt-4o' instead.")
            elif "max_tokens" in error_msg.lower() or "max_completion_tokens" in error_msg.lower():
                logger.error("Token parameter error - this should be fixed!")
            return None

    async def analyze_async(self, image_path, prompt, max_retries=3, retry_delay=2):
        """
        Async version of analyze for concurrent processing.
        
        Includes retry logic for rate limiting and transient errors.
        
        Args:
            image_path (str): Path to the image file
            prompt (str): Text prompt for the model
            max_retries (int): Maximum number of retry attempts
            retry_delay (int): Base delay between retries in seconds
            
        Returns:
            str: Model response text, or None if all retries failed
        """
        logger.info(f"OpenAI.analyze_async() called with model: {self.model_name}")
        logger.info(f"Image path: {image_path}")
        
        base64_image = DataLoader.encode_image(image_path)
        if not base64_image:
            logger.error(f"Failed to encode image: {image_path}")
            return "Error: Image not found"

        for attempt in range(max_retries):
            try:
                is_gpt5 = "gpt-5" in self.model_name.lower()
                
                api_params = {
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
                                    },
                                },
                            ],
                        }
                    ],
                }
                
                if is_gpt5:
                    api_params["max_completion_tokens"] = Config.GENERATION_MAX_TOKENS
                    if "5.2" in self.model_name:
                        api_params["reasoning_effort"] = "low"
                else:
                    api_params["max_tokens"] = Config.GENERATION_MAX_TOKENS
                
                # GPT-5 models only support temperature=1.0
                if "gpt-5" in self.model_name.lower():
                    api_params["temperature"] = 1.0
                else:
                    api_params["temperature"] = Config.GENERATION_TEMPERATURE
                
                logger.info(f"Sending async API request (attempt {attempt + 1}/{max_retries})...")
                response = await self.async_client.chat.completions.create(**api_params)
                
                response_content = response.choices[0].message.content
                logger.info(f"✅ Async API call successful! Response length: {len(response_content)} characters")
                
                return response_content
                
            except Exception as e:
                error_msg = str(e)
                error_str = error_msg.lower()
                
                # Check for rate limit errors
                is_rate_limit = (
                    "429" in error_str or 
                    "rate limit" in error_str or
                    "rate_limit" in error_str
                )
                
                if is_rate_limit and attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(f"⚠️  OpenAI rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                elif attempt < max_retries - 1:
                    # Other errors - shorter wait
                    wait_time = retry_delay
                    logger.warning(f"⚠️  OpenAI error (attempt {attempt + 1}/{max_retries}): {error_msg}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Final attempt failed
                    logger.error(f"❌ OpenAI async call failed after all attempts: {error_msg}")
                    return None
        
        return None
