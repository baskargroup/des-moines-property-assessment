import base64
import PIL.Image
import time
import asyncio
from src.providers.base import BaseVLM
from src.config import Config

# Try to import the new google-genai package first (if user has it), fallback to google-generativeai
try:
    from google import genai
    from google.genai import types
    HAS_NEW_GOOGLE_API = True
except ImportError:
    try:
        import google.generativeai as genai
        HAS_NEW_GOOGLE_API = False
    except ImportError:
        genai = None
        HAS_NEW_GOOGLE_API = None
        print("Warning: Google Generative AI package not found. Install with: pip install google-generativeai")

class GoogleVLM(BaseVLM):
    """Google Gemini Vision Language Model provider with automatic API version detection."""

    def __init__(self, model_name=Config.MODEL_GOOGLE):
        """
        Initialize the Google Gemini VLM provider.
        
        Automatically detects and uses either the new google-genai package
        or the legacy google-generativeai package.
        
        Args:
            model_name (str): Name of the Gemini model to use (default from config)
            
        Raises:
            ImportError: If no Google Generative AI package is installed
        """
        super().__init__(model_name)
        
        if genai is None:
            raise ImportError("Google Generative AI package not found. Install with: pip install google-generativeai or pip install google-genai")
        
        if HAS_NEW_GOOGLE_API:
            # New API format (google-genai package)
            self.client = genai.Client(
                api_key=Config.GOOGLE_API_KEY,
                http_options={'api_version': 'v1alpha'}
            )
            self.use_new_api = True
        else:
            # Standard API format (google-generativeai package)
            genai.configure(api_key=Config.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel(model_name)
            self.use_new_api = False

    def analyze(self, image_path, prompt, max_retries=3, retry_delay=2):
        """
        Analyze image with retry logic for rate limiting and transient errors.
        
        Google Gemini API rate limits:
        - Free tier: ~15 requests/minute
        - Paid tier: ~60 requests/minute
        
        This function implements exponential backoff retry for rate limit errors.
        
        Args:
            image_path (str): Path to the image file
            prompt (str): Text prompt for the model
            max_retries (int): Maximum number of retry attempts
            retry_delay (int): Base delay between retries in seconds
            
        Returns:
            str: Model response text, or None if all retries failed
        """
        return self.analyze_with_temperature(
            image_path=image_path,
            prompt=prompt,
            temperature=Config.GENERATION_TEMPERATURE,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
    
    def analyze_with_temperature(self, image_path, prompt, temperature=0.0, max_retries=3, retry_delay=2):
        """
        Analyze image with custom temperature and retry logic for rate limiting and transient errors.
        
        Args:
            image_path (str): Path to image file
            prompt (str): Text prompt for the model
            temperature (float): Temperature for generation (0.0 = deterministic, higher = more random)
            max_retries (int): Maximum number of retry attempts
            retry_delay (int): Base delay between retries in seconds
        
        Returns:
            str: Model response text, or None if all retries failed
        """
        import re
        
        for attempt in range(max_retries):
            try:
                if self.use_new_api:
                    # New API format
                    with open(image_path, "rb") as f:
                        image_bytes = f.read()
                    
                    # New API format - try with max_output_tokens if supported
                    try:
                        # Try with generation config (may not be supported in all API versions)
                        generation_config = types.GenerationConfig(
                            max_output_tokens=Config.GENERATION_MAX_TOKENS,
                            temperature=temperature,
                            top_p=Config.GENERATION_TOP_P,
                            top_k=Config.GENERATION_TOP_K
                        )
                        response = self.client.models.generate_content(
                            model=self.model_name,
                            contents=[
                                types.Content(
                                    parts=[
                                        types.Part(text=prompt),
                                        types.Part(
                                            inline_data=types.Blob(
                                                mime_type="image/jpeg",
                                                data=image_bytes,
                                            )
                                        )
                                    ]
                                )
                            ],
                            config=generation_config
                        )
                    except (TypeError, AttributeError):
                        # Fallback if config parameter not supported
                        response = self.client.models.generate_content(
                            model=self.model_name,
                            contents=[
                                types.Content(
                                    parts=[
                                        types.Part(text=prompt),
                                        types.Part(
                                            inline_data=types.Blob(
                                                mime_type="image/jpeg",
                                                data=image_bytes,
                                            )
                                        )
                                    ]
                                )
                            ]
                        )
                    return response.text
                else:
                    # Standard API format
                    img = PIL.Image.open(image_path)
                    # Add generation config from centralized Config
                    generation_config = {
                        "max_output_tokens": Config.GENERATION_MAX_TOKENS,
                        "temperature": temperature,
                        "top_p": Config.GENERATION_TOP_P,
                        "top_k": Config.GENERATION_TOP_K
                    }
                    response = self.model.generate_content(
                        [prompt, img],
                        generation_config=generation_config
                    )
                    return response.text
            except Exception as e:
                error_str = str(e).lower()
                error_msg = str(e)
                # Check for rate limit errors (429, quota, rate limit)
                is_rate_limit = (
                    "429" in error_str or 
                    "rate limit" in error_str or 
                    "quota" in error_str or
                    "resource_exhausted" in error_str
                )
                
                if is_rate_limit and attempt < max_retries - 1:
                    # Try to extract retry_delay from error message if available
                    wait_time = retry_delay * (2 ** attempt)
                    
                    # Parse retry_delay from error message (format: "Please retry in X.XXs" or "retry_delay { seconds: XX }")
                    retry_match = re.search(r'retry in ([\d.]+)s', error_msg, re.IGNORECASE)
                    if retry_match:
                        api_wait_time = float(retry_match.group(1))
                        # Use API's suggested wait time, but add a small buffer
                        wait_time = api_wait_time + 1
                        print(f"⚠️  Google API rate limit hit (attempt {attempt + 1}/{max_retries}). API suggests waiting {api_wait_time}s. Waiting {wait_time}s before retry...")
                    else:
                        # Also check for "retry_delay { seconds: XX }" format
                        retry_match = re.search(r'seconds:\s*([\d.]+)', error_msg)
                        if retry_match:
                            api_wait_time = float(retry_match.group(1))
                            wait_time = api_wait_time + 1
                            print(f"⚠️  Google API rate limit hit (attempt {attempt + 1}/{max_retries}). API suggests waiting {api_wait_time}s. Waiting {wait_time}s before retry...")
                        else:
                            # Exponential backoff: wait longer for each retry
                            wait_time = retry_delay * (2 ** attempt)
                            print(f"⚠️  Google API rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting {wait_time}s before retry...")
                    
                    time.sleep(wait_time)
                    continue
                else:
                    # Non-rate-limit error or final attempt
                    if attempt == max_retries - 1:
                        print(f"❌ Error calling Google Gemini (final attempt): {e}")
                    else:
                        print(f"⚠️  Error calling Google Gemini (attempt {attempt + 1}/{max_retries}): {e}")
                    return None
        
        return None

    async def analyze_async(self, image_path, prompt, temperature=None, max_retries=3, retry_delay=2):
        """
        Async version of analyze_with_temperature for concurrent processing.
        
        Note: Google's genai library doesn't have native async support, so this wraps
        the synchronous method in an executor to avoid blocking the event loop.
        
        Args:
            image_path (str): Path to image file
            prompt (str): Text prompt for the model
            temperature (float): Temperature for generation (None uses config default)
            max_retries (int): Maximum number of retry attempts
            retry_delay (int): Base delay between retries in seconds
        
        Returns:
            str: Model response text, or None if all retries failed
        """
        if temperature is None:
            temperature = Config.GENERATION_TEMPERATURE
        
        # Run the synchronous analyze_with_temperature in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.analyze_with_temperature,
            image_path,
            prompt,
            temperature,
            max_retries,
            retry_delay
        )
        return result
