import os
from dotenv import load_dotenv

# Load .env file only from src directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

# Load .env file from src directory only
if os.path.exists(ENV_FILE_SRC):
    load_dotenv(ENV_FILE_SRC)
else:
    # Fallback to default behavior (current directory) if src/.env doesn't exist
    load_dotenv()

class Config:
    # Base directory - project root
    BASE_DIR = BASE_DIR
    
    # Paths - Support both old "Data" structure and new "data" structure
    # Use absolute paths based on BASE_DIR
    
    # Check if Data folder exists (old structure) or data folder (new structure)
    if os.path.exists(os.path.join(BASE_DIR, "Data")):
        DATA_DIR = os.path.join(BASE_DIR, "Data")
        RAW_IMAGES_DIR = os.path.join(BASE_DIR, "Data", "extractedimages")
        ANNOTATIONS_DIR = os.path.join(BASE_DIR, "Data", "annotation")
        DATA_50_VLM_DIR = os.path.join(BASE_DIR, "Data", "50-VLM")
    else:
        DATA_DIR = os.path.join(BASE_DIR, "data")
        RAW_IMAGES_DIR = os.path.join(BASE_DIR, "data", "raw", "images")
        ANNOTATIONS_DIR = os.path.join(BASE_DIR, "data", "annotation")
        DATA_50_VLM_DIR = os.path.join(BASE_DIR, "data", "50-VLM")
    
    PROMPTS_DIR = os.path.join(BASE_DIR, "data", "prompts")
    OUTPUTS_DIR = os.path.join(BASE_DIR, "data", "outputs")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    # Support both TOGETHER_API_KEY and TOGETHER_AI_API_KEY for compatibility
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY") or os.getenv("TOGETHER_AI_API_KEY")

    # Model Names - Updated to Latest Versions (2024-2025)
    
    # OpenAI: GPT-5 model
    MODEL_OPENAI = "gpt-5.2-2025-12-11"  # Updated to GPT-5.2 thinking model
    # Alternatives: "gpt-4o", "gpt-4-turbo", "gpt-4o-2024-08-06"
    
    # Google: Gemini 3 Flash Preview (fast)
    MODEL_GOOGLE = "gemini-3-flash-preview"  # Fast Gemini 3 model with vision
    # Alternatives if preview not available: "gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.5-pro-latest"
    
    # Together AI: Vision-capable model
    MODEL_TOGETHER = "Qwen/Qwen3-VL-32B-Instruct"  # Qwen 3 Vision model (32B) - fast and accurate
    # Alternatives: "Qwen/Qwen2.5-VL-72B-Instruct", "Qwen/Qwen3-VL-8B-Instruct"
    
    # Local (Ollama): Latest vision model
    # Note: Update this to match your installed Ollama model
    MODEL_LOCAL = "gemma3:27b"  # Currently installed model (supports vision)
    # To install other vision models, run: ollama pull llama3.2-vision
    # Alternatives: "llama3.2-vision", "llama3.1:8b-vision", "llava:latest", "gemma2:9b-vision"

    # Settings
    OLLAMA_BASE_URL = "http://localhost:11434"

    # Generation Parameters - Centralized Configuration
    # These settings control the behavior of all VLM providers
    
    # Temperature: Controls randomness in responses
    # - 0.0 = Most deterministic (recommended for scientific/medical analysis)
    # - 1.0 = More creative/random
    GENERATION_TEMPERATURE = 0.0
    
    # Max output tokens: Maximum length of generated responses
    # - 8000 is sufficient for detailed JSON responses with all subcategories
    GENERATION_MAX_TOKENS = 8000
    
    # Top-p (nucleus sampling): Focuses on most likely tokens
    # - 0.95 = Consider top 95% probability mass
    # - Lower values = more focused, higher = more diverse
    GENERATION_TOP_P = 0.95
    
    # Top-k: Limits to top K most likely tokens
    # - 40 = Standard value (Google's default)
    # - Higher = more diverse, lower = more focused
    GENERATION_TOP_K = 40
    
    # Repetition penalty (Together AI only)
    # - 1.0 = No penalty
    # - >1.0 = Discourage repetition
    GENERATION_REPETITION_PENALTY = 1.0
