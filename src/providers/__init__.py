"""
Vision Language Model (VLM) Providers

This module provides a unified interface for different VLM providers including
local models (Ollama), OpenAI, Google Gemini, and Together AI.
"""

from src.providers.local import LocalVLM
from src.providers.openai import OpenAIVLM
from src.providers.google import GoogleVLM
from src.providers.together import TogetherVLM
from src.config import Config

def get_provider(provider_name):
    """
    Factory function to get the appropriate VLM provider instance.
    
    Args:
        provider_name (str): Name of the provider ('local', 'openai', 'google', 'together')
        
    Returns:
        BaseVLM: An instance of the requested VLM provider
        
    Raises:
        ValueError: If the provider name is not recognized
        
    Examples:
        >>> provider = get_provider("google")
        >>> response = provider.analyze("image.jpg", "Describe this image")
    """
    if provider_name == "local":
        return LocalVLM()
    elif provider_name == "openai":
        return OpenAIVLM()
    elif provider_name == "google":
        return GoogleVLM()
    elif provider_name == "together":
        return TogetherVLM()
    else:
        raise ValueError(f"Unknown provider: {provider_name}")

