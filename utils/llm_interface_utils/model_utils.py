"""
Model utilities for LLM interfaces.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Model information database with verification notes

MODEL_INFO = {
    # --- Local Models ---

    # VERIFIED: Information is correct based on official announcements.
    "meta-llama/Llama-3.2-1B-Instruct": {
        "type": "local",
        "family": "llama",
        "supports_thinking": False,
        "context_length": 131072,
        "recommended_device": "cuda"
    },
    # VERIFIED: Information is correct based on official announcements.
    "meta-llama/Llama-3.2-3B-Instruct": {
        "type": "local",
        "family": "llama",
        "supports_thinking": False,
        "context_length": 131072,
        "recommended_device": "cuda"
    },
    # PLAUSIBLE: Properties are consistent with DeepSeek models, but the name is very specific
    # and may not be an official release.
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {
        "type": "local",
        "family": "deepseek",
        "supports_thinking": True,
        "thinking_format": "<think>",
        "context_length": 32768,
        "recommended_device": "cuda"
    },
    # VERIFIED: Information is correct based on official announcements.
    "Qwen/Qwen2-7B-Instruct": {
        "type": "local",
        "family": "qwen",
        "supports_thinking": False,
        "context_length": 32768,
        "recommended_device": "cuda"
    },

    # --- API Models ---

    # NOTE: The following 'gpt-5-mini' models are SPECULATIVE.
    # They have not been announced or released by OpenAI.
    # Properties are plausible but unconfirmed.
    "gpt-5-mini-2025-08-07": {
        "type": "api",
        "provider": "openai",
        "supports_thinking": True,
        "thinking_format": "reasoning",
        "context_length": 128000
    },
    "gpt-5-mini": {
        "type": "api",
        "provider": "openai",
        "supports_thinking": True,
        "thinking_format": "reasoning",
        "context_length": 128000
    },
    "gpt-5-mini-reasoning": {
        "type": "api",
        "provider": "openai",
        "supports_thinking": True,
        "thinking_format": "reasoning",
        "context_length": 128000
    },

    # VERIFIED: GPT-4o mini exists and these properties are correct.
    "gpt-4o-mini": {
        "type": "api",
        "provider": "openai",
        "supports_thinking": True,
        "thinking_format": "reasoning",
        "context_length": 128000
    },
    
    # NOTE: This name likely refers to a model snapshot of GPT-4o.
    # The properties are correct for the base GPT-4o model family.
    "gpt-4o-2024-11-20": {
        "type": "api",
        "provider": "openai",
        "supports_thinking": True,
        "thinking_format": "reasoning",
        "context_length": 128000
    },

    # NOTE: The following Claude models are SPECULATIVE.
    # Anthropic has not released Claude 4 or 3.7.
    # Properties are plausible extrapolations from the Claude 3.5 family.
    "claude-sonnet-4-20250514": {
        "type": "api",
        "provider": "anthropic",
        "supports_thinking": True,
        "thinking_format": "thinking",
        "context_length": 200000
    },
    "claude-3-7-sonnet-20250219": {
        "type": "api",
        "provider": "anthropic",
        "supports_thinking": True,
        "thinking_format": "thinking",
        "context_length": 200000
    },

    # NOTE: The following Gemini models are SPECULATIVE.
    # Google has not released a Gemini 2.5 series.
    # Context lengths are consistent with the current Gemini 1.5 series.
    # 'thinking_budget' appears to be a custom application parameter.
    "gemini-2.5-flash": {
        "type": "api",
        "provider": "google",
        "supports_thinking": True,
        "thinking_format": "thinking",
        "context_length": 1000000,
        "thinking_budget_default": 4000,
        "thinking_budget_max": 24576
    },
    "gemini-2.5-pro": {
        "type": "api",
        "provider": "google",
        "supports_thinking": True,
        "thinking_format": "thinking",
        "context_length": 2000000,
        "thinking_budget_default": -1,  # Always thinking
        "thinking_budget_max": 24576
    },

    # --- xAI Grok Models ---
    # xAI Grok API models
    "grok-4-0709": {
        "type": "api",
        "provider": "xai",
        "supports_thinking": False,
        "context_length": 256000
    },
    "grok-4-1-fast-reasoning": {
        "type": "api",
        "provider": "xai",
        "supports_thinking": False,
        "context_length": 256000
    },
    "grok-2-1212": {
        "type": "api",
        "provider": "xai",
        "supports_thinking": False,
        "context_length": 131072
    },
    "grok-2-vision-1212": {
        "type": "api",
        "provider": "xai",
        "supports_thinking": False,
        "context_length": 131072
    },
    "grok-beta": {
        "type": "api",
        "provider": "xai",
        "supports_thinking": False,
        "context_length": 131072
    },

    # --- TogetherAI Models ---
    # TogetherAI provides access to open-source models via OpenAI-compatible API
    # Model names should match TogetherAI's model registry names
    "Qwen/Qwen2.5-72B-Instruct": {
        "type": "api",
        "provider": "togetherai",
        "supports_thinking": False,
        "context_length": 32768
    },
    "Qwen/Qwen2.5-32B-Instruct": {
        "type": "api",
        "provider": "togetherai",
        "supports_thinking": False,
        "context_length": 32768
    },
    "meta-llama/Llama-3.1-70B-Instruct": {
        "type": "api",
        "provider": "togetherai",
        "supports_thinking": False,
        "context_length": 131072
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "type": "api",
        "provider": "togetherai",
        "supports_thinking": False,
        "context_length": 131072
    },
    # User-requested models (use TogetherAI model names as they appear in their API)
    "Qwen/Qwen3-235B": {
        "type": "api",
        "provider": "togetherai",
        "supports_thinking": False,
        "context_length": 131072
    },
    "meta-llama/Llama-4-Maverick": {
        "type": "api",
        "provider": "togetherai",
        "supports_thinking": False,
        "context_length": 131072
    },
    "THUDM/GLM-4.5": {
        "type": "api",
        "provider": "togetherai",
        "supports_thinking": False,
        "context_length": 131072
    }
}

def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with model information
    """
    info = MODEL_INFO.get(model_name, {})
    
    if not info:
        logger.warning(f"No information available for model: {model_name}")
        # Return default info
        info = {
            "type": "unknown",
            "family": "unknown",
            "supports_thinking": False,
            "context_length": 4096,
            "recommended_device": "cuda"
        }
    
    return info.copy()

def supports_thinking(model_name: str) -> bool:
    """
    Check if a model supports thinking mode.
    
    Args:
        model_name: Name of the model
        
    Returns:
        True if model supports thinking, False otherwise
    """
    info = get_model_info(model_name)
    return info.get("supports_thinking", False)

def get_thinking_format(model_name: str) -> Optional[str]:
    """
    Get the thinking format for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Thinking format string or None if not supported
    """
    info = get_model_info(model_name)
    return info.get("thinking_format")

def get_recommended_device(model_name: str) -> str:
    """
    Get the recommended device for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Recommended device string
    """
    info = get_model_info(model_name)
    return info.get("recommended_device", "cuda")

def is_local_model(model_name: str) -> bool:
    """
    Check if a model is a local model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        True if local model, False otherwise
    """
    info = get_model_info(model_name)
    return info.get("type") == "local"

def is_api_model(model_name: str) -> bool:
    """
    Check if a model is an API model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        True if API model, False otherwise
    """
    info = get_model_info(model_name)
    return info.get("type") == "api"

def get_model_provider(model_name: str) -> Optional[str]:
    """
    Get the provider for an API model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Provider name or None if not an API model
    """
    info = get_model_info(model_name)
    return info.get("provider")

def validate_model_name(model_name: str) -> bool:
    """
    Validate if a model name is supported.
    
    Args:
        model_name: Name of the model
        
    Returns:
        True if model is supported, False otherwise
    """
    return model_name in MODEL_INFO
