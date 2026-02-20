"""
Token counting utilities for different models and tokenizers.
"""

from typing import Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

def count_tokens(
    text: str, 
    tokenizer=None, 
    model_name: Optional[str] = None
) -> int:
    """
    Count tokens in text using the appropriate tokenizer.
    
    Args:
        text: Text to count tokens for
        tokenizer: Tokenizer object (if available)
        model_name: Model name for fallback estimation
        
    Returns:
        Number of tokens
    """
    if tokenizer is not None:
        try:
            # Try different tokenizer methods
            if hasattr(tokenizer, 'encode'):
                return len(tokenizer.encode(text))
            elif hasattr(tokenizer, '__call__'):
                encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
                return encoded['input_ids'].shape[1]
            else:
                logger.warning("Tokenizer doesn't have expected methods, using estimation")
        except Exception as e:
            logger.warning(f"Failed to count tokens with tokenizer: {e}")
    
    # Fallback to estimation
    return estimate_tokens(text, model_name)

def estimate_tokens(text: str, model_name: Optional[str] = None) -> int:
    """
    Estimate token count using heuristics.
    
    Args:
        text: Text to estimate tokens for
        model_name: Model name for model-specific estimation
        
    Returns:
        Estimated number of tokens
    """
    # Basic estimation: ~4 characters per token for most models
    base_estimate = len(text) // 4
    
    # Model-specific adjustments
    if model_name:
        if 'gpt' in model_name.lower():
            # GPT models tend to have slightly more tokens
            return int(base_estimate * 1.1)
        elif 'llama' in model_name.lower():
            # Llama models are similar to base estimate
            return base_estimate
        elif 'qwen' in model_name.lower() or 'deepseek' in model_name.lower():
            # Chinese-capable models might have different tokenization
            return int(base_estimate * 1.2)
    
    return max(1, base_estimate)

def get_token_usage(
    input_text: str,
    output_text: str,
    tokenizer=None,
    model_name: Optional[str] = None
) -> Dict[str, int]:
    """
    Get token usage statistics for input and output.
    
    Args:
        input_text: Input text
        output_text: Generated output text
        tokenizer: Tokenizer object (if available)
        model_name: Model name for estimation
        
    Returns:
        Dictionary with input_tokens, output_tokens, and total_tokens
    """
    input_tokens = count_tokens(input_text, tokenizer, model_name)
    output_tokens = count_tokens(output_text, tokenizer, model_name)
    
    return {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': input_tokens + output_tokens
    }
