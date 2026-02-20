"""
Chat template utilities for different models and thinking modes.
"""

from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Models that support thinking
THINKING_MODELS = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "<think>",
    "claude-sonnet-4-20250514": "thinking",
    "claude-3-7-sonnet-20250219": "thinking", 
    "gemini-2.5-flash": "thinking",
    "gemini-2.5-pro": "thinking",
    "gpt-5-mini-2025-08-07": "reasoning",
    "gpt-4o-2024-11-20": "reasoning"
}

def supports_thinking(model_name: str) -> bool:
    """Check if a model supports thinking mode."""
    return model_name in THINKING_MODELS

def get_thinking_template(model_name: str) -> Optional[str]:
    """Get the thinking template for a model."""
    return THINKING_MODELS.get(model_name)

def apply_chat_template(
    prompt: str,
    model_name: str,
    tokenizer=None,
    enable_thinking: bool = False,
    system_prompt: Optional[str] = None
) -> Dict[str, str]:
    """
    Apply chat template to a prompt based on model type and thinking mode.
    
    Args:
        prompt: The user prompt
        model_name: Name of the model
        tokenizer: Tokenizer object (for HuggingFace models)
        enable_thinking: Whether to enable thinking mode
        system_prompt: Optional system prompt
        
    Returns:
        Dictionary with original_prompt, formatted_prompt, and template_applied
    """
    result = {
        "original_prompt": prompt,
        "formatted_prompt": prompt,
        "template_applied": "none"
    }
    
    # Check if thinking is requested but not supported
    if enable_thinking and not supports_thinking(model_name):
        logger.warning(f"Thinking mode requested but not supported for {model_name}")
        enable_thinking = False
    
    # Handle different model families
    if "llama" in model_name.lower() or "meta-llama" in model_name.lower():
        result.update(_apply_llama_template(prompt, tokenizer, system_prompt))
    elif "qwen" in model_name.lower():
        result.update(_apply_qwen_template(prompt, tokenizer, system_prompt))
    elif "deepseek" in model_name.lower():
        result.update(_apply_deepseek_template(prompt, tokenizer, enable_thinking, system_prompt))
    else:
        # Default template for unknown models
        if system_prompt:
            result["formatted_prompt"] = f"System: {system_prompt}\n\nUser: {prompt}"
            result["template_applied"] = "basic_system_user"
        else:
            result["formatted_prompt"] = prompt
            result["template_applied"] = "none"
    
    return result

def _apply_llama_template(
    prompt: str, 
    tokenizer=None, 
    system_prompt: Optional[str] = None
) -> Dict[str, str]:
    """Apply Llama chat template."""
    if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            formatted = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            return {
                "formatted_prompt": formatted,
                "template_applied": "llama_chat_template"
            }
        except Exception as e:
            logger.warning(f"Failed to apply Llama chat template: {e}")
    
    # Fallback to manual template
    if system_prompt:
        formatted = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        formatted = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    return {
        "formatted_prompt": formatted,
        "template_applied": "llama_manual_template"
    }

def _apply_qwen_template(
    prompt: str, 
    tokenizer=None, 
    system_prompt: Optional[str] = None
) -> Dict[str, str]:
    """Apply Qwen chat template."""
    if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            formatted = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            return {
                "formatted_prompt": formatted,
                "template_applied": "qwen_chat_template"
            }
        except Exception as e:
            logger.warning(f"Failed to apply Qwen chat template: {e}")
    
    # Fallback to manual template
    if system_prompt:
        formatted = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    else:
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    return {
        "formatted_prompt": formatted,
        "template_applied": "qwen_manual_template"
    }

def _apply_deepseek_template(
    prompt: str, 
    tokenizer=None, 
    enable_thinking: bool = False,
    system_prompt: Optional[str] = None
) -> Dict[str, str]:
    """Apply DeepSeek chat template with thinking support."""
    if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            formatted = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Add thinking instruction if enabled
            if enable_thinking:
                formatted += "I need to think about this step by step.\n\n<think>\n"
            
            return {
                "formatted_prompt": formatted,
                "template_applied": "deepseek_chat_template_with_thinking" if enable_thinking else "deepseek_chat_template"
            }
        except Exception as e:
            logger.warning(f"Failed to apply DeepSeek chat template: {e}")
    
    # Fallback to manual template
    if system_prompt:
        formatted = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    else:
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    if enable_thinking:
        formatted += "I need to think about this step by step.\n\n<think>\n"
    
    return {
        "formatted_prompt": formatted,
        "template_applied": "deepseek_manual_template_with_thinking" if enable_thinking else "deepseek_manual_template"
    }
