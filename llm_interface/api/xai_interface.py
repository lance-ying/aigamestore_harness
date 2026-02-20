"""
xAI Grok API interface for LLM inference.

Requirements:
    - Install xAI SDK: pip install xai-sdk
    - Set XAI_API_KEY environment variable with your xAI API key
    - Get API key from: https://x.ai/api/

Usage:
    from llm_interface.api import XAIInterface
    
    interface = XAIInterface(
        model_name="grok-4-0709",  # or "grok-2-1212", "grok-2-vision-1212", etc.
        enable_thinking=False
    )
    
    response = interface.generate(
        prompt="Your prompt here",
        temperature=0.7,
        max_tokens=1000
    )

Available Models:
    - grok-4-0709: Grok 4 model
    - grok-2-1212: Grok 2 model
    - grok-2-vision-1212: Grok 2 with vision capabilities
    - grok-beta: Beta version of Grok
    Note: Any model name supported by the xAI API can be used.
"""

import os
import base64
from typing import Dict, Any, Optional, List, Union
import logging
from datetime import datetime

try:
    from xai_sdk import Client
    from xai_sdk.chat import user, system, assistant
    XAI_SDK_AVAILABLE = True
except ImportError:
    XAI_SDK_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("xai-sdk not installed. Install with: pip install xai-sdk")

from utils.llm_interface_utils import (
    validate_generation_params,
    get_model_defaults,
    estimate_tokens,
    get_model_info,
    supports_thinking
)

logger = logging.getLogger(__name__)

class XAIInterface:
    """
    xAI Grok API interface for LLM inference.
    Uses the xAI SDK to interact with Grok models.
    """
    
    def __init__(
        self,
        model_name: str,
        enable_thinking: bool = False,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize xAI Grok interface.
        
        Args:
            model_name: Name of the Grok model (e.g., 'grok-4-0709', 'grok-2-1212', 'grok-2-vision-1212')
            enable_thinking: Whether to enable thinking mode (if supported)
            api_key: xAI API key (uses XAI_API_KEY env var if None)
            **kwargs: Additional client arguments
        """
        if not XAI_SDK_AVAILABLE:
            raise ImportError(
                "xai-sdk is not installed. Install it with: pip install xai-sdk"
            )
        
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self.model_info = get_model_info(model_name)
        
        # Check if thinking is supported
        if enable_thinking and not supports_thinking(model_name):
            logger.warning(f"Thinking mode requested but not supported for {model_name}")
            self.enable_thinking = False
        
        # Get API key
        self.api_key = api_key or os.getenv('XAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "xAI API key not found. Set XAI_API_KEY environment variable or pass api_key parameter."
            )
        
        # Initialize client
        self.client = Client(api_key=self.api_key, **kwargs)
        
        # Store configuration
        self.model_config = {
            'model_name': model_name,
            'enable_thinking': self.enable_thinking,
            'provider': 'xai',
            'model_info': self.model_info
        }
        self.ignore_temperature = False
        self.ignore_top_p = False

        logger.info(f"Successfully initialized xAI interface for {model_name}")

    def _encode_file_to_base64(self, file_path: str) -> Optional[str]:
        """Encode a local image file as a base64 string."""
        try:
            with open(file_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to encode image file {file_path}: {e}")
            return None

    def _to_data_url(self, base64_data: str, mime_type: str = "image/png") -> str:
        """Convert a raw base64 string into a data URL."""
        if base64_data.startswith("data:"):
            return base64_data
        return f"data:{mime_type};base64,{base64_data}"

    def _process_message_content(self, content: Union[str, List[Dict[str, Any]]]) -> Union[str, List[Dict[str, Any]]]:
        """
        Process message content to handle image uploads if necessary.
        xAI Grok supports vision models, so we can include images in messages.
        """
        if isinstance(content, str):
            return content
            
        processed_content = []
        text_parts = []
        
        for part in content:
            part_type = part.get("type")
            if part_type == "text":
                text_parts.append(part.get("text", ""))
            elif part_type == "image_base64":
                # xAI SDK supports base64 images in vision models
                base64_data = part.get("data", "")
                mime_type = part.get("mime_type", "image/png")
                # Store image data for xAI SDK
                processed_content.append({
                    "type": "image",
                    "data": base64_data,
                    "mime_type": mime_type
                })
            elif part_type == "image_url":
                # Extract base64 from data URL if needed
                url = part.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    # Extract base64 from data URL
                    header, data = url.split(",", 1)
                    mime_type = header.split(";")[0].split(":")[1]
                    processed_content.append({
                        "type": "image",
                        "data": data,
                        "mime_type": mime_type
                    })
            elif part_type == "image_path":
                # Convert file path to base64
                file_path = part.get("path")
                if file_path:
                    base64_data = self._encode_file_to_base64(file_path)
                    if base64_data:
                        mime_type = part.get("mime_type", "image/png")
                        processed_content.append({
                            "type": "image",
                            "data": base64_data,
                            "mime_type": mime_type
                        })
        
        # Combine text parts
        if text_parts:
            processed_content.insert(0, {
                "type": "text",
                "text": "\n".join(text_parts)
            })
        
        return processed_content if processed_content else [{"type": "text", "text": ""}]

    def _normalize_content(self, content: Union[str, List[Dict[str, Any]]], role: str = "user") -> Union[str, List[Dict[str, Any]]]:
        """
        Normalize content for xAI API.
        """
        if isinstance(content, str):
            return content
        
        # Process content parts
        processed = self._process_message_content(content)
        
        # If only one text part, return as string
        if len(processed) == 1 and processed[0].get("type") == "text":
            return processed[0].get("text", "")
        
        return processed

    def _build_xai_message(self, role: str, content: Union[str, List[Dict[str, Any]]]):
        """
        Build an xAI SDK message from role and content.
        """
        normalized_content = self._normalize_content(content, role)
        
        if isinstance(normalized_content, str):
            # Simple text message
            if role == "system":
                return system(normalized_content)
            elif role == "assistant":
                return assistant(normalized_content)
            else:
                return user(normalized_content)
        else:
            # Multi-modal content (text + images)
            text_content = None
            images = []
            
            for part in normalized_content:
                if part.get("type") == "text":
                    text_content = part.get("text", "")
                elif part.get("type") == "image":
                    images.append({
                        "data": part.get("data"),
                        "mime_type": part.get("mime_type", "image/png")
                    })
            
            # xAI SDK supports images in user messages
            if role == "user" and images:
                # For vision models, we need to combine text and images
                # The xAI SDK may handle this differently, so we'll use the text and attach images
                if text_content:
                    # Create user message with text
                    msg = user(text_content)
                    # Note: xAI SDK image handling may need to be adjusted based on actual SDK API
                    # This is a placeholder - you may need to check xAI SDK documentation
                    return msg
                else:
                    # Images only
                    return user("")  # Placeholder - may need image-specific handling
            else:
                # Text only
                text = text_content or ""
                if role == "system":
                    return system(text)
                elif role == "assistant":
                    return assistant(text)
                else:
                    return user(text)

    def generate(
        self,
        prompt: Optional[Union[str, List[Dict[str, Any]]]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using xAI Grok API.
        
        Args:
            prompt: Input prompt (string or list of content parts) - ignored if messages is provided
            messages: List of message dictionaries (role, content) for full history
            system_prompt: Optional system prompt (prepended if messages not provided)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            seed: Random seed for generation
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing generation results and metadata
        """
        start_time = datetime.now()

        defaults = get_model_defaults(self.model_name)
        generation_params = {
            'temperature': temperature if not self.ignore_temperature else None,
            'top_p': top_p if top_p is not None and not self.ignore_top_p else None,
            'max_tokens': max_tokens,
            'seed': seed,
            **kwargs
        }
        generation_params = {k: v for k, v in generation_params.items() if v is not None}
        final_params = {**defaults, **generation_params}

        try:
            # Prepare chat creation parameters
            chat_params = {
                'model': self.model_name,
                'temperature': final_params.get('temperature'),
                'top_p': final_params.get('top_p'),
                'max_tokens': final_params.get('max_tokens'),
                'seed': final_params.get('seed'),
            }
            
            # Add reasoning_effort for models that support it
            # Default to 'low' for faster responses (only supported by grok-3-mini currently)
            # Allow override via kwargs, but default to 'low' if not specified
            if 'reasoning_effort' in kwargs:
                reasoning_effort = kwargs['reasoning_effort']
            elif 'grok-3-mini' in self.model_name.lower():
                # Default to 'low' for grok-3-mini if not specified
                reasoning_effort = 'low'
            else:
                reasoning_effort = None
            
            if reasoning_effort is not None:
                chat_params['reasoning_effort'] = reasoning_effort
                logger.debug(f"Using reasoning_effort={reasoning_effort} for {self.model_name}")
            
            # Create chat session
            chat = self.client.chat.create(**chat_params)
            
            # Prepare messages
            if messages:
                for msg in messages:
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    xai_msg = self._build_xai_message(role, content)
                    chat.append(xai_msg)
            else:
                # Single prompt mode
                if system_prompt:
                    chat.append(system(system_prompt))
                
                normalized_prompt = self._normalize_content(prompt or "")
                user_msg = self._build_xai_message("user", normalized_prompt)
                chat.append(user_msg)
            
            # Generate response
            response = chat.sample()
            
            # Extract response text
            generated_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract token usage if available
            token_usage = {}
            if hasattr(response, 'usage'):
                usage = response.usage
                token_usage = {
                    'input_tokens': getattr(usage, 'prompt_tokens', 0) or getattr(usage, 'input_tokens', 0),
                    'output_tokens': getattr(usage, 'completion_tokens', 0) or getattr(usage, 'output_tokens', 0),
                    'total_tokens': getattr(usage, 'total_tokens', 0)
                }
            elif hasattr(response, 'token_usage'):
                token_usage = response.token_usage
            
            # Estimate tokens if not provided by API
            if not token_usage or token_usage.get('total_tokens', 0) == 0:
                # Estimate tokens
                input_text = str(messages or prompt or "")
                output_text = generated_text
                token_usage = {
                    'input_tokens': estimate_tokens(input_text, self.model_name),
                    'output_tokens': estimate_tokens(output_text, self.model_name),
                    'total_tokens': estimate_tokens(input_text, self.model_name) + estimate_tokens(output_text, self.model_name)
                }
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = {
                'output': generated_text,
                'token_usage': token_usage,
                'model': self.model_name,
                'provider': 'xai',
                'duration': duration,
                'timestamp': end_time.isoformat()
            }
            
            logger.info(
                f"xAI Grok generation completed in {duration:.2f}s - "
                f"Tokens: {token_usage.get('total_tokens', 'N/A')}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating with xAI Grok API: {e}")
            raise

