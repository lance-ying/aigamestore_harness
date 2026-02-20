"""
TogetherAI API interface for LLM inference.
TogetherAI provides access to open-source models via OpenAI-compatible API.
"""

import os
import base64
from typing import Dict, Any, Optional, List, Union
from openai import OpenAI
import logging
from datetime import datetime

from utils.llm_interface_utils import (
    validate_generation_params,
    get_model_defaults,
    estimate_tokens,
    get_model_info,
    supports_thinking
)

logger = logging.getLogger(__name__)

class TogetherAIInterface:
    """
    TogetherAI API interface for LLM inference.
    Uses OpenAI-compatible API to access open-source models.
    """
    
    def __init__(
        self,
        model_name: str,
        enable_thinking: bool = False,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize TogetherAI interface.
        
        Args:
            model_name: Name of the TogetherAI model (e.g., 'Qwen/Qwen2.5-72B-Instruct')
            enable_thinking: Whether to enable thinking mode (if supported)
            api_key: TogetherAI API key (uses TOGETHER_API_KEY env var if None)
            base_url: API base URL (defaults to TogetherAI endpoint)
            **kwargs: Additional client arguments
        """
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self.model_info = get_model_info(model_name)
        
        # Check if thinking is supported
        if enable_thinking and not supports_thinking(model_name):
            logger.warning(f"Thinking mode requested but not supported for {model_name}")
            self.enable_thinking = False
        
        # Get API key
        self.api_key = api_key or os.getenv('TOGETHER_API_KEY')
        if not self.api_key:
            raise ValueError(
                "TogetherAI API key not found. Set TOGETHER_API_KEY environment variable or pass api_key parameter.")
        
        # Set base URL (TogetherAI uses OpenAI-compatible API)
        if base_url is None:
            base_url = "https://api.together.xyz/v1"
        
        # Initialize client with TogetherAI endpoint
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            **kwargs
        )
        
        # Store configuration
        self.model_config = {
            'model_name': model_name,
            'enable_thinking': self.enable_thinking,
            'provider': 'togetherai',
            'model_info': self.model_info
        }
        self.ignore_temperature = False
        self.ignore_top_p = False

        logger.info(f"Successfully initialized TogetherAI interface for {model_name}")

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
        TogetherAI supports vision models, so we can include images in messages.
        """
        if isinstance(content, str):
            return content
            
        processed_content = []
        for part in content:
            part_type = part.get("type")
            if part_type == "text":
                processed_content.append({
                    "type": "text",
                    "text": part.get("text", "")
                })
            elif part_type == "image_base64":
                # TogetherAI supports base64 images in vision models
                base64_data = part.get("data", "")
                mime_type = part.get("mime_type", "image/png")
                data_url = self._to_data_url(base64_data, mime_type)
                processed_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                })
            elif part_type == "image_url":
                processed_content.append(part)
            elif part_type == "image_path":
                # Convert file path to base64
                file_path = part.get("path")
                if file_path:
                    base64_data = self._encode_file_to_base64(file_path)
                    if base64_data:
                        mime_type = part.get("mime_type", "image/png")
                        data_url = self._to_data_url(base64_data, mime_type)
                        processed_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": data_url
                            }
                        })
        
        return processed_content if processed_content else [{"type": "text", "text": ""}]

    def _normalize_content(self, content: Union[str, List[Dict[str, Any]]], role: str = "user") -> Union[str, List[Dict[str, Any]]]:
        """
        Normalize content for TogetherAI API.
        """
        if isinstance(content, str):
            return content
        
        # Process content parts
        processed = self._process_message_content(content)
        
        # If only one text part, return as string
        if len(processed) == 1 and processed[0].get("type") == "text":
            return processed[0].get("text", "")
        
        return processed

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
        Generate text using TogetherAI API.
        
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

        # Prepare messages
        input_messages = []
        if messages:
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                normalized_content = self._normalize_content(content, role)
                
                # Handle system messages
                if role == 'system':
                    # TogetherAI supports system messages
                    input_messages.append({
                        "role": "system",
                        "content": normalized_content if isinstance(normalized_content, str) else str(normalized_content)
                    })
                else:
                    input_messages.append({
                        "role": role,
                        "content": normalized_content
                    })
        else:
            # Single prompt mode
            if system_prompt:
                input_messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            normalized_prompt = self._normalize_content(prompt or "")
            input_messages.append({
                "role": "user",
                "content": normalized_prompt
            })

        try:
            # Call TogetherAI API (OpenAI-compatible)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=input_messages,
                temperature=final_params.get('temperature'),
                top_p=final_params.get('top_p'),
                max_tokens=final_params.get('max_tokens'),
                seed=final_params.get('seed'),
            )
            
            # Extract response
            generated_text = response.choices[0].message.content if response.choices else ""
            
            # Extract token usage
            token_usage = {}
            if hasattr(response, 'usage') and response.usage:
                token_usage = {
                    'input_tokens': getattr(response.usage, 'prompt_tokens', 0),
                    'output_tokens': getattr(response.usage, 'completion_tokens', 0),
                    'total_tokens': getattr(response.usage, 'total_tokens', 0)
                }
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = {
                'output': generated_text,
                'token_usage': token_usage,
                'model': self.model_name,
                'provider': 'togetherai',
                'duration': duration,
                'timestamp': end_time.isoformat()
            }
            
            logger.info(
                f"TogetherAI generation completed in {duration:.2f}s - "
                f"Tokens: {token_usage.get('total_tokens', 'N/A')}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating with TogetherAI API: {e}")
            raise







