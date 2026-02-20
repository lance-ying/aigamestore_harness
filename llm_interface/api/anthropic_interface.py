"""
Anthropic API interface for LLM inference with reasoning support.
"""

import os
import base64
from typing import Dict, Any, Optional, List, Union
import anthropic
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

class AnthropicInterface:
    """
    Anthropic API interface for LLM inference with reasoning support.
    """
    
    def __init__(
        self,
        model_name: str,
        enable_thinking: bool = False,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Anthropic interface.
        
        Args:
            model_name: Name of the Anthropic model
            enable_thinking: Whether to enable reasoning mode
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if None)
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
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize client
        self.client = anthropic.Anthropic(api_key=self.api_key, **kwargs)
        
        # Store configuration
        self.model_config = {
            'model_name': model_name,
            'enable_thinking': self.enable_thinking,
            'provider': 'anthropic',
            'model_info': self.model_info
        }
        
        logger.info(f"Successfully initialized Anthropic interface for {model_name}")
    
    def _process_message_content(self, content: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Process message content to handle images and text for Anthropic API.
        Anthropic API expects content as a list of content blocks.
        """
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        
        processed_content = []
        for part in content:
            part_type = part.get('type')
            if part_type in ('text', 'input_text'):
                text_value = part.get('text', '')
                if text_value:
                    processed_content.append({"type": "text", "text": text_value})
            elif part_type == 'image_base64':
                base64_data = part.get('data')
                mime_type = part.get('mime_type', 'image/png')
                if base64_data:
                    # Anthropic expects base64 data without data URL prefix
                    # Remove data URL prefix if present
                    if base64_data.startswith('data:'):
                        # Extract base64 part after comma
                        base64_data = base64_data.split(',', 1)[1] if ',' in base64_data else base64_data
                    processed_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": base64_data
                        }
                    })
            elif part_type == 'image_path':
                file_path = part.get('path')
                mime_type = part.get('mime_type', 'image/png')
                if file_path and os.path.exists(file_path):
                    base64_data = self._encode_file_to_base64(file_path)
                    if base64_data:
                        processed_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": base64_data
                            }
                        })
        
        # If no content was processed, return empty text
        if not processed_content:
            processed_content = [{"type": "text", "text": ""}]
        
        return processed_content
    
    def _encode_file_to_base64(self, file_path: str) -> Optional[str]:
        """Encode a local image file as a base64 string."""
        try:
            with open(file_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to encode image file {file_path}: {e}")
            return None
    
    def generate(
        self,
        prompt: Optional[Union[str, List[Dict[str, Any]]]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using Anthropic API.
        
        Args:
            prompt: Input prompt (string or list of content parts) - ignored if messages is provided
            messages: List of message dictionaries (role, content) for full history
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing generation results and metadata
        """
        start_time = datetime.now()
        
        # Get model defaults and validate parameters
        defaults = get_model_defaults(self.model_name)
        generation_params = {
            'max_tokens': max_tokens,
            **kwargs
        }
        
        # Remove None values and validate
        generation_params = {k: v for k, v in generation_params.items() if v is not None}
        generation_params = validate_generation_params(generation_params)
        
        # Merge with defaults
        final_params = {**defaults, **generation_params}
        
        # Note: Anthropic API doesn't allow both temperature and top_p together
        # We'll only use temperature if provided, otherwise skip both
        # For reasoning models, we use reasoning_effort instead
        
        # Process messages or prompt
        if messages is not None:
            # Use messages format (preferred for multi-modal content)
            anthropic_messages = []
            extracted_system_prompt = system_prompt
            
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                # Extract system prompt from messages if present
                if role == 'system':
                    if isinstance(content, str):
                        extracted_system_prompt = content
                    elif isinstance(content, list):
                        # Extract text from system message content
                        text_parts = [p.get('text', '') for p in content if p.get('type') == 'text']
                        extracted_system_prompt = ' '.join(text_parts)
                    continue
                
                # Process content (handles text and images)
                processed_content = self._process_message_content(content)
                
                # Only add non-empty messages
                if processed_content:
                    anthropic_messages.append({
                        "role": role,
                        "content": processed_content
                    })
            
            # Use extracted system prompt if available
            if extracted_system_prompt:
                system_prompt = extracted_system_prompt
            
            # Format prompt for token counting
            formatted_prompt = self._format_messages_for_counting(anthropic_messages, system_prompt)
            input_text = formatted_prompt
        elif prompt is not None:
            # Use prompt format (backward compatibility)
            if isinstance(prompt, str):
                anthropic_messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
                input_text = prompt
            else:
                # Prompt is a list of content parts
                processed_content = self._process_message_content(prompt)
                anthropic_messages = [{"role": "user", "content": processed_content}]
                # Extract text for input_text
                text_parts = [p.get('text', '') for p in processed_content if p.get('type') == 'text']
                input_text = ' '.join(text_parts)
            
            formatted_prompt = self._format_messages_for_counting(anthropic_messages, system_prompt)
        else:
            raise ValueError("Either 'prompt' or 'messages' must be provided")
        
        try:
            # Prepare API call parameters
            api_params = {
                'model': self.model_name,
                'messages': anthropic_messages,
                'max_tokens': final_params.get('max_tokens', 512)
            }
            
            # Add system prompt if provided
            if system_prompt:
                api_params['system'] = system_prompt
            
            # For reasoning models, set reasoning_effort to medium
            if self.enable_thinking and supports_thinking(self.model_name):
                logger.info(f"Generating with reasoning using {self.model_name}")
                api_params['reasoning_effort'] = 'medium'
            else:
                logger.info(f"Generating with standard completion using {self.model_name}")
                # Only add temperature if not a reasoning model
                # Anthropic doesn't allow both temperature and top_p together
                if temperature is not None:
                    api_params['temperature'] = temperature
                elif top_p is not None:
                    api_params['top_p'] = top_p
            
            # Make API call
            response = self.client.messages.create(**api_params)
            
            # Extract generated text
            generated_text = ""
            reasoning_text = None
            
            for content_block in response.content:
                if content_block.type == "text":
                    generated_text += content_block.text
                elif content_block.type == "thinking":
                    reasoning_text = content_block.content
            
            # Get token usage from response
            usage = response.usage
            token_usage = {
                'input_tokens': usage.input_tokens,
                'output_tokens': usage.output_tokens,
                'total_tokens': usage.input_tokens + usage.output_tokens
            }
            
            # Add reasoning tokens if available
            if reasoning_text:
                reasoning_tokens = estimate_tokens(reasoning_text, self.model_name)
                token_usage['reasoning_tokens'] = reasoning_tokens
                token_usage['total_tokens'] += reasoning_tokens
            
            # Add reasoning to result if available
            additional_data = {'reasoning': reasoning_text} if reasoning_text else {}
            
        except Exception as e:
            logger.error(f"Error generating with Anthropic API: {e}")
            raise
        
        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds()
        
        # Prepare result
        result = {
            'input_text': input_text,
            'system_prompt': system_prompt,
            'chat_template_applied': 'anthropic_messages',
            'formatted_prompt': formatted_prompt,
            'output': generated_text,
            'model_config': self.model_config,
            'generation_params': final_params,
            'token_usage': token_usage,
            'generation_time_seconds': generation_time,
            'tokens_per_second': token_usage['output_tokens'] / generation_time if generation_time > 0 else 0,
            'timestamp': start_time.isoformat(),
            **additional_data
        }
        
        logger.info(f"Generated {token_usage['output_tokens']} tokens in {generation_time:.2f}s")
        return result
    
    def _format_messages_for_counting(self, messages: List[Dict[str, Any]], system_prompt: Optional[str] = None) -> str:
        """Format messages for token counting."""
        formatted = ""
        if system_prompt:
            formatted += f"system: {system_prompt}\n"
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            # Handle content as list (Anthropic format)
            if isinstance(content, list):
                text_parts = [p.get('text', '') for p in content if isinstance(p, dict) and p.get('type') == 'text']
                content_str = ' '.join(text_parts)
                image_count = sum(1 for p in content if isinstance(p, dict) and p.get('type') == 'image')
                if image_count > 0:
                    content_str += f" [{image_count} image(s)]"
            else:
                content_str = str(content)
            
            formatted += f"{role}: {content_str}\n"
        
        return formatted.strip()
