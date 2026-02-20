"""
OpenAI API interface for LLM inference with reasoning support.
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

class OpenAIInterface:
    """
    OpenAI API interface for LLM inference with reasoning support.
    """
    
    def __init__(
        self,
        model_name: str,
        enable_thinking: bool = False,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OpenAI interface.
        
        Args:
            model_name: Name of the OpenAI model
            enable_thinking: Whether to enable reasoning mode
            api_key: OpenAI API key (uses OPENAI_KEY env var if None)
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
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize client
        self.client = OpenAI(api_key=self.api_key, **kwargs)
        
        # Store configuration
        self.model_config = {
            'model_name': model_name,
            'enable_thinking': self.enable_thinking,
            'provider': 'openai',
            'model_info': self.model_info
        }
        self.ignore_temperature = False
        self.ignore_top_p = False
        if model_name == "gpt-5-mini":
            self.ignore_temperature = True
            self.ignore_top_p = True

        logger.info(f"Successfully initialized OpenAI interface for {model_name}")

    def _encode_file_to_base64(self, file_path: str) -> Optional[str]:
        """Encode a local image file as a base64 string."""
        try:
            with open(file_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to encode image file {file_path}: {e}")
            return None

    def _to_data_url(self, base64_data: str, mime_type: str = "image/png") -> str:
        """Convert a raw base64 string into a data URL the Responses API accepts."""
        if base64_data.startswith("data:"):
            return base64_data
        return f"data:{mime_type};base64,{base64_data}"

    def _process_message_content(self, content: Union[str, List[Dict[str, Any]]]) -> Union[str, List[Dict[str, Any]]]:
        """
        Process message content to handle image uploads if necessary.
        Converts base64 images to file IDs for OpenAI.
        """
        if isinstance(content, str):
            return [{"type": "input_text", "text": content}]
            
        processed_content = []
        for part in content:
            part_type = part.get('type')
            if part_type in ('text', 'input_text'):
                text_value = part.get('text', '')
                processed_content.append({"type": "input_text", "text": text_value})
            elif part_type == 'image_base64':
                base64_data = part.get('data')
                mime_type = part.get('mime_type', 'image/png')
                if base64_data:
                    processed_content.append({
                        "type": "input_image",
                        "image_url": self._to_data_url(base64_data, mime_type)
                    })
            elif part_type == 'image_path':
                file_path = part.get('path')
                mime_type = part.get('mime_type', 'image/png')
                if file_path and os.path.exists(file_path):
                    base64_data = self._encode_file_to_base64(file_path)
                    if base64_data:
                        processed_content.append({
                            "type": "input_image",
                            "image_url": self._to_data_url(base64_data, mime_type)
                        })
            elif part_type == 'image_url':
                image_url = part.get('image_url', '')
                if isinstance(image_url, dict):
                    image_url = image_url.get('url', '')
                if isinstance(image_url, str) and image_url:
                    processed_content.append({
                        "type": "input_image",
                        "image_url": image_url
                    })
            elif part_type == 'input_image':
                processed_content.append(part)
            else:
                processed_content.append(part)
                
        return processed_content

    def _normalize_content(self, content: Union[str, List[Dict[str, Any]]], role: str) -> List[Dict[str, Any]]:
        """
        Normalize arbitrary content into Responses API parts.
        """
        if role == 'assistant':
            if isinstance(content, list):
                text = " ".join(
                    part.get('text', '')
                    for part in content
                    if part.get('type') in ('text', 'input_text', 'output_text')
                )
                return [{"type": "output_text", "text": text}]
            return [{"type": "output_text", "text": str(content)}]

        if isinstance(content, str):
            return [{"type": "input_text", "text": content}]

        if isinstance(content, list):
            if role == 'user':
                return self._process_message_content(content)
            normalized = []
            for part in content:
                part_type = part.get('type')
                if part_type in ('text', 'input_text'):
                    normalized.append({"type": "input_text", "text": part.get('text', '')})
                else:
                    normalized.append(part)
            return normalized

        return [{"type": "input_text", "text": str(content)}]
    
    def create_conversation(self) -> Dict[str, Any]:
        """
        Create a new conversation state for stateful interactions.
        
        Returns:
            Dictionary with conversation ID and initial state
        """
        import uuid
        conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
        conversation_state = {
            'id': conversation_id,
            'messages': [],
            'created_at': datetime.now().isoformat()
        }
        logger.info(f"Created conversation: {conversation_id}")
        return conversation_state
    
    def add_message_to_conversation(
        self, 
        conversation_state: Dict[str, Any], 
        content: Union[str, List[Dict[str, Any]]], 
        role: str = "user"
    ) -> None:
        """
        Add a message to an existing conversation state.
        
        Args:
            conversation_state: The conversation state dictionary
            content: Message content (string or list of content parts for multimodal)
            role: Message role (user, assistant, system)
        """
        # Process content (upload images if needed)
        if role == "user" and isinstance(content, list):
             content = self._process_message_content(content)

        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        conversation_state['messages'].append(message)
        logger.info(f"Added {role} message to conversation {conversation_state['id']}")
    
    def generate_with_conversation(
        self,
        conversation_state: Dict[str, Any],
        message: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        response_format: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate response using conversation state management.
        
        Args:
            conversation_state: The conversation state dictionary
            message: New message to add (optional if already added)
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            seed: Random seed for generation
            response_format: Response format for structured outputs
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing generation results and metadata
        """
        start_time = datetime.now()
        
        # Add message to conversation if provided
        if message:
            self.add_message_to_conversation(conversation_state, message, "user")
        
        # Prepare messages for API call
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        for msg in conversation_state['messages']:
            messages.append({"role": msg['role'], "content": msg['content']})
        
        # Get model defaults and validate parameters
        defaults = get_model_defaults(self.model_name)
        generation_params = {
            'temperature': temperature if not self.ignore_temperature else None,
            'top_p': top_p if top_p is not None and not self.ignore_top_p else None,
            'max_tokens': max_tokens,
            'seed': seed,
            'response_format': response_format,
            **kwargs
        }
        
        # Remove None values and validate
        generation_params = {k: v for k, v in generation_params.items() if v is not None}
        final_params = {**defaults, **generation_params}
        
        # Format prompt for token counting
        formatted_prompt = self._format_messages_for_counting(messages)
        
        try:
            # Prepare parameters for chat completion including structured outputs
            chat_params = {}
            for k, v in final_params.items():
                if k == 'temperature' and v is not None:
                    chat_params['temperature'] = v
                elif k == 'top_p' and v is not None:
                    chat_params['top_p'] = v
                elif k == 'max_tokens' and v is not None:
                    chat_params['max_tokens'] = v
                elif k == 'seed' and v is not None:
                    chat_params['seed'] = v
                elif k == 'response_format' and v is not None:
                    chat_params['response_format'] = v
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **chat_params
            )
            
            generated_text = response.choices[0].message.content
            
            # Add assistant response to conversation state
            self.add_message_to_conversation(conversation_state, generated_text, "assistant")
            
            # Get token usage from response
            usage = response.usage
            token_usage = {
                'input_tokens': usage.prompt_tokens,
                'output_tokens': usage.completion_tokens,
                'total_tokens': usage.total_tokens
            }
            
            additional_data = {
                'conversation_id': conversation_state['id'],
                'message_count': len(conversation_state['messages']),
                'raw_response': response
            }
            
        except Exception as e:
            logger.error(f"Error generating with conversation state: {e}")
            raise
        
        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds()
        
        # Prepare result
        result = {
            'input_text': message or f"Conversation {conversation_state['id']}",
            'system_prompt': system_prompt,
            'chat_template_applied': 'openai_conversation',
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
        
        logger.info(f"Generated {token_usage['output_tokens']} tokens in {generation_time:.2f}s using conversation state")
        return result
    
    def generate(
        self,
        prompt: Optional[Union[str, List[Dict[str, Any]]]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        response_format: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using OpenAI API with structured output support.
        
        Args:
            prompt: Input prompt (string or list of content parts) - ignored if messages is provided
            messages: List of message dictionaries (role, content) for full history
            system_prompt: Optional system prompt (prepended if messages not provided)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            seed: Random seed for generation
            response_format: Response format for structured outputs (str or dict)
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing generation results and metadata
        """
        return self._generate_with_responses(
            prompt=prompt,
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
            response_format=response_format,
            **kwargs
        )

    def _generate_with_responses(
        self,
        prompt: Optional[Union[str, List[Dict[str, Any]]]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        response_format: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Modern Responses API implementation used by generate().
        """
        start_time = datetime.now()

        defaults = get_model_defaults(self.model_name)
        generation_params = {
            'temperature': temperature if not self.ignore_temperature else None,
            'top_p': top_p if top_p is not None and not self.ignore_top_p else None,
            'max_completion_tokens': max_tokens,
            'seed': seed,
            'response_format': response_format,
            **kwargs
        }
        generation_params = {k: v for k, v in generation_params.items() if v is not None}
        final_params = {**defaults, **generation_params}

        input_payload = []
        if messages:
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                normalized_content = self._normalize_content(content, role)
                input_payload.append({
                    "role": role,
                    "content": normalized_content
                })
        else:
            if system_prompt:
                input_payload.append({
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}]
                })
            normalized_content = self._normalize_content(prompt or "", "user")
            input_payload.append({"role": "user", "content": normalized_content})

        formatted_prompt = self._format_messages_for_counting([
            {"role": entry["role"], "content": entry["content"]} for entry in input_payload
        ])

        try:
            responses_params = {}
            for k, v in final_params.items():
                if k == 'temperature' and v is not None:
                    responses_params['temperature'] = v
                elif k == 'top_p' and v is not None:
                    responses_params['top_p'] = v
                elif k == 'max_completion_tokens' and v is not None:
                    responses_params['max_output_tokens'] = v
                elif k == 'seed' and v is not None:
                    responses_params['seed'] = v
                elif k == 'response_format' and v is not None:
                    responses_params['response_format'] = v

            response = self.client.responses.create(
                model=self.model_name,
                input=input_payload,
                **responses_params
            )

            if hasattr(response, 'output_text'):
                generated_text = response.output_text
            else:
                output_segments = []
                for item in getattr(response, 'output', []) or []:
                    if getattr(item, 'type', '') == 'output_text':
                        output_segments.append(getattr(item, 'text', ''))
                generated_text = "\n".join(segment for segment in output_segments if segment)

            usage = getattr(response, 'usage', None)
            if usage:
                token_usage = {
                    'input_tokens': getattr(usage, 'input_tokens', 0),
                    'output_tokens': getattr(usage, 'output_tokens', 0),
                    'total_tokens': getattr(usage, 'total_tokens', 0)
                }
                if hasattr(usage, 'reasoning_tokens'):
                    token_usage['reasoning_tokens'] = getattr(usage, 'reasoning_tokens')
            else:
                input_tokens = estimate_tokens(formatted_prompt, self.model_name)
                output_tokens = estimate_tokens(generated_text, self.model_name)
                token_usage = {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': input_tokens + output_tokens
                }

            additional_data = {'raw_response': response}

        except Exception as e:
            logger.error(f"Error generating with OpenAI Responses API: {e}")
            raise

        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds()

        result = {
            'input_text': prompt if prompt else str(messages),
            'system_prompt': system_prompt,
            'chat_template_applied': 'openai_responses',
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

        logger.info(f"Generated {token_usage['output_tokens']} tokens in {generation_time:.2f}s via Responses API")
        return result
    
    def _format_messages_for_counting(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages for token counting."""
        formatted = ""
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            if isinstance(content, list):
                # Handle structured content (e.g. text + images)
                text_parts = []
                for part in content:
                    part_type = part.get('type')
                    if part_type in ('text', 'input_text'):
                        text_parts.append(part.get('text', ''))
                    elif part_type in ('image_url', 'input_image', 'image_path'):
                        text_parts.append("[IMAGE]")
                content_str = " ".join(text_parts)
            else:
                content_str = str(content)
            formatted += f"{role}: {content_str}\n"
        return formatted.strip()
