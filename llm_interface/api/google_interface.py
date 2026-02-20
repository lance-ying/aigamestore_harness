"""
Google Gemini API interface for LLM inference with thinking support.
"""

import os
import re
import base64
import time
from typing import Dict, Any, Optional, List, Union
from google import genai
from google.genai import types
import logging
from datetime import datetime
import uuid

from utils.llm_interface_utils import (
    validate_generation_params,
    get_model_defaults,
    estimate_tokens,
    get_model_info,
    supports_thinking
)

logger = logging.getLogger(__name__)

def _is_quota_limit_error(error: Exception) -> bool:
    """Check if an error is a quota/rate limit error."""
    error_str = str(error).lower()
    quota_patterns = [
        "quota",
        "rate limit",
        "rate_limit",
        "429",
        "resource exhausted",
        "too many requests"
    ]
    return any(pattern in error_str for pattern in quota_patterns)

class GoogleInterface:
    """
    Google Gemini API interface for LLM inference with thinking support.
    """
    
    def __init__(
        self,
        model_name: str,
        enable_thinking: bool = False,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Google Gemini interface.
        
        Args:
            model_name: Name of the Gemini model
            enable_thinking: Whether to enable thinking mode
            api_key: Google API key (uses GOOGLE_API_KEY env var if None)
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
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize client
        self.client = genai.Client(api_key=self.api_key)
        
        # Store configuration
        self.model_config = {
            'model_name': model_name,
            'enable_thinking': self.enable_thinking,
            'provider': 'google',
            'model_info': self.model_info
        }
        
        logger.info(f"Successfully initialized Google interface for {model_name}")
    
    def create_conversation(self) -> Dict[str, Any]:
        """
        Create a new conversation state for stateful interactions.
        
        Returns:
            Dictionary with conversation ID and initial state
        """
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
            content: Message content (string or list of content parts)
            role: Message role (user, assistant, system)
        """
        # Note: We'll process content conversion (e.g. file uploads) just-in-time before generation
        # or store raw content here. For consistency with OpenAI interface, we store structured content.
        
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        conversation_state['messages'].append(message)
        logger.info(f"Added {role} message to conversation {conversation_state['id']}")

    def _part_from_base64(self, base64_image: str, mime_type: str = "image/png"):
        """Convert a base64 string into an inline Gemini part."""
        try:
            image_bytes = base64.b64decode(base64_image)
            return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        except Exception as e:
            logger.error(f"Failed to decode base64 image for Gemini: {e}")
            return None

    def _parse_content_for_gemini(self, content: Union[str, List[Dict[str, Any]]]) -> List[Any]:
        """
        Parse unified content format into Gemini parts using inline data for images.
        Returns a list of Part objects.
        """
        if isinstance(content, str):
            return [types.Part.from_text(text=content)]
        
        parts = []
        for part in content:
            part_type = part.get('type')
            if part_type in ('text', 'input_text'):
                text = part.get('text', '')
                parts.append(types.Part.from_text(text=text))
            elif part_type == 'image_path':
                file_path = part.get('path')
                mime_type = part.get('mime_type', 'image/png')
                if file_path and os.path.exists(file_path):
                    try:
                        with open(file_path, "rb") as image_file:
                            image_bytes = image_file.read()
                        parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
                    except Exception as e:
                        logger.error(f"Failed to load image file {file_path}: {e}")
            elif part_type == 'image_url':
                image_url = part.get('image_url', '')
                if isinstance(image_url, str) and image_url.startswith('data:image/'):
                    match = re.match(r'data:(image/[a-zA-Z0-9.+-]+);base64,(.+)', image_url)
                    if match:
                        mime_type, b64_data = match.groups()
                        gemini_part = self._part_from_base64(b64_data, mime_type)
                        if gemini_part:
                            parts.append(gemini_part)
            elif part_type == 'image_base64':
                base64_data = part.get('data')
                mime_type = part.get('mime_type', 'image/png')
                if base64_data:
                    gemini_part = self._part_from_base64(base64_data, mime_type)
                    if gemini_part:
                        parts.append(gemini_part)
            elif part_type == 'video_path':
                video_path = part.get('path')
                mime_type = part.get('mime_type', 'video/mp4')
                if video_path and os.path.exists(video_path):
                    try:
                        # Use File API for reliable video processing
                        logger.info(f"Uploading video to Gemini File API: {video_path}")
                        uploaded_file = self.client.files.upload(file=video_path)
                        
                        # Wait for the file to be processed
                        while uploaded_file.state == "PROCESSING":
                            logger.info(f"Waiting for video to process: {video_path}...")
                            time.sleep(5)
                            uploaded_file = self.client.files.get(name=uploaded_file.name)
                        
                        if uploaded_file.state != "ACTIVE":
                            logger.error(f"Video {video_path} failed to process. State: {uploaded_file.state}")
                            return parts # Skip if failed
                            
                        # Use FileData to avoid pydantic validation issues with the File object
                        parts.append(types.Part(
                            file_data=types.FileData(
                                file_uri=uploaded_file.uri,
                                mime_type=uploaded_file.mime_type
                            )
                        ))
                    except Exception as e:
                        logger.error(f"Failed to upload video file {video_path} to File API: {e}")
        return parts

    def _generate_content_with_retry(
        self,
        model: str,
        contents: List[Any],
        system_instruction: Optional[str] = None,
        config: Optional[Any] = None,
        max_retries: int = 1,
        retry_delay: float = 5.0
    ) -> Any:
        """
        Generate content with retry logic for quota limit errors.
        
        Args:
            model: Model name
            contents: List of Content objects
            system_instruction: Optional system instruction
            config: Optional GenerateContentConfig
            max_retries: Maximum number of retries (default: 1, meaning 1 initial attempt + 1 retry)
            retry_delay: Delay in seconds before retrying (default: 5.0)
        
        Returns:
            Response from generate_content
        """
        for attempt in range(max_retries + 1):
            try:
                if system_instruction:
                    return self.client.models.generate_content(
                        model=model,
                        contents=contents,
                        system_instruction=system_instruction,
                        config=config
                    )
                else:
                    return self.client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config
                    )
            except Exception as e:
                if _is_quota_limit_error(e) and attempt < max_retries:
                    logger.warning(
                        f"Quota limit error (attempt {attempt + 1}/{max_retries + 1}): {str(e)[:200]}. "
                        f"Sleeping for {retry_delay} seconds before retry..."
                    )
                    time.sleep(retry_delay)
                    continue
                else:
                    # Re-raise if not a quota error or max retries reached
                    logger.error(f"Error generating with Google API: {e}")
                    raise
        
        # This should never be reached, but just in case
        raise RuntimeError("Failed to generate content after all retries")

    # def generate_with_conversation(
    #     self,
    #     conversation_state: Dict[str, Any],
    #     message: Optional[Union[str, List[Dict[str, Any]]]] = None,
    #     system_prompt: Optional[str] = None,
    #     temperature: Optional[float] = None,
    #     top_p: Optional[float] = None,
    #     top_k: Optional[int] = None,
    #     max_tokens: Optional[int] = None,
    #     **kwargs
    # ) -> Dict[str, Any]:
    #     """
    #     Generate response using conversation state management.
    #     """
    #     start_time = datetime.now()
        
    #     # Add message to conversation if provided
    #     if message:
    #         self.add_message_to_conversation(conversation_state, message, "user")
        
    #     # Get model defaults and validate parameters
    #     defaults = get_model_defaults(self.model_name)
    #     generation_params = {
    #         'temperature': temperature,
    #         'top_p': top_p if top_p is not None else 1.0,
    #         'top_k': top_k if top_k is not None else 40,
    #         'max_tokens': max_tokens,
    #         **kwargs
    #     }
        
    #     generation_params = {k: v for k, v in generation_params.items() if v is not None}
    #     final_params = {**defaults, **generation_params}
    #     if 'top_p' not in final_params:
    #         final_params['top_p'] = 1.0
    #     if 'top_k' not in final_params:
    #         final_params['top_k'] = 40
        
    #     messages = conversation_state['messages']
    #     if not messages:
    #         raise ValueError("Conversation is empty")
            
    #     history_messages = messages[:-1]
    #     current_message = messages[-1]
        
    #     # Build chat history
    #     chat_history = []
    #     for msg in history_messages:
    #         role = 'user' if msg['role'] == 'user' else 'model'
    #         parts = self._parse_content_for_gemini(msg['content'])
    #         chat_history.append(types.Content(role=role, parts=parts))
        
    #     current_parts = self._parse_content_for_gemini(current_message['content'])
    #     current_content = types.Content(role='user', parts=current_parts)
        
    #     try:
    #         # Build generation config with only valid parameters
    #         config_params = {
    #             'temperature': final_params.get('temperature', 0.9),
    #             'top_p': final_params.get('top_p', 1.0),
    #             'top_k': final_params.get('top_k', 40),
    #             'max_output_tokens': final_params.get('max_tokens', 512)
    #         }
            
    #         # Add thinking_budget if thinking is enabled
    #         if self.enable_thinking and supports_thinking(self.model_name):
    #             if "gemini-2.5-flash" in self.model_name:
    #                 config_params['thinking_budget'] = 4000
    #             elif "gemini-2.5-pro" in self.model_name:
    #                 pass  # Use default
    #             else:
    #                 config_params['thinking_budget'] = 4000
            
    #         # Create generation config
    #         print(config_params)
    #         generation_config = types.GenerationConfig(**config_params)

    #         # Use client to generate content with chat history
    #         contents = chat_history + [current_content]
    #         if system_prompt:
    #             response = self.client.models.generate_content(
    #                 model=self.model_name,
    #                 contents=contents,
    #                 config=generation_config,
    #                 system_instruction=system_prompt
    #             )
    #         else:
    #             response = self.client.models.generate_content(
    #                 model=self.model_name,
    #                 contents=contents,
    #                 config=generation_config
    #             )
            
    #         generated_text = response.text if response.text else ""
            
    #         self.add_message_to_conversation(conversation_state, generated_text, "assistant")
            
    #         token_usage = {
    #             'input_tokens': 0,
    #             'output_tokens': 0,
    #             'total_tokens': 0
    #         }
    #         if hasattr(response, 'usage_metadata'):
    #             usage = response.usage_metadata
    #             token_usage = {
    #                 'input_tokens': usage.prompt_token_count,
    #                 'output_tokens': usage.candidates_token_count,
    #                 'total_tokens': usage.total_token_count
    #             }
            
    #         additional_data = {'raw_response': response}
    #         if hasattr(response, 'thinking') and response.thinking:
    #              additional_data['reasoning'] = response.thinking

    #     except Exception as e:
    #         logger.error(f"Error generating with Google API: {e}")
    #         raise

    #     end_time = datetime.now()
    #     generation_time = (end_time - start_time).total_seconds()
        
    #     result = {
    #         'input_text': str(current_message['content']),
    #         'system_prompt': system_prompt,
    #         'chat_template_applied': 'google_chat',
    #         'output': generated_text,
    #         'model_config': self.model_config,
    #         'generation_params': final_params,
    #         'token_usage': token_usage,
    #         'generation_time_seconds': generation_time,
    #         'tokens_per_second': token_usage['output_tokens'] / generation_time if generation_time > 0 else 0,
    #         'timestamp': start_time.isoformat(),
    #         **additional_data
    #     }
        
    #     logger.info(f"Generated {token_usage['output_tokens']} tokens in {generation_time:.2f}s")
    #     return result

    def generate(
        self,
        prompt: Optional[Union[str, List[Dict[str, Any]]]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using Google Gemini API.
        """
        start_time = datetime.now()
        
        # Determine input content
        if messages:
            # Multi-turn chat
            # Convert messages to history + last message
            history_messages = messages[:-1]
            current_message = messages[-1]
            
            chat_history = []
            for msg in history_messages:
                role = 'user' if msg['role'] == 'user' else 'model'
                parts = self._parse_content_for_gemini(msg['content'])
                chat_history.append(types.Content(role=role, parts=parts))
            
            current_parts = self._parse_content_for_gemini(current_message['content'])
            current_content = types.Content(role='user', parts=current_parts)
            contents = chat_history + [current_content]
            
            # Set config for gemini-3-pro-preview
            config = None
            if "gemini-3-pro-preview" in self.model_name:
                config = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="low")
                )
            
            # Use retry helper for quota limit handling
            response = self._generate_content_with_retry(
                model=self.model_name,
                contents=contents,
                system_instruction=system_prompt,
                config=config,
                max_retries=1,
                retry_delay=5.0
            )
            
        else:
            # Single prompt
            # Handle multimodal content
            if isinstance(prompt, list):
                full_prompt = self._parse_content_for_gemini(prompt)
                contents = [types.Content(role='user', parts=full_prompt)]
            else:
                # Convert string prompt to Part object
                prompt_parts = [types.Part.from_text(text=prompt)]
                contents = [types.Content(role='user', parts=prompt_parts)]
            
            # Set config for gemini-3-pro-preview
            config = None
            if "gemini-3-pro-preview" in self.model_name:
                config = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="low")
                )
            
            # Use retry helper for quota limit handling
            response = self._generate_content_with_retry(
                model=self.model_name,
                contents=contents,
                system_instruction=system_prompt,
                config=config,
                max_retries=1,
                retry_delay=5.0
            )
        
        generated_text = response.text if response.text else ""
        
        token_usage = {
            'input_tokens': 0, 
            'output_tokens': 0,
            'total_tokens': 0
        }
        if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                token_usage = {
                'input_tokens': usage.prompt_token_count,
                'output_tokens': usage.candidates_token_count,
                'total_tokens': usage.total_token_count
                }
        
        additional_data = {}
        if hasattr(response, 'thinking') and response.thinking:
                additional_data['reasoning'] = response.thinking
            
        # except Exception as e:
        #     logger.error(f"Error generating with Google API: {e}")
        #     raise
        
        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds()
        
        result = {
            'input_text': prompt if prompt else str(messages),
            'system_prompt': system_prompt,
            'chat_template_applied': 'google_prompt',
            'formatted_prompt': str(messages) if messages else str(prompt),
            'output': generated_text,
            'model_config': self.model_config,
            'generation_params': {},
            'token_usage': token_usage,
            'generation_time_seconds': generation_time,
            'tokens_per_second': token_usage['output_tokens'] / generation_time if generation_time > 0 else 0,
            'timestamp': start_time.isoformat(),
            **additional_data
        }
        
        logger.info(f"Generated {token_usage['output_tokens']} tokens in {generation_time:.2f}s")
        return result
