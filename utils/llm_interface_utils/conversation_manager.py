"""
Conversation manager for multi-turn chat functionality.
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConversationManager:
    """
    Manages conversation history for multi-turn chat across different LLM providers.
    """
    
    def __init__(self, model_name: str, system_prompt: Optional[str] = None):
        """
        Initialize conversation manager.
        
        Args:
            model_name: Name of the model being used
            system_prompt: Optional system prompt to start conversation
        """
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.messages: List[Dict[str, str]] = []
        
        # Add system message if provided
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self.messages.append({"role": "user", "content": content})
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation."""
        self.messages.append({"role": "assistant", "content": content})
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages in the conversation."""
        return self.messages.copy()
    
    def get_messages_for_api(self, provider: str) -> Any:
        """
        Get messages formatted for specific API provider.
        
        Args:
            provider: API provider ('openai', 'anthropic', 'google', 'local')
            
        Returns:
            Messages formatted for the specific provider
        """
        if provider in ['openai']:
            # OpenAI uses standard messages format
            return self.messages
        
        elif provider == 'anthropic':
            # Anthropic separates system prompt from messages
            messages = [msg for msg in self.messages if msg['role'] != 'system']
            return messages
        
        elif provider == 'google':
            # Google Gemini uses a different format - convert to single prompt
            return self._format_for_google()
        
        elif provider in ['local', 'hf', 'vllm']:
            # Local models (HF/vLLM) use standard messages format for chat templates
            # Based on HuggingFace documentation: https://huggingface.co/docs/transformers/en/conversations
            return self.messages
        
        else:
            return self.messages
    
    def get_system_prompt_for_api(self, provider: str) -> Optional[str]:
        """Get system prompt for APIs that handle it separately."""
        if provider == 'anthropic':
            # Anthropic handles system prompt separately
            for msg in self.messages:
                if msg['role'] == 'system':
                    return msg['content']
        return None
    
    def _format_for_google(self) -> str:
        """Format conversation for Google Gemini as a single prompt."""
        formatted_parts = []
        
        for msg in self.messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'system':
                formatted_parts.append(f"System: {content}")
            elif role == 'user':
                formatted_parts.append(f"User: {content}")
            elif role == 'assistant':
                formatted_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(formatted_parts)
    
    def _format_for_local(self) -> str:
        """Format conversation for local models as a single prompt."""
        # This will be used with chat templates
        formatted_parts = []
        
        for msg in self.messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'system':
                formatted_parts.append(f"System: {content}")
            elif role == 'user':
                formatted_parts.append(f"User: {content}")
            elif role == 'assistant':
                formatted_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(formatted_parts)
    
    def clear_conversation(self) -> None:
        """Clear all messages except system prompt."""
        if self.system_prompt:
            self.messages = [{"role": "system", "content": self.system_prompt}]
        else:
            self.messages = []
    
    def get_conversation_length(self) -> int:
        """Get the number of messages in conversation."""
        return len(self.messages)
    
    def get_last_user_message(self) -> Optional[str]:
        """Get the content of the last user message."""
        for msg in reversed(self.messages):
            if msg['role'] == 'user':
                return msg['content']
        return None
    
    def get_last_assistant_message(self) -> Optional[str]:
        """Get the content of the last assistant message."""
        for msg in reversed(self.messages):
            if msg['role'] == 'assistant':
                return msg['content']
        return None
