"""
Configuration utilities for LLM interfaces.
"""

import os
import json
from typing import Dict, Any, Optional, Union, List
from omegaconf import OmegaConf, DictConfig
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> DictConfig:
    """
    Load configuration from YAML file using OmegaConf.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        OmegaConf configuration object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        config = OmegaConf.load(config_path)
        return config
    except Exception as e:
        raise ValueError(f"Failed to load configuration from {config_path}: {e}")

def validate_generation_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean generation parameters.
    
    Args:
        params: Dictionary of generation parameters
        
    Returns:
        Validated and cleaned parameters
    """
    validated = {}
    
    # Temperature validation
    if 'temperature' in params:
        temp = params['temperature']
        if temp is not None:
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                logger.warning(f"Invalid temperature {temp}, using default")
            else:
                validated['temperature'] = float(temp)
    
    # Top-p validation
    if 'top_p' in params:
        top_p = params['top_p']
        if top_p is not None:
            if not isinstance(top_p, (int, float)) or top_p < 0 or top_p > 1:
                logger.warning(f"Invalid top_p {top_p}, using default")
            else:
                validated['top_p'] = float(top_p)
    
    # Top-k validation
    if 'top_k' in params:
        top_k = params['top_k']
        if top_k is not None:
            if not isinstance(top_k, int) or top_k < 1:
                logger.warning(f"Invalid top_k {top_k}, using default")
            else:
                validated['top_k'] = int(top_k)
    
    # Max tokens validation
    if 'max_tokens' in params:
        max_tokens = params['max_tokens']
        if max_tokens is not None:
            if not isinstance(max_tokens, int) or max_tokens < 1:
                logger.warning(f"Invalid max_tokens {max_tokens}, using default")
            else:
                validated['max_tokens'] = int(max_tokens)
    
    # Max new tokens validation (for HuggingFace)
    if 'max_new_tokens' in params:
        max_new_tokens = params['max_new_tokens']
        if max_new_tokens is not None:
            if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
                logger.warning(f"Invalid max_new_tokens {max_new_tokens}, using default")
            else:
                validated['max_new_tokens'] = int(max_new_tokens)
    
    # Do sample validation
    if 'do_sample' in params:
        validated['do_sample'] = bool(params['do_sample'])
    
    # Seed validation
    if 'seed' in params:
        seed = params['seed']
        if seed is not None:
            validated['seed'] = int(seed)
    
    # Structured output parameters validation
    structured_params = validate_structured_output_params(params)
    validated.update(structured_params)
    
    return validated

def get_model_defaults(model_name: str) -> Dict[str, Any]:
    """
    Get default generation parameters for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary of default parameters
    """
    defaults = {
        'temperature': 0.7,
        'top_p': 0.9,
        'max_tokens': 512,
        'do_sample': True
    }
    
    # Model-specific defaults
    if 'llama' in model_name.lower():
        defaults.update({
            'temperature': 0.6,
            'top_p': 0.9,
            'max_tokens': 512
        })
    elif 'qwen' in model_name.lower():
        defaults.update({
            'temperature': 0.7,
            'top_p': 0.8,
            'max_tokens': 512
        })
    elif 'deepseek' in model_name.lower():
        defaults.update({
            'temperature': 0.7,
            'top_p': 0.95,
            'max_tokens': 1024
        })
    elif 'gpt' in model_name.lower():
        defaults.update({
            'temperature': 1.0,
            'top_p': 1.0,
            'max_tokens': 512
        })
    elif 'claude' in model_name.lower():
        defaults.update({
            'temperature': 1.0,
            'top_p': 1.0,
            'max_tokens': 512
        })
    elif 'gemini' in model_name.lower():
        defaults.update({
            'temperature': 0.9,
            'top_p': 1.0,
            'max_tokens': 512
        })
    elif 'qwen' in model_name.lower() and ('together' in model_name.lower() or 'Qwen3' in model_name):
        defaults.update({
            'temperature': 0.7,
            'top_p': 0.8,
            'max_tokens': 2048
        })
    elif 'llama' in model_name.lower() and ('together' in model_name.lower() or 'Maverick' in model_name):
        defaults.update({
            'temperature': 0.6,
            'top_p': 0.9,
            'max_tokens': 2048
        })
    elif 'glm' in model_name.lower():
        defaults.update({
            'temperature': 0.7,
            'top_p': 0.9,
            'max_tokens': 2048
        })
    elif 'grok' in model_name.lower():
        defaults.update({
            'temperature': 0.7,
            'top_p': 0.9,
            'max_tokens': 2048
        })
    
    return defaults

def merge_configs(base_config: DictConfig, override_config: Optional[DictConfig] = None) -> DictConfig:
    """
    Merge base configuration with override configuration.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration (optional)
        
    Returns:
        Merged configuration
    """
    if override_config is None:
        return base_config
    
    return OmegaConf.merge(base_config, override_config)

def validate_structured_output_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate structured output parameters for both OpenAI and vLLM interfaces.
    
    Args:
        params: Dictionary of parameters that may contain structured output settings
        
    Returns:
        Dictionary of validated structured output parameters
    """
    validated = {}
    
    # OpenAI structured output parameters
    if 'response_format' in params:
        response_format = params['response_format']
        if response_format is not None:
            validated['response_format'] = validate_response_format(response_format)
    
    # vLLM guided decoding parameters
    guided_params = ['guided_json', 'guided_regex', 'guided_choice', 'guided_grammar', 'guided_whitespace_pattern']
    for param in guided_params:
        if param in params and params[param] is not None:
            validated[param] = validate_guided_param(param, params[param])
    
    if 'guided_decoding_backend' in params and params['guided_decoding_backend'] is not None:
        backend = params['guided_decoding_backend']
        if isinstance(backend, str) and backend in ['outlines', 'lm-format-enforcer', 'xgrammar']:
            validated['guided_decoding_backend'] = backend
        else:
            logger.warning(f"Invalid guided_decoding_backend {backend}, using default")
    
    return validated

def validate_response_format(response_format: Union[Dict[str, Any], str]) -> Union[Dict[str, Any], str]:
    """
    Validate OpenAI response_format parameter.
    
    Args:
        response_format: Response format specification
        
    Returns:
        Validated response format
    """
    if isinstance(response_format, str):
        if response_format in ['text', 'json_object']:
            return response_format
        else:
            logger.warning(f"Invalid response_format string {response_format}, using 'text'")
            return 'text'
    
    elif isinstance(response_format, dict):
        # Validate JSON schema format
        if 'type' in response_format:
            if response_format['type'] == 'json_schema':
                if 'json_schema' in response_format:
                    schema = response_format['json_schema']
                    if isinstance(schema, dict) and 'schema' in schema:
                        # Ensure additionalProperties is set to false for OpenAI strict mode
                        schema_def = schema['schema']
                        if isinstance(schema_def, dict) and 'additionalProperties' not in schema_def:
                            schema_def['additionalProperties'] = False
                        return response_format
                    else:
                        logger.warning("Invalid json_schema format, missing 'schema' field")
                        return 'text'
                else:
                    logger.warning("Invalid json_schema format, missing 'json_schema' field")
                    return 'text'
            elif response_format['type'] in ['text', 'json_object']:
                return response_format
        
        logger.warning(f"Invalid response_format dict {response_format}, using 'text'")
        return 'text'
    
    else:
        logger.warning(f"Invalid response_format type {type(response_format)}, using 'text'")
        return 'text'

def validate_guided_param(param_name: str, param_value: Any) -> Any:
    """
    Validate vLLM guided decoding parameters.
    
    Args:
        param_name: Name of the guided parameter
        param_value: Value to validate
        
    Returns:
        Validated parameter value
    """
    if param_name == 'guided_json':
        if isinstance(param_value, (dict, str)):
            if isinstance(param_value, str):
                try:
                    json.loads(param_value)
                    return param_value
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON string for guided_json: {param_value}")
                    return None
            return param_value
        else:
            logger.warning(f"Invalid type for guided_json: {type(param_value)}")
            return None
    
    elif param_name == 'guided_regex':
        if isinstance(param_value, str):
            try:
                import re
                re.compile(param_value)
                return param_value
            except re.error as e:
                logger.warning(f"Invalid regex pattern: {param_value}, error: {e}")
                return None
        else:
            logger.warning(f"Invalid type for guided_regex: {type(param_value)}")
            return None
    
    elif param_name == 'guided_choice':
        if isinstance(param_value, list) and all(isinstance(x, str) for x in param_value):
            if len(param_value) > 0:
                return param_value
            else:
                logger.warning("guided_choice cannot be empty list")
                return None
        else:
            logger.warning(f"Invalid type for guided_choice: {type(param_value)}")
            return None
    
    elif param_name == 'guided_grammar':
        if isinstance(param_value, str):
            return param_value
        else:
            logger.warning(f"Invalid type for guided_grammar: {type(param_value)}")
            return None
    
    elif param_name == 'guided_whitespace_pattern':
        if isinstance(param_value, str):
            try:
                import re
                re.compile(param_value)
                return param_value
            except re.error as e:
                logger.warning(f"Invalid whitespace pattern: {param_value}, error: {e}")
                return None
        else:
            logger.warning(f"Invalid type for guided_whitespace_pattern: {type(param_value)}")
            return None
    
    return param_value

def make_openai_compatible_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make a Pydantic schema compatible with OpenAI's strict mode requirements.
    
    Args:
        schema: Pydantic model schema dictionary
        
    Returns:
        OpenAI-compatible schema
    """
    if not isinstance(schema, dict):
        return schema
    
    # Create a deep copy to avoid modifying the original
    import copy
    compatible_schema = copy.deepcopy(schema)
    
    def ensure_additional_properties_false(obj):
        """Recursively ensure additionalProperties is false for all objects."""
        if isinstance(obj, dict):
            if obj.get('type') == 'object':
                obj['additionalProperties'] = False
            
            # Recursively process nested objects
            for key, value in obj.items():
                if isinstance(value, dict):
                    ensure_additional_properties_false(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            ensure_additional_properties_false(item)
    
    ensure_additional_properties_false(compatible_schema)
    return compatible_schema
