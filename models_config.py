"""
Simple configuration loader for LLM models.
Dynamically loads model configurations from environment variables.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def _discover_models():
    """
    Discover models from environment variables by looking for variable groups
    with the pattern DEPLOYMENT_NAME_*.
    """
    models = {}
    
    # Find all variables that match DEPLOYMENT_NAME_*
    deployment_vars = [v for v in os.environ if v.startswith('DEPLOYMENT_NAME_')]
    
    for var in deployment_vars:
        # Extract the suffix (everything after "DEPLOYMENT_NAME_")
        suffix = var.replace('DEPLOYMENT_NAME_', '')
        model_id = suffix.lower()
        
        # Check if all required variables exist for this suffix
        required_vars_exist = all(
            f"{key}_{suffix}" in os.environ
            for key in ["ENDPOINT", "API_KEY", "API_VERSION", "DEPLOYMENT_NAME", "API_TYPE"]
        )
        
        if required_vars_exist:
            # Get the display name from MODEL_{suffix} if available, otherwise use deployment name
            display_name = os.environ.get(f"MODEL_{suffix}", os.environ[var])
            
            models[model_id] = {
                "name": display_name,
                "suffix": suffix
            }
    
    return models

# Build models dictionary once at import time
MODELS = _discover_models()

def get_model_names():
    """Return a list of tuples containing model IDs and their display names."""
    return [(key, model["name"]) for key, model in MODELS.items()]

def get_model_info(model_id):
    """Retrieve configuration information for a given model ID."""
    if model_id in MODELS:
        return MODELS[model_id]
    else:
        raise ValueError(f"Model ID '{model_id}' not found in configuration")

def get_env_variable_keys(model_id):
    """Retrieve environment variable keys for the specified model ID."""
    model = get_model_info(model_id)
    suffix = model["suffix"]
    
    return {
        "endpoint": f"ENDPOINT_{suffix}",
        "api_key": f"API_KEY_{suffix}",
        "api_version": f"API_VERSION_{suffix}",
        "deployment_name": f"DEPLOYMENT_NAME_{suffix}",
        "api_type": f"API_TYPE_{suffix}"
    }
