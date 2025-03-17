import os
import time
import uuid
import shutil
import tempfile
from pathlib import Path
import streamlit as st
from app_config import UI_CONFIG

def create_temp_dir():
    """Create a temporary directory for file uploads if it doesn't exist."""
    temp_dir = UI_CONFIG["temp_upload_dir"]
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

def save_uploaded_file(uploaded_file):
    """Save an uploaded file to the temporary directory."""
    temp_dir = create_temp_dir()
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def validate_file(uploaded_file):
    """Validate file type and size."""
    # Check file type
    file_type = uploaded_file.name.split('.')[-1].lower()
    if file_type not in UI_CONFIG["allowed_file_types"]:
        return False, f"File type not supported. Please upload: {', '.join(UI_CONFIG['allowed_file_types'])}"
    
    # Check file size
    max_size_bytes = UI_CONFIG["max_file_size_mb"] * 1024 * 1024
    if uploaded_file.size > max_size_bytes:
        return False, f"File too large. Maximum size is {UI_CONFIG['max_file_size_mb']}MB."
    
    return True, "File is valid."

def clear_temp_files():
    """Clear temporary files from the upload directory."""
    temp_dir = UI_CONFIG["temp_upload_dir"]
    if os.path.exists(temp_dir):
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                st.error(f"Error deleting {file_path}: {e}")

def generate_document_id(file_path):
    """Generate a unique document ID based on filename and timestamp."""
    base_name = os.path.basename(file_path).split('.')[0]
    timestamp = int(time.time())
    return f"{base_name}_{timestamp}"

def format_time(seconds):
    """Format time in seconds to a readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{minutes}m {remaining_seconds:.2f}s"
