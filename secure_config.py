import hashlib
import os
import streamlit as st

def get_password_hash() -> str:
    """Gets password hash from environment or secrets"""
    # Try to load from environment variable
    env_hash = os.environ.get("PASSWORD_HASH")
    if env_hash:
        return env_hash
        
    # Then try Streamlit secrets
    try:
        secrets_hash = st.secrets.get("PASSWORD_HASH")
        if secrets_hash:
            return secrets_hash
    except:
        pass
        
    
    return "5ff171d62e4d576f6e870020a480f1acac85e8b3f9a2950fd27c73cd632e1897"

def verify_password(password: str) -> bool:
    """Verifies if the provided password matches the stored hash"""
    try:
        hashed = hashlib.sha256(password.encode()).hexdigest()
        return hashed == get_password_hash()
    except Exception:
        return False

def get_api_key(password: str) -> str:
    """Gets API key if the correct password is provided"""
    if verify_password(password):
        try:
            # Try to load from environment variable
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                return api_key
                
            # Then try Streamlit secrets
            api_key = st.secrets.get("OPENAI_API_KEY")
            if api_key:
                return api_key
            
            return ""
        except Exception:
            return ""
    return "" 
