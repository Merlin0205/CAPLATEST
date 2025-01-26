import hashlib
import os
import streamlit as st

# Stored hashes (SHA256)
PASSWORD_HASH = "5ff171d62e4d576f6e870020a480f1acac85e8b3f9a2950fd27c73cd632e1897"

def verify_password(password: str) -> bool:
    """Verifies if the provided password matches the stored hash"""
    try:
        hashed = hashlib.sha256(password.encode()).hexdigest()
        return hashed == PASSWORD_HASH
    except Exception:
        return False

def get_api_key(password: str) -> str:
    """Gets API key if the correct password is provided"""
    if verify_password(password):
        try:
            # Zkusíme načíst API klíč ze Streamlit secrets
            api_key = st.secrets.get("OPENAI_API_KEY")
            if api_key:
                return api_key
            
            # Pokud není v secrets, zkusíme environment proměnnou
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                return api_key
            
            return ""
        except Exception:
            return ""
    return "" 