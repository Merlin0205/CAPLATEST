import hashlib
import base64

# Password hash for access control
PASSWORD_HASH = "5ff171d62e4d576f6e870020a480f1acac85e8b3f9a2950fd27c73cd632e1897"

# Hashed API key
HASHED_API_KEY = base64.b64encode("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".encode()).decode()

def verify_password(password: str) -> bool:
    """Verifies if the provided password matches the hash"""
    try:
        hashed = hashlib.sha256(password.encode()).hexdigest()
        return hashed == PASSWORD_HASH
    except Exception:
        return False

def get_api_key(password: str) -> str:
    """Gets API key if the correct password is provided"""
    if verify_password(password):
        try:
            return base64.b64decode(HASHED_API_KEY).decode()
        except Exception:
            return ""
    return "" 