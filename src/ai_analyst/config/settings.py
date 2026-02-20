from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings managed via Pydantic Settings.
    Reads variables from environment and .env file.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # --- Application Meta ---
    APP_NAME: str = "AI Data Analyst"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"  # Added for logging configuration

    # --- AI/LLM Configuration ---
    # Made Optional[str] and default None to allow Streamlit Cloud startup without .env
    GROQ_API_KEY: Optional[str] = Field(None, description="API Key for Groq Cloud")
    
    DEFAULT_MODEL: str = "openai/gpt-oss-120b"  # Updated to your preference
    TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 1024

    # --- Server Configuration ---
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # --- Data Ingestion Limits ---
    MAX_UPLOAD_SIZE_MB: int = 10

    @field_validator("GROQ_API_KEY")
    @classmethod
    def validate_api_key(cls, v: Optional[str]) -> Optional[str]:
        """
        Validates the API key if present.
        Returns None if missing (allows app to start), allowing the UI to prompt the user.
        """
        # If v is None (missing in .env), just return None to prevent crash
        if v is None:
            return v
        
        # If v is provided but empty, return None
        if not v:
            return None
        
        # Check format (optional logic)
        if not v.startswith("gsk_"):
            pass 
        return v

settings = Settings()