from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    GROQ_API_KEY: str = Field(..., description="API Key for Groq Cloud")
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
    def validate_api_key(cls, v: str) -> str:
        if not v:
            raise ValueError("GROQ_API_KEY cannot be empty.")
        if not v.startswith("gsk_"):
            pass
        return v

settings = Settings()