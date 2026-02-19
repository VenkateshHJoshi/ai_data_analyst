import uvicorn
from src.ai_analyst.api.routes import app
from src.ai_analyst.config import settings
from src.ai_analyst.utils.logger import get_logger

logger = get_logger(__name__)

def start():
    """
    Main entry point to start the AI Data Analyst API server.
    Reads configuration from settings.py.
    """
    logger.info("=" * 50)
    logger.info(f"STARTING {settings.APP_NAME}")
    logger.info(f"Version: {settings.APP_VERSION}")
    logger.info(f"Environment: {'Development' if settings.DEBUG else 'Production'}")
    logger.info(f"LLM Model: {settings.DEFAULT_MODEL}")
    logger.info("=" * 50)

    try:
        uvicorn.run(
            "src.ai_analyst.api.routes:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=settings.DEBUG,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise

if __name__ == "__main__":
    start()