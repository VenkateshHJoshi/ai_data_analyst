"""
Custom exception classes for the AI Data Analyst application.
These allow us to differentiate between user errors (4xx) and system errors (5xx).
"""

class AppException(Exception):
    """Base class for all application exceptions."""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class FileProcessingError(AppException):
    """Raised when file upload or parsing fails."""
    def __init__(self, message: str = "Failed to process the uploaded file."):
        super().__init__(message, status_code=400)

class QueryGenerationError(AppException):
    """Raised when the LLM fails to generate valid code or logic."""
    def __init__(self, message: str = "AI failed to understand the query."):
        super().__init__(message, status_code=500)

class CodeExecutionError(AppException):
    """Raised when the generated Python code fails during execution."""
    def __init__(self, message: str = "Failed to execute the analysis logic."):
        super().__init__(message, status_code=500)

class VisualizationError(AppException):
    """Raised when plot generation fails."""
    def __init__(self, message: str = "Failed to generate visualization."):
        super().__init__(message, status_code=500)

class InvalidQueryError(AppException):
    """Raised when the user input is ambiguous or invalid."""
    def __init__(self, message: str = "The query is invalid or incomplete."):
        super().__init__(message, status_code=400)