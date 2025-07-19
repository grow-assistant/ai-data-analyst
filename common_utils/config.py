"""Typed configuration management using Pydantic settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """Loads and validates settings from the environment."""
    model_config = SettingsConfigDict(
        env_file='.env', 
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'  # Ignore extra environment variables
    )
    
    # Google API Configuration
    google_api_key: str
    
    # MCP Configuration
    mcp_api_key: str = "mcp-dev-key"  # Default for development, should be set in production
    
    # Agent Ports (used only by start_mcp_system.py for startup)
    data_loader_port: int = 10006
    data_analyst_port: int = 10007
    orchestrator_port: int = 10000
    mcp_port: int = 10001
    
    # MCP Configuration - agents only need to know the MCP endpoint
    mcp_server_url: str = "http://localhost:10001"
    
    # Agent Configuration
    agent_name: str = "data_analyst_agent"
    agent_model: str = "gemini-2.5-flash"
    
    # Security Configuration
    enable_safety_checks: bool = True
    max_input_length: int = 4096
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Database Configuration (if needed)
    database_url: Optional[str] = None
    
    # Email Configuration (for notifications)
    email_user: Optional[str] = None
    email_password: Optional[str] = None
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    
    # Development/Testing flags
    debug_mode: bool = False
    enable_verbose_logging: bool = False
    
    def setup_logging(self) -> None:
        """Configure logging based on settings."""
        level = getattr(logging, self.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format=self.log_format,
            force=True  # Override existing configuration
        )
        
        if self.enable_verbose_logging:
            # Enable debug logging for specific modules
            logging.getLogger('google.adk').setLevel(logging.DEBUG)
            logging.getLogger('common_utils').setLevel(logging.DEBUG)
        
        logger.info(f"Logging configured with level: {self.log_level}")
    
    def validate_required_settings(self) -> None:
        """Validate that all required settings are present."""
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required but not set")
        
        logger.info("All required settings validated successfully")

# Create a single, importable instance
try:
    settings = Settings()
    settings.setup_logging()
    settings.validate_required_settings()
    logger.info("Configuration loaded successfully")
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    # Use default settings in case of error
    settings = Settings(
        google_api_key="MISSING",
        orchestrator_port=10000,
        agent_name="fallback_agent"
    )
    logger.warning("Using fallback configuration due to errors")

# Export commonly used values for backward compatibility
# NOTE: Direct usage of 'settings' object is preferred over these module-level variables
GOOGLE_API_KEY = settings.google_api_key
AGENT_NAME = settings.agent_name
AGENT_MODEL = settings.agent_model
ENABLE_SAFETY_CHECKS = settings.enable_safety_checks
MAX_INPUT_LENGTH = settings.max_input_length
LOG_LEVEL = settings.log_level
MCP_SERVER_URL = settings.mcp_server_url 