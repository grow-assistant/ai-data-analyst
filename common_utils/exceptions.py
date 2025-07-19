"""Custom exceptions for the multi-agent system."""

from typing import Optional, Any, Dict, List


class MultiAgentSystemError(Exception):
    """Base exception for all multi-agent system errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details
        }


# Tool-related exceptions
class ToolError(MultiAgentSystemError):
    """Base exception for all tool-related errors."""
    pass


class ColumnNotFoundError(ToolError):
    """Raised when a specified column is not in the DataFrame."""
    
    def __init__(self, column_name: str, available_columns: Optional[List[str]] = None):
        message = f"Column '{column_name}' not found in the dataset"
        if available_columns:
            message += f". Available columns: {', '.join(available_columns)}"
        super().__init__(
            message=message,
            error_code="COLUMN_NOT_FOUND",
            details={"column_name": column_name, "available_columns": available_columns}
        )


class InvalidDataError(ToolError):
    """Raised when data format or content is invalid."""
    
    def __init__(self, reason: str, data_info: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Invalid data: {reason}",
            error_code="INVALID_DATA",
            details=data_info or {}
        )


class InsufficientDataError(ToolError):
    """Raised when there's not enough data for analysis."""
    
    def __init__(self, reason: str, required_rows: Optional[int] = None, actual_rows: Optional[int] = None):
        message = f"Insufficient data: {reason}"
        if required_rows and actual_rows:
            message += f" (required: {required_rows}, actual: {actual_rows})"
        super().__init__(
            message=message,
            error_code="INSUFFICIENT_DATA",
            details={"required_rows": required_rows, "actual_rows": actual_rows}
        )


class CalculationError(ToolError):
    """Raised when mathematical calculations fail."""
    
    def __init__(self, operation: str, reason: str):
        super().__init__(
            message=f"Calculation failed for {operation}: {reason}",
            error_code="CALCULATION_ERROR",
            details={"operation": operation, "reason": reason}
        )


# A2A Communication exceptions
class A2AError(MultiAgentSystemError):
    """Base exception for A2A communication errors."""
    pass


class A2AConnectionError(A2AError):
    """Raised when A2A connection fails."""
    
    def __init__(self, host: str, port: int, reason: str):
        super().__init__(
            message=f"Failed to connect to A2A server at {host}:{port}: {reason}",
            error_code="A2A_CONNECTION_ERROR",
            details={"host": host, "port": port, "reason": reason}
        )


class A2ATimeoutError(A2AError):
    """Raised when A2A communication times out."""
    
    def __init__(self, timeout_seconds: int, operation: str):
        super().__init__(
            message=f"A2A {operation} timed out after {timeout_seconds} seconds",
            error_code="A2A_TIMEOUT",
            details={"timeout_seconds": timeout_seconds, "operation": operation}
        )


class A2AProtocolError(A2AError):
    """Raised when A2A protocol violations occur."""
    
    def __init__(self, reason: str, received_data: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"A2A protocol error: {reason}",
            error_code="A2A_PROTOCOL_ERROR",
            details={"received_data": received_data}
        )


# Configuration exceptions
class ConfigurationError(MultiAgentSystemError):
    """Base exception for configuration-related errors."""
    pass


class MissingConfigurationError(ConfigurationError):
    """Raised when required configuration is missing."""
    
    def __init__(self, config_key: str, source: str = "environment"):
        super().__init__(
            message=f"Required configuration '{config_key}' is missing from {source}",
            error_code="MISSING_CONFIGURATION",
            details={"config_key": config_key, "source": source}
        )


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration values are invalid."""
    
    def __init__(self, config_key: str, value: Any, reason: str):
        super().__init__(
            message=f"Invalid configuration for '{config_key}': {reason} (value: {value})",
            error_code="INVALID_CONFIGURATION",
            details={"config_key": config_key, "value": value, "reason": reason}
        )


# Agent execution exceptions
class AgentExecutionError(MultiAgentSystemError):
    """Base exception for agent execution errors."""
    pass


class AgentNotFoundError(AgentExecutionError):
    """Raised when a required agent is not found."""
    
    def __init__(self, agent_name: str, available_agents: Optional[List[str]] = None):
        message = f"Agent '{agent_name}' not found"
        if available_agents:
            message += f". Available agents: {', '.join(available_agents)}"
        super().__init__(
            message=message,
            error_code="AGENT_NOT_FOUND",
            details={"agent_name": agent_name, "available_agents": available_agents}
        )


class AgentExecutionTimeout(AgentExecutionError):
    """Raised when agent execution times out."""
    
    def __init__(self, agent_name: str, timeout_seconds: int):
        super().__init__(
            message=f"Agent '{agent_name}' execution timed out after {timeout_seconds} seconds",
            error_code="AGENT_EXECUTION_TIMEOUT",
            details={"agent_name": agent_name, "timeout_seconds": timeout_seconds}
        )


# Security exceptions
class SecurityError(MultiAgentSystemError):
    """Base exception for security-related errors."""
    pass


class InputValidationError(SecurityError):
    """Raised when input validation fails."""
    
    def __init__(self, reason: str, input_data: Optional[str] = None):
        super().__init__(
            message=f"Input validation failed: {reason}",
            error_code="INPUT_VALIDATION_ERROR",
            details={"reason": reason, "input_length": len(input_data) if input_data else None}
        )


class SecurityThreatDetected(SecurityError):
    """Raised when a potential security threat is detected."""
    
    def __init__(self, threat_type: str, pattern: str, input_text: Optional[str] = None):
        super().__init__(
            message=f"Security threat detected: {threat_type} (pattern: {pattern})",
            error_code="SECURITY_THREAT_DETECTED",
            details={"threat_type": threat_type, "pattern": pattern, "input_preview": input_text[:100] if input_text else None}
        )


# Utility functions for error handling
def format_error_response(exception: Exception) -> Dict[str, Any]:
    """Format any exception into a standardized error response."""
    if isinstance(exception, MultiAgentSystemError):
        return exception.to_dict()
    else:
        return {
            "error_type": exception.__class__.__name__,
            "message": str(exception),
            "error_code": "UNKNOWN_ERROR",
            "details": {}
        }


def is_retryable_error(exception: Exception) -> bool:
    """Check if an error is retryable (network issues, timeouts, etc.)."""
    retryable_errors = (
        A2AConnectionError,
        A2ATimeoutError,
        AgentExecutionTimeout
    )
    return isinstance(exception, retryable_errors) 