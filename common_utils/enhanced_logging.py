#!/usr/bin/env python3

# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Enhanced Logging Module for Multi-Agent Framework.

Provides structured logging with context, correlation IDs, and error tracking
for improved debugging and monitoring across the AI Data Analyst Multi-Agent Framework.
"""

import logging
import json
import sys
import traceback
from typing import Dict, Any, Optional
from contextlib import contextmanager
from datetime import datetime
import uuid
import threading

# Thread-local storage for context
_context = threading.local()

class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs structured JSON logs with context.
    """
    
    def format(self, record):
        # Base log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add context if available
        context = getattr(_context, 'data', {})
        if context:
            log_entry['context'] = context.copy()
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                          'module', 'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'getMessage',
                          'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)

class ContextualLogger:
    """
    Enhanced logger that maintains context across operations.
    """
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.name = name
    
    def _log_with_context(self, log_level: int, message: str, **kwargs):
        """Log with current context and additional keyword arguments."""
        # Add kwargs to the log record as extra fields
        extra = kwargs.copy()
        
        # Remove any conflicting parameter names
        extra.pop('level', None)  # Remove 'level' if it exists in kwargs
        
        # Add correlation ID if available
        context = getattr(_context, 'data', {})
        if 'correlation_id' in context:
            extra['correlation_id'] = context['correlation_id']
        
        # Add session ID if available
        if 'session_id' in context:
            extra['session_id'] = context['session_id']
            
        # Add agent name if available
        if 'agent_name' in context:
            extra['agent_name'] = context['agent_name']
        
        self.logger.log(log_level, message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        self._log_with_context(logging.ERROR, message, exc_info=True, **kwargs)

class LoggingManager:
    """
    Central manager for enhanced logging configuration.
    """
    
    def __init__(self):
        self.configured = False
        self.loggers: Dict[str, ContextualLogger] = {}
    
    def configure(self, 
                 level: str = "INFO",
                 format_type: str = "structured",
                 output_file: Optional[str] = None):
        """Configure the logging system."""
        if self.configured:
            return
        
        # Set logging level
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        
        # Create root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create handler
        if output_file:
            handler = logging.FileHandler(output_file)
        else:
            handler = logging.StreamHandler(sys.stdout)
        
        # Set formatter
        if format_type == "structured":
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        handler.setFormatter(formatter)
        handler.setLevel(numeric_level)
        root_logger.addHandler(handler)
        
        self.configured = True
        
        # Log configuration
        config_logger = self.get_logger("logging_manager")
        config_logger.info("Enhanced logging configured", 
                          level=level, 
                          format_type=format_type,
                          output_file=output_file)
    
    def get_logger(self, name: str) -> ContextualLogger:
        """Get or create a contextual logger."""
        if name not in self.loggers:
            self.loggers[name] = ContextualLogger(name)
        return self.loggers[name]

# Context management functions
@contextmanager
def logging_context(**context_data):
    """Context manager for adding context to logs."""
    # Get current context
    current_context = getattr(_context, 'data', {})
    
    # Merge with new context
    new_context = {**current_context, **context_data}
    
    # Set new context
    _context.data = new_context
    
    try:
        yield
    finally:
        # Restore previous context
        _context.data = current_context

def add_logging_context(**context_data):
    """Add context data to the current logging context."""
    current_context = getattr(_context, 'data', {})
    current_context.update(context_data)
    _context.data = current_context

def clear_logging_context():
    """Clear the current logging context."""
    _context.data = {}

def get_logging_context() -> Dict[str, Any]:
    """Get the current logging context."""
    return getattr(_context, 'data', {}).copy()

def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return f"corr_{uuid.uuid4().hex[:12]}"

@contextmanager
def correlated_operation(operation_name: str, **context_data):
    """Context manager that adds correlation ID and operation context."""
    correlation_id = generate_correlation_id()
    
    with logging_context(
        correlation_id=correlation_id,
        operation=operation_name,
        **context_data
    ):
        logger = get_logging_manager().get_logger("operation_tracker")
        logger.info(f"Starting operation: {operation_name}")
        
        try:
            yield correlation_id
            logger.info(f"Completed operation: {operation_name}")
        except Exception as e:
            logger.error(f"Failed operation: {operation_name}", error=str(e))
            raise

# Global instance
_logging_manager = None

def get_logging_manager() -> LoggingManager:
    """Get the global logging manager instance."""
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = LoggingManager()
    return _logging_manager

def get_logger(name: str) -> ContextualLogger:
    """Convenience function to get a contextual logger."""
    return get_logging_manager().get_logger(name)

# Auto-configure if not already done
def auto_configure():
    """Auto-configure logging with sensible defaults."""
    manager = get_logging_manager()
    if not manager.configured:
        manager.configure(level="INFO", format_type="structured")

# Call auto-configure on import
auto_configure() 