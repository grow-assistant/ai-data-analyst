"""Circuit Breaker Pattern Implementation for Agent Communications"""

import asyncio
import time
import logging
from enum import Enum
from typing import Callable, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Circuit is open, requests fail fast
    HALF_OPEN = "half_open" # Testing if service has recovered

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5        # Number of failures to open circuit
    recovery_timeout: int = 60        # Seconds before trying half-open
    success_threshold: int = 2        # Successes needed to close circuit
    timeout: float = 30.0            # Request timeout in seconds

class CircuitBreakerException(Exception):
    """Exception raised when circuit breaker is open."""
    pass

class CircuitBreaker:
    """
    Circuit breaker implementation to prevent cascading failures.
    
    Monitors failure rates and temporarily blocks requests to failing services,
    allowing them time to recover while preventing resource exhaustion.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_success_time = 0
        
        logger.info(f"Circuit breaker '{name}' initialized: {self.config}")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Result from function execution
            
        Raises:
            CircuitBreakerException: If circuit is open
            Exception: Original exception from function
        """
        # Check if circuit should transition states
        await self._check_state_transition()
        
        # Fast fail if circuit is open
        if self.state == CircuitState.OPEN:
            logger.warning(f"Circuit breaker '{self.name}' is OPEN - failing fast")
            raise CircuitBreakerException(f"Circuit breaker '{self.name}' is open")
        
        try:
            # Execute function with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            # Record success
            await self._record_success()
            return result
            
        except asyncio.TimeoutError as e:
            logger.error(f"Circuit breaker '{self.name}' - timeout after {self.config.timeout}s")
            await self._record_failure()
            raise
            
        except Exception as e:
            logger.error(f"Circuit breaker '{self.name}' - execution failed: {e}")
            await self._record_failure()
            raise
    
    async def _check_state_transition(self):
        """Check if circuit breaker should change state."""
        current_time = time.time()
        
        if self.state == CircuitState.OPEN:
            # Check if we should try half-open
            if current_time - self.last_failure_time >= self.config.recovery_timeout:
                logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
        
        elif self.state == CircuitState.HALF_OPEN:
            # Check if we should close circuit
            if self.success_count >= self.config.success_threshold:
                logger.info(f"Circuit breaker '{self.name}' transitioning to CLOSED")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
    
    async def _record_success(self):
        """Record successful execution."""
        self.last_success_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            logger.debug(f"Circuit breaker '{self.name}' - success {self.success_count}/{self.config.success_threshold}")
        
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    async def _record_failure(self):
        """Record failed execution."""
        self.last_failure_time = time.time()
        self.failure_count += 1
        
        logger.warning(f"Circuit breaker '{self.name}' - failure {self.failure_count}/{self.config.failure_threshold}")
        
        # Check if we should open circuit
        if (self.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN] and
            self.failure_count >= self.config.failure_threshold):
            
            logger.error(f"Circuit breaker '{self.name}' transitioning to OPEN")
            self.state = CircuitState.OPEN
            self.success_count = 0
    
    def get_status(self) -> dict:
        """Get current circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout
            }
        }


class CircuitBreakerManager:
    """Manages multiple circuit breakers for different services."""
    
    def __init__(self):
        self._breakers = {}
        logger.info("Circuit breaker manager initialized")
    
    def get_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get or create circuit breaker for service."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name, config)
            logger.info(f"Created circuit breaker for service: {name}")
        
        return self._breakers[name]
    
    def get_all_status(self) -> dict:
        """Get status of all circuit breakers."""
        return {
            name: breaker.get_status() 
            for name, breaker in self._breakers.items()
        }
    
    def reset_breaker(self, name: str):
        """Reset a circuit breaker to closed state."""
        if name in self._breakers:
            breaker = self._breakers[name]
            breaker.state = CircuitState.CLOSED
            breaker.failure_count = 0
            breaker.success_count = 0
            logger.info(f"Reset circuit breaker: {name}")

# Global circuit breaker manager instance
_circuit_breaker_manager = CircuitBreakerManager()

def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get the global circuit breaker manager instance."""
    return _circuit_breaker_manager 