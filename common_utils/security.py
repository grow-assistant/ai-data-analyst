"""Enhanced Security Module for A2A/MCP System with OAuth2 and ACL Support"""

import logging
import hashlib
import hmac
import json
import time
import jwt
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import asyncio
import aiofiles

logger = logging.getLogger(__name__)

class SecurityThreatDetected(Exception):
    """Exception raised when a security threat is detected."""
    pass

class InputValidationError(Exception):
    """Exception raised when input validation fails."""
    pass

class AuthenticationError(Exception):
    """Exception raised when authentication fails."""
    pass

class AuthorizationError(Exception):
    """Exception raised when authorization fails."""
    pass

class PermissionLevel(Enum):
    """Permission levels for ACL."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"

@dataclass
class OAuthToken:
    """OAuth2 token structure."""
    access_token: str
    token_type: str
    expires_in: int
    refresh_token: Optional[str] = None
    scope: Optional[str] = None
    issued_at: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if token is expired."""
        if not self.issued_at:
            return True
        return datetime.utcnow() > (self.issued_at + timedelta(seconds=self.expires_in))

@dataclass
class AccessControlEntry:
    """Access Control List entry."""
    agent_id: str
    resource: str
    permissions: Set[PermissionLevel]
    conditions: Optional[Dict[str, Any]] = None

@dataclass
class AuditLogEntry:
    """Audit log entry structure."""
    timestamp: datetime
    agent_id: str
    action: str
    resource: str
    result: str
    details: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

class OAuth2Manager:
    """OAuth2 authentication manager for MCP connections."""
    
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.active_tokens: Dict[str, OAuthToken] = {}
        self.client_credentials: Dict[str, str] = {}  # client_id -> client_secret
        
    def register_client(self, client_id: str, client_secret: str):
        """Register OAuth2 client credentials."""
        self.client_credentials[client_id] = client_secret
        logger.info(f"OAuth2 client registered: {client_id}")
    
    async def authenticate_client(self, client_id: str, client_secret: str) -> bool:
        """Authenticate OAuth2 client credentials."""
        stored_secret = self.client_credentials.get(client_id)
        if not stored_secret:
            logger.warning(f"Unknown OAuth2 client: {client_id}")
            return False
        
        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(stored_secret, client_secret)
    
    async def generate_access_token(self, client_id: str, scope: str = "mcp:tools") -> OAuthToken:
        """Generate OAuth2 access token."""
        now = datetime.utcnow()
        expires_in = 3600  # 1 hour
        
        payload = {
            "client_id": client_id,
            "scope": scope,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(seconds=expires_in)).timestamp()),
            "jti": secrets.token_urlsafe(16)
        }
        
        access_token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        refresh_token = secrets.token_urlsafe(32)
        
        oauth_token = OAuthToken(
            access_token=access_token,
            token_type="Bearer",
            expires_in=expires_in,
            refresh_token=refresh_token,
            scope=scope,
            issued_at=now
        )
        
        self.active_tokens[access_token] = oauth_token
        logger.info(f"Access token generated for client: {client_id}")
        return oauth_token
    
    async def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate OAuth2 access token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            
            # Check if token exists in active tokens
            if token not in self.active_tokens:
                logger.warning("Token not found in active tokens")
                return None
            
            oauth_token = self.active_tokens[token]
            if oauth_token.is_expired():
                logger.warning("Token has expired")
                del self.active_tokens[token]
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token has expired")
            if token in self.active_tokens:
                del self.active_tokens[token]
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
        
        return None
    
    async def revoke_token(self, token: str):
        """Revoke OAuth2 access token."""
        if token in self.active_tokens:
            del self.active_tokens[token]
            logger.info("Token revoked successfully")

class AccessControlManager:
    """Access Control List (ACL) manager."""
    
    def __init__(self):
        self.acl_entries: List[AccessControlEntry] = []
        self.default_permissions: Dict[str, Set[PermissionLevel]] = {}
        
    def add_acl_entry(self, agent_id: str, resource: str, permissions: List[PermissionLevel], 
                      conditions: Optional[Dict[str, Any]] = None):
        """Add ACL entry for agent-resource permission."""
        entry = AccessControlEntry(
            agent_id=agent_id,
            resource=resource,
            permissions=set(permissions),
            conditions=conditions
        )
        self.acl_entries.append(entry)
        logger.info(f"ACL entry added: {agent_id} -> {resource} with {permissions}")
    
    def set_default_permissions(self, resource_pattern: str, permissions: List[PermissionLevel]):
        """Set default permissions for resource pattern."""
        self.default_permissions[resource_pattern] = set(permissions)
        logger.info(f"Default permissions set for {resource_pattern}: {permissions}")
    
    async def check_permission(self, agent_id: str, resource: str, 
                              permission: PermissionLevel, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if agent has permission for resource."""
        # Check explicit ACL entries first
        for entry in self.acl_entries:
            if entry.agent_id == agent_id and self._match_resource(entry.resource, resource):
                if permission in entry.permissions:
                    # Check conditions if present
                    if entry.conditions and not self._check_conditions(entry.conditions, context):
                        continue
                    logger.debug(f"Permission granted: {agent_id} -> {resource} ({permission})")
                    return True
        
        # Check default permissions
        for pattern, permissions in self.default_permissions.items():
            if self._match_resource(pattern, resource) and permission in permissions:
                logger.debug(f"Default permission granted: {agent_id} -> {resource} ({permission})")
                return True
        
        logger.warning(f"Permission denied: {agent_id} -> {resource} ({permission})")
        return False
    
    def _match_resource(self, pattern: str, resource: str) -> bool:
        """Match resource pattern against actual resource."""
        # Simple wildcard matching
        if pattern == "*":
            return True
        if pattern.endswith("*"):
            return resource.startswith(pattern[:-1])
        return pattern == resource
    
    def _check_conditions(self, conditions: Dict[str, Any], context: Optional[Dict[str, Any]]) -> bool:
        """Check if conditions are met."""
        if not context:
            return False
        
        for key, value in conditions.items():
            if key not in context or context[key] != value:
                return False
        return True

class AuditLogger:
    """Comprehensive audit logger for MCP tool calls."""
    
    def __init__(self, log_file: str = "audit.log"):
        self.log_file = log_file
        self.log_buffer: List[AuditLogEntry] = []
        self.buffer_size = 100
        
    async def log_action(self, agent_id: str, action: str, resource: str, result: str,
                        details: Optional[Dict[str, Any]] = None, ip_address: Optional[str] = None,
                        user_agent: Optional[str] = None):
        """Log audit action."""
        entry = AuditLogEntry(
            timestamp=datetime.utcnow(),
            agent_id=agent_id,
            action=action,
            resource=resource,
            result=result,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.log_buffer.append(entry)
        
        # Flush buffer if full
        if len(self.log_buffer) >= self.buffer_size:
            await self._flush_buffer()
    
    async def _flush_buffer(self):
        """Flush log buffer to file."""
        if not self.log_buffer:
            return
        
        try:
            async with aiofiles.open(self.log_file, 'a') as f:
                for entry in self.log_buffer:
                    log_line = {
                        "timestamp": entry.timestamp.isoformat(),
                        "agent_id": entry.agent_id,
                        "action": entry.action,
                        "resource": entry.resource,
                        "result": entry.result,
                        "details": entry.details,
                        "ip_address": entry.ip_address,
                        "user_agent": entry.user_agent
                    }
                    await f.write(json.dumps(log_line) + "\n")
            
            self.log_buffer.clear()
            logger.debug(f"Audit log flushed to {self.log_file}")
            
        except Exception as e:
            logger.error(f"Failed to flush audit log: {e}")
    
    async def search_logs(self, agent_id: Optional[str] = None, action: Optional[str] = None,
                         start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> List[Dict]:
        """Search audit logs with filters."""
        results = []
        
        try:
            async with aiofiles.open(self.log_file, 'r') as f:
                async for line in f:
                    try:
                        entry = json.loads(line.strip())
                        
                        # Apply filters
                        if agent_id and entry.get("agent_id") != agent_id:
                            continue
                        if action and entry.get("action") != action:
                            continue
                        
                        entry_time = datetime.fromisoformat(entry["timestamp"])
                        if start_time and entry_time < start_time:
                            continue
                        if end_time and entry_time > end_time:
                            continue
                        
                        results.append(entry)
                        
                    except json.JSONDecodeError:
                        continue
                        
        except FileNotFoundError:
            logger.warning(f"Audit log file not found: {self.log_file}")
        
        return results

class SecurityManager:
    """Enhanced security manager combining all security features."""
    
    def __init__(self):
        self.oauth2_manager = OAuth2Manager()
        self.acl_manager = AccessControlManager()
        self.audit_logger = AuditLogger()
        self._setup_default_permissions()
    
    def _setup_default_permissions(self):
        """Setup default ACL permissions."""
        # Data loader agent permissions
        self.acl_manager.set_default_permissions("mcp:tools:load_*", [PermissionLevel.EXECUTE])
        
        # Data cleaning agent permissions
        self.acl_manager.set_default_permissions("mcp:tools:clean_*", [PermissionLevel.EXECUTE])
        
        # Data enrichment agent permissions
        self.acl_manager.set_default_permissions("mcp:tools:enrich_*", [PermissionLevel.EXECUTE])
        self.acl_manager.set_default_permissions("external:api:*", [PermissionLevel.READ])
        
        # Presentation agent permissions
        self.acl_manager.set_default_permissions("external:storage:*", [PermissionLevel.WRITE])
        self.acl_manager.set_default_permissions("external:email:*", [PermissionLevel.EXECUTE])
        
        # Orchestrator permissions (admin access)
        self.acl_manager.set_default_permissions("*", [PermissionLevel.ADMIN, PermissionLevel.READ, PermissionLevel.WRITE, PermissionLevel.EXECUTE])
    
    async def authenticate_agent(self, client_id: str, client_secret: str) -> Optional[OAuthToken]:
        """Authenticate agent and return OAuth token."""
        if await self.oauth2_manager.authenticate_client(client_id, client_secret):
            token = await self.oauth2_manager.generate_access_token(client_id)
            await self.audit_logger.log_action(client_id, "authenticate", "oauth2", "success")
            return token
        else:
            await self.audit_logger.log_action(client_id, "authenticate", "oauth2", "failure")
            raise AuthenticationError(f"Invalid credentials for client: {client_id}")
    
    async def authorize_action(self, token: str, resource: str, permission: PermissionLevel,
                              context: Optional[Dict[str, Any]] = None) -> bool:
        """Authorize agent action using token and ACL."""
        payload = await self.oauth2_manager.validate_token(token)
        if not payload:
            await self.audit_logger.log_action("unknown", "authorize", resource, "failure", {"reason": "invalid_token"})
            raise AuthenticationError("Invalid or expired token")
        
        agent_id = payload.get("client_id")
        authorized = await self.acl_manager.check_permission(agent_id, resource, permission, context)
        
        result = "success" if authorized else "failure"
        await self.audit_logger.log_action(agent_id, "authorize", resource, result, {"permission": permission.value})
        
        if not authorized:
            raise AuthorizationError(f"Agent {agent_id} not authorized for {permission.value} on {resource}")
        
        return authorized
    
    async def log_tool_call(self, agent_id: str, tool_name: str, parameters: Dict[str, Any],
                           result: str, ip_address: Optional[str] = None):
        """Log MCP tool call for audit trail."""
        await self.audit_logger.log_action(
            agent_id=agent_id,
            action="tool_call",
            resource=f"mcp:tools:{tool_name}",
            result=result,
            details={"parameters": parameters},
            ip_address=ip_address
        )

# Global security manager instance
security_manager = SecurityManager()

# Existing validation functions (enhanced)
def validate_input_safety(text: str, max_length: int = 10000) -> bool:
    """Enhanced input validation with additional security checks."""
    if not isinstance(text, str):
        return False
    
    if len(text) > max_length:
        return False
    
    # Enhanced threat detection patterns
    threat_patterns = [
        # Script injection
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'vbscript:',
        r'onload\s*=',
        r'onerror\s*=',
        
        # SQL injection
        r'(union|select|insert|update|delete|drop|create|alter)\s+',
        r';\s*(drop|delete|truncate)',
        r'--\s*',
        r'/\*.*?\*/',
        
        # Command injection
        r'[;&|`$(){}[\]\\]',
        r'(rm|del|format|sudo|su)\s+',
        
        # Path traversal
        r'\.\./+',
        r'\.\.\\+',
        
        # XXE attacks
        r'<!ENTITY',
        r'SYSTEM\s+["\']',
        
        # LDAP injection
        r'[()&|!]',
        
        # NoSQL injection
        r'\$where',
        r'\$regex',
    ]
    
    import re
    text_lower = text.lower()
    
    for pattern in threat_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            logger.warning(f"Security threat detected: {pattern}")
            return False
    
    return True

def create_safety_callback(agent_name: str):
    """Enhanced safety callback with security manager integration."""
    async def safety_callback(query: str, context: Optional[Dict] = None) -> str:
        # Validate input safety
        if not validate_input_safety(query):
            await security_manager.audit_logger.log_action(
                agent_id=agent_name,
                action="safety_check",
                resource="input_validation",
                result="failure",
                details={"query_length": len(query)}
            )
            raise SecurityThreatDetected(f"Input validation failed for agent {agent_name}")
        
        await security_manager.audit_logger.log_action(
            agent_id=agent_name,
            action="safety_check", 
            resource="input_validation",
            result="success"
        )
        
        return query
    
    return safety_callback

def format_error_response(error: Exception) -> Dict[str, Any]:
    """Format error response with security considerations."""
    error_type = type(error).__name__
    
    # Don't expose sensitive internal details
    safe_message = str(error)
    if isinstance(error, (AuthenticationError, AuthorizationError)):
        safe_message = "Access denied"
    elif isinstance(error, SecurityThreatDetected):
        safe_message = "Security policy violation"
    
    return {
        "error": error_type,
        "message": safe_message,
        "timestamp": datetime.utcnow().isoformat()
    } 