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
Session Manager for Advanced Context & State Management.

Provides a robust framework for managing sessions, state, and events,
enabling multi-turn conversations and persistent context for agents.
"""

import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import uuid
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class Event(BaseModel):
    """Represents an immutable event within a session."""
    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex}")
    event_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = {}
    
class State(BaseModel):
    """Represents the mutable, short-term working memory of a session."""
    data_handles: Dict[str, str] = {}
    user_preferences: Dict[str, Any] = {}
    intermediate_results: Dict[str, Any] = {}
    current_step: Optional[str] = None
    error_log: List[str] = []

class Session(BaseModel):
    """Encapsulates a single interaction thread or workflow."""
    session_id: str = Field(default_factory=lambda: f"sid_{uuid.uuid4().hex}")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    history: List[Event] = []
    state: State = Field(default_factory=State)
    metadata: Dict[str, Any] = {}

class SessionManager:
    """Manages the lifecycle of sessions, including persistence."""

    def __init__(self, storage_path: str = "./sessions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self._sessions: Dict[str, Session] = {}
        self._load_sessions()

    def create_session(self, metadata: Optional[Dict[str, Any]] = None) -> Session:
        """Creates a new session."""
        session = Session(metadata=metadata or {})
        self._sessions[session.session_id] = session
        self.add_event(session.session_id, "session_created", {"metadata": metadata})
        self._save_session(session.session_id)
        logger.info(f"Created new session: {session.session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieves a session by its ID."""
        return self._sessions.get(session_id)

    def update_state(self, session_id: str, updates: Dict[str, Any]) -> Optional[State]:
        """Updates the state of a session."""
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"Session not found for state update: {session_id}")
            return None
        
        # Update state fields directly
        for key, value in updates.items():
            if hasattr(session.state, key):
                setattr(session.state, key, value)
            else:
                logger.warning(f"Unknown state field: {key}")
        
        session.last_updated = datetime.utcnow()
        self.add_event(session_id, "state_updated", {"updates": updates})
        self._save_session(session_id)
        return session.state

    def add_event(self, session_id: str, event_type: str, details: Optional[Dict[str, Any]] = None) -> Optional[Event]:
        """Adds an event to a session's history."""
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"Session not found for adding event: {session_id}")
            return None
        
        event = Event(event_type=event_type, details=details or {})
        session.history.append(event)
        session.last_updated = datetime.utcnow()
        # No need to save here, as it will be saved with state updates.
        return event

    def _save_session(self, session_id: str):
        """Saves a session to a JSON file."""
        session = self.get_session(session_id)
        if not session:
            return
        
        file_path = self.storage_path / f"{session_id}.json"
        try:
            with open(file_path, "w") as f:
                f.write(session.model_dump_json(indent=2))
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")

    def _load_sessions(self):
        """Loads all sessions from the storage path on initialization."""
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    session = Session(**data)
                    self._sessions[session.session_id] = session
            except Exception as e:
                logger.error(f"Failed to load session from {file_path}: {e}")
        logger.info(f"Loaded {len(self._sessions)} sessions from {self.storage_path}")

# Global instance
_session_manager = None

def get_session_manager() -> SessionManager:
    """Gets the global instance of the SessionManager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager 