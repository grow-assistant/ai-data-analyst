# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data Handle Manager for efficient data passing between agents."""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
import pickle
import gzip
from pathlib import Path
import tempfile
import os
import json

from .types import DataHandle
from .constants import DATA_HANDLE_EXPIRY_HOURS, MAX_DATA_HANDLE_SIZE_MB

logger = logging.getLogger(__name__)


class DataHandleManager:
    """
    Manages data handles for efficient data passing between agents.
    
    Instead of passing large datasets directly in A2A messages, agents can
    create data handles that reference the data, and pass only the handle.
    """
    
    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = Path(storage_dir) if storage_dir else Path(tempfile.gettempdir()) / "adk_data_handles"
        self.storage_dir.mkdir(exist_ok=True, parents=True)
        self._handles: Dict[str, DataHandle] = {}
        self._handles_file = self.storage_dir / "handles_metadata.json"
        self._cleanup_task = None
        
        # Load existing handles from file
        self._load_handles_metadata()
        
    async def start(self):
        """Start the data handle manager and cleanup task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_handles())
        logger.info(f"DataHandleManager started with storage dir: {self.storage_dir}")
        
    async def stop(self):
        """Stop the data handle manager and cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("DataHandleManager stopped")
        
    def create_handle(self, data: Any, data_type: str, metadata: Optional[Dict[str, Any]] = None) -> DataHandle:
        """
        Create a data handle for the given data.
        
        Args:
            data: The data to store
            data_type: Type of data (e.g., "dataframe", "json", "csv")
            metadata: Optional metadata about the data
            
        Returns:
            DataHandle object with unique ID
            
        Raises:
            ValueError: If data is too large or invalid
        """
        handle_id = str(uuid.uuid4())
        
        # Serialize and compress the data
        try:
            serialized_data = pickle.dumps(data)
            compressed_data = gzip.compress(serialized_data)
            
            # Check size limit
            size_mb = len(compressed_data) / (1024 * 1024)
            if size_mb > MAX_DATA_HANDLE_SIZE_MB:
                raise ValueError(f"Data size ({size_mb:.2f}MB) exceeds limit ({MAX_DATA_HANDLE_SIZE_MB}MB)")
                
            # Save to disk
            handle_path = self.storage_dir / f"{handle_id}.pkl.gz"
            with open(handle_path, 'wb') as f:
                f.write(compressed_data)
                
            # Create handle
            now = datetime.utcnow()
            expires_at = now + timedelta(hours=DATA_HANDLE_EXPIRY_HOURS)
            
            handle = DataHandle(
                handle_id=handle_id,
                data_type=data_type,
                metadata=metadata or {},
                created_at=now.isoformat(),
                expires_at=expires_at.isoformat(),
                size_bytes=len(compressed_data)
            )
            
            # Store handle metadata
            self._handles[handle_id] = handle
            self._save_handles_metadata() # Save after creating a new handle
            
            logger.info(f"Created data handle {handle_id} for {data_type} data ({size_mb:.2f}MB)")
            return handle
            
        except Exception as e:
            logger.error(f"Failed to create data handle: {e}")
            raise ValueError(f"Failed to create data handle: {e}")
            
    def get_data(self, handle_id: str) -> Any:
        """
        Retrieve data for the given handle ID.
        
        Args:
            handle_id: The handle ID to retrieve data for
            
        Returns:
            The original data
            
        Raises:
            ValueError: If handle not found or expired
        """
        if handle_id not in self._handles:
            raise ValueError(f"Data handle {handle_id} not found")
            
        handle = self._handles[handle_id]
        
        # Check expiration
        if handle.expires_at:
            expires_at = datetime.fromisoformat(handle.expires_at)
            if datetime.utcnow() > expires_at:
                self._cleanup_handle(handle_id)
                raise ValueError(f"Data handle {handle_id} has expired")
                
        # Load from disk
        handle_path = self.storage_dir / f"{handle_id}.pkl.gz"
        if not handle_path.exists():
            raise ValueError(f"Data file for handle {handle_id} not found")
            
        try:
            with open(handle_path, 'rb') as f:
                compressed_data = f.read()
            
            serialized_data = gzip.decompress(compressed_data)
            data = pickle.loads(serialized_data)
            
            logger.debug(f"Retrieved data for handle {handle_id}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to retrieve data for handle {handle_id}: {e}")
            raise ValueError(f"Failed to retrieve data: {e}")
            
    def get_handle(self, handle_id: str) -> Optional[DataHandle]:
        """Get handle metadata by ID."""
        handle = self._handles.get(handle_id)
        if handle is None:
            # Try reloading metadata from file in case another agent created it
            self._load_handles_metadata()
            handle = self._handles.get(handle_id)
        return handle
        
    def delete_handle(self, handle_id: str) -> bool:
        """
        Delete a data handle and its associated data.
        
        Args:
            handle_id: The handle ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        return self._cleanup_handle(handle_id)
        
    def list_handles(self) -> Dict[str, DataHandle]:
        """List all active data handles."""
        return self._handles.copy()
        
    def _cleanup_handle(self, handle_id: str) -> bool:
        """Clean up a single handle."""
        if handle_id not in self._handles:
            return False
            
        # Remove from memory
        del self._handles[handle_id]
        self._save_handles_metadata() # Save after deleting a handle
        
        # Remove from disk
        handle_path = self.storage_dir / f"{handle_id}.pkl.gz"
        try:
            if handle_path.exists():
                handle_path.unlink()
            logger.debug(f"Cleaned up handle {handle_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete file for handle {handle_id}: {e}")
            return False
            
    async def _cleanup_expired_handles(self):
        """Periodic cleanup of expired handles."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                now = datetime.utcnow()
                expired_handles = []
                
                for handle_id, handle in self._handles.items():
                    if handle.expires_at:
                        expires_at = datetime.fromisoformat(handle.expires_at)
                        if now > expires_at:
                            expired_handles.append(handle_id)
                            
                for handle_id in expired_handles:
                    self._cleanup_handle(handle_id)
                    
                if expired_handles:
                    logger.info(f"Cleaned up {len(expired_handles)} expired data handles")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in data handle cleanup: {e}")

    def _load_handles_metadata(self):
        """Load handle metadata from persistent storage."""
        if self._handles_file.exists():
            try:
                with open(self._handles_file, 'r') as f:
                    handles_data = json.load(f)
                    for handle_id, handle_dict in handles_data.items():
                        self._handles[handle_id] = DataHandle(**handle_dict)
                logger.debug(f"Loaded {len(self._handles)} handles from metadata file")
            except Exception as e:
                logger.warning(f"Failed to load handles metadata: {e}")
    
    def _save_handles_metadata(self):
        """Save handle metadata to persistent storage."""
        try:
            handles_data = {}
            for handle_id, handle in self._handles.items():
                # Convert handle to dict and ensure JSON serializable
                handle_dict = handle.model_dump()
                # Convert any numpy types to native Python types
                def convert_numpy_types(obj):
                    if hasattr(obj, 'dtype'):  # numpy types
                        return obj.item() if hasattr(obj, 'item') else str(obj)
                    elif isinstance(obj, dict):
                        return {k: convert_numpy_types(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy_types(v) for v in obj]
                    else:
                        return obj
                
                handles_data[handle_id] = convert_numpy_types(handle_dict)
            
            with open(self._handles_file, 'w') as f:
                json.dump(handles_data, f, indent=2)
            logger.debug(f"Saved {len(self._handles)} handles to metadata file")
        except Exception as e:
            logger.error(f"Failed to save handles metadata: {e}")


# Global instance
_data_handle_manager: Optional[DataHandleManager] = None


def get_data_handle_manager() -> DataHandleManager:
    """Get the global data handle manager instance."""
    global _data_handle_manager
    if _data_handle_manager is None:
        _data_handle_manager = DataHandleManager()
    return _data_handle_manager


async def initialize_data_handle_manager(storage_dir: Optional[str] = None):
    """Initialize the global data handle manager."""
    global _data_handle_manager
    _data_handle_manager = DataHandleManager(storage_dir)
    await _data_handle_manager.start()


async def shutdown_data_handle_manager():
    """Shutdown the global data handle manager."""
    global _data_handle_manager
    if _data_handle_manager:
        await _data_handle_manager.stop()
        _data_handle_manager = None 