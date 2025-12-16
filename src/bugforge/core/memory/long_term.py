"""
Long-term memory implementation for persistent information storage.

LongTermMemory provides durable storage for important information
with indexing, search capabilities, and persistence features.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Union

from ..interfaces.memory_store import MemoryStore
from ..primitives import Observation, Action, State


class LongTermMemory(MemoryStore):
    """
    Long-term memory with persistence and search capabilities.
    
    Provides durable storage for important information with
    indexing, search, and backup/restore functionality.
    """
    
    def __init__(self, storage_path: str, 
                 auto_save: bool = True,
                 index_search: bool = True):
        """
        Initialize long-term memory.
        
        Args:
            storage_path: Path to storage directory
            auto_save: Whether to automatically save changes
            index_search: Whether to maintain search index
        """
        self.storage_path = Path(storage_path)
        self.auto_save = auto_save
        self.index_search = index_search
        
        self._store: Dict[str, Dict[str, Any]] = {}
        self._index: Dict[str, set] = {}  # Search index
        self._lock = RLock()
        
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self._load_from_disk()
    
    def put(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store a value with persistence and indexing.
        
        Args:
            key: Unique identifier for the memory
            value: The value to store
            metadata: Additional metadata (optional)
            
        Returns:
            True if storage was successful
        """
        with self._lock:
            now = datetime.now()
            
            self._store[key] = {
                "value": value,
                "metadata": metadata or {},
                "created_at": now,
                "updated_at": now,
                "access_count": 0
            }
            
            # Update search index
            if self.index_search:
                self._update_index(key, value, metadata or {})
            
            # Auto-save if enabled
            if self.auto_save:
                self._save_to_disk()
            
            return True
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value by key, updating access statistics.
        
        Args:
            key: The key to retrieve
            
        Returns:
            The stored value, or None if not found
        """
        with self._lock:
            if key not in self._store:
                return None
            
            # Update access statistics
            self._store[key]["access_count"] += 1
            self._store[key]["last_accessed"] = datetime.now()
            
            return self._store[key]["value"]
    
    def query(self, query: Union[str, Dict[str, Any]], 
              limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Query memories with advanced search capabilities.
        
        Args:
            query: Query string or dictionary of criteria
            limit: Maximum number of results (optional)
            
        Returns:
            List of matching memory entries
        """
        with self._lock:
            results = []
            
            if isinstance(query, str):
                # Text search using index
                candidate_keys = self._search_index(query)
                for key in candidate_keys:
                    if key in self._store:
                        item = self._store[key]
                        results.append({
                            "key": key,
                            "value": item["value"],
                            "metadata": item["metadata"],
                            "created_at": item["created_at"],
                            "updated_at": item["updated_at"],
                            "access_count": item["access_count"]
                        })
            elif isinstance(query, dict):
                # Advanced criteria-based search
                for key, item in self._store.items():
                    if self._matches_criteria(key, item, query):
                        results.append({
                            "key": key,
                            "value": item["value"],
                            "metadata": item["metadata"],
                            "created_at": item["created_at"],
                            "updated_at": item["updated_at"],
                            "access_count": item["access_count"]
                        })
            
            # Sort by relevance or recency
            results.sort(key=lambda x: (
                -x.get("access_count", 0),  # Higher access count first
                x.get("updated_at", x.get("created_at"))  # More recent first
            ))
            
            if limit:
                results = results[:limit]
            
            return results
    
    def delete(self, key: str) -> bool:
        """
        Delete a memory by key.
        
        Args:
            key: The key to delete
            
        Returns:
            True if deletion was successful
        """
        with self._lock:
            if key not in self._store:
                return False
            
            # Remove from index
            if self.index_search:
                self._remove_from_index(key)
            
            # Remove from store
            del self._store[key]
            
            # Auto-save if enabled
            if self.auto_save:
                self._save_to_disk()
            
            return True
    
    def clear(self) -> None:
        """Clear all memories from the store."""
        with self._lock:
            self._store.clear()
            self._index.clear()
            
            if self.auto_save:
                self._save_to_disk()
    
    def size(self) -> int:
        """
        Get the number of stored memories.
        
        Returns:
            Number of memories in the store
        """
        with self._lock:
            return len(self._store)
    
    def get_capabilities(self) -> List[str]:
        """
        Get the memory store's declared capabilities.
        
        Returns:
            List of capability identifiers
        """
        return ["persistent_storage", "search_indexing", "access_tracking", "metadata_support"]
    
    def get_most_accessed(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most frequently accessed memories.
        
        Args:
            count: Number of items to retrieve
            
        Returns:
            List of most accessed items
        """
        with self._lock:
            sorted_items = sorted(
                self._store.items(),
                key=lambda x: x[1].get("access_count", 0),
                reverse=True
            )
            
            results = []
            for key, item in sorted_items[:count]:
                results.append({
                    "key": key,
                    "value": item["value"],
                    "metadata": item["metadata"],
                    "created_at": item["created_at"],
                    "updated_at": item["updated_at"],
                    "access_count": item["access_count"]
                })
            
            return results
    
    def get_recently_updated(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recently updated memories.
        
        Args:
            count: Number of items to retrieve
            
        Returns:
            List of recently updated items
        """
        with self._lock:
            sorted_items = sorted(
                self._store.items(),
                key=lambda x: x[1].get("updated_at", x[1].get("created_at")),
                reverse=True
            )
            
            results = []
            for key, item in sorted_items[:count]:
                results.append({
                    "key": key,
                    "value": item["value"],
                    "metadata": item["metadata"],
                    "created_at": item["created_at"],
                    "updated_at": item["updated_at"],
                    "access_count": item["access_count"]
                })
            
            return results
    
    def backup(self, backup_path: Optional[str] = None) -> str:
        """
        Create a backup of the memory store.
        
        Args:
            backup_path: Path for backup file (optional)
            
        Returns:
            Path to the created backup file
        """
        with self._lock:
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = str(self.storage_path / f"backup_{timestamp}.json")
            
            backup_data = {
                "store": self._serialize_store(),
                "index": self._index,
                "backup_timestamp": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)
            
            return backup_path
    
    def restore(self, backup_path: str) -> None:
        """
        Restore memory store from a backup.
        
        Args:
            backup_path: Path to backup file
        """
        with self._lock:
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)
            
            self._store = self._deserialize_store(backup_data["store"])
            self._index = backup_data.get("index", {})
            
            if self.auto_save:
                self._save_to_disk()
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save memory store to disk.
        
        Args:
            path: Path to save to (uses default if None)
        """
        with self._lock:
            save_path = Path(path) if path else self.storage_path / "memory_store.json"
            self._save_to_disk(str(save_path))
    
    def load(self, path: Optional[str] = None) -> None:
        """
        Load memory store from disk.
        
        Args:
            path: Path to load from (uses default if None)
        """
        with self._lock:
            load_path = Path(path) if path else self.storage_path / "memory_store.json"
            self._load_from_disk(str(load_path))
    
    def _load_from_disk(self, path: Optional[str] = None) -> None:
        """Load memory store from disk."""
        load_path = Path(path) if path else self.storage_path / "memory_store.json"
        
        if load_path.exists():
            try:
                with open(load_path, 'r') as f:
                    data = json.load(f)
                self._store = self._deserialize_store(data.get("store", {}))
                self._index = data.get("index", {})
            except (json.JSONDecodeError, KeyError):
                # Start with empty store if file is corrupted
                self._store = {}
                self._index = {}
    
    def _save_to_disk(self, path: Optional[str] = None) -> None:
        """Save memory store to disk."""
        save_path = Path(path) if path else self.storage_path / "memory_store.json"
        
        data = {
            "store": self._serialize_store(),
            "index": self._index,
            "last_saved": datetime.now().isoformat(),
            "version": "1.0"
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _serialize_store(self) -> Dict[str, Any]:
        """Serialize store for JSON storage."""
        serialized = {}
        for key, item in self._store.items():
            serialized[key] = {
                "value": item["value"],
                "metadata": item["metadata"],
                "created_at": item["created_at"].isoformat(),
                "updated_at": item["updated_at"].isoformat(),
                "access_count": item["access_count"]
            }
            if "last_accessed" in item:
                serialized[key]["last_accessed"] = item["last_accessed"].isoformat()
        
        return serialized
    
    def _deserialize_store(self, serialized: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize store from JSON storage."""
        deserialized = {}
        for key, item in serialized.items():
            deserialized[key] = {
                "value": item["value"],
                "metadata": item["metadata"],
                "created_at": datetime.fromisoformat(item["created_at"]),
                "updated_at": datetime.fromisoformat(item["updated_at"]),
                "access_count": item["access_count"]
            }
            if "last_accessed" in item:
                deserialized[key]["last_accessed"] = datetime.fromisoformat(item["last_accessed"])
        
        return deserialized
    
    def _update_index(self, key: str, value: Any, metadata: Dict[str, Any]) -> None:
        """Update search index for a key."""
        # Remove existing index entries
        self._remove_from_index(key)
        
        # Add new index entries
        text_content = f"{key} {value} {' '.join(str(v) for v in metadata.values())}"
        words = text_content.lower().split()
        
        for word in words:
            if word not in self._index:
                self._index[word] = set()
            self._index[word].add(key)
    
    def _remove_from_index(self, key: str) -> None:
        """Remove a key from the search index."""
        for word_keys in self._index.values():
            word_keys.discard(key)
    
    def _search_index(self, query: str) -> set:
        """Search the index for matching keys."""
        query_words = query.lower().split()
        
        if not query_words:
            return set()
        
        # Find keys that contain all query words
        result_sets = []
        for word in query_words:
            if word in self._index:
                result_sets.append(self._index[word])
            else:
                # No results if any word is not found
                return set()
        
        # Intersection of all result sets
        if result_sets:
            return set.intersection(*result_sets)
        
        return set()
    
    def _matches_criteria(self, key: str, item: Dict[str, Any], 
                         criteria: Dict[str, Any]) -> bool:
        """Check if an item matches the search criteria."""
        for field, condition in criteria.items():
            if field == "key":
                if not self._matches_string(key, condition):
                    return False
            elif field == "value":
                if not self._matches_string(str(item["value"]), condition):
                    return False
            elif field == "metadata":
                if not self._matches_metadata(item["metadata"], condition):
                    return False
            elif field == "created_after":
                if item["created_at"] < condition:
                    return False
            elif field == "created_before":
                if item["created_at"] > condition:
                    return False
            elif field == "updated_after":
                if item["updated_at"] < condition:
                    return False
            elif field == "updated_before":
                if item["updated_at"] > condition:
                    return False
            elif field == "min_access_count":
                if item["access_count"] < condition:
                    return False
        
        return True
    
    def _matches_string(self, text: str, condition: Union[str, List[str]]) -> bool:
        """Check if text matches string condition."""
        if isinstance(condition, str):
            return condition.lower() in text.lower()
        elif isinstance(condition, list):
            return any(cond.lower() in text.lower() for cond in condition)
        
        return False
    
    def _matches_metadata(self, metadata: Dict[str, Any], 
                         condition: Dict[str, Any]) -> bool:
        """Check if metadata matches condition."""
        for key, value in condition.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        with self._lock:
            access_counts = [item.get("access_count", 0) for item in self._store.values()]
            
            return {
                "size": len(self._store),
                "storage_path": str(self.storage_path),
                "auto_save": self.auto_save,
                "index_search": self.index_search,
                "index_size": len(self._index),
                "total_access_count": sum(access_counts),
                "average_access_count": sum(access_counts) / len(access_counts) if access_counts else 0,
                "most_accessed_count": max(access_counts) if access_counts else 0
            }
