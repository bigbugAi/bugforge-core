"""
Short-term memory implementation for recent information storage.

ShortTermMemory provides temporary storage for recent observations,
actions, and other information with automatic expiration and cleanup.
"""

from collections import deque
from datetime import datetime, timedelta
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

from ..interfaces.memory_store import MemoryStore
from ..primitives import Observation, Action, State


class ShortTermMemory(MemoryStore):
    """
    Short-term memory with time-based expiration and limited capacity.
    
    Stores recent information with automatic cleanup of expired items
    and capacity management through eviction policies.
    """
    
    def __init__(self, max_capacity: int = 1000, 
                 max_age_seconds: int = 300,
                 eviction_policy: str = "fifo"):
        """
        Initialize short-term memory.
        
        Args:
            max_capacity: Maximum number of items to store
            max_age_seconds: Maximum age of items before expiration
            eviction_policy: Eviction policy ("fifo", "lru", "time")
        """
        self.max_capacity = max_capacity
        self.max_age_seconds = max_age_seconds
        self.eviction_policy = eviction_policy
        
        self._store: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, datetime] = {}
        self._insertion_times: Dict[str, datetime] = {}
        self._lock = RLock()
    
    def put(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store a value with automatic cleanup and capacity management.
        
        Args:
            key: Unique identifier for the memory
            value: The value to store
            metadata: Additional metadata (optional)
            
        Returns:
            True if storage was successful
        """
        with self._lock:
            now = datetime.now()
            
            # Clean up expired items
            self._cleanup_expired()
            
            # Handle capacity limits
            if len(self._store) >= self.max_capacity and key not in self._store:
                self._evict_one()
            
            # Store the item
            self._store[key] = {
                "value": value,
                "metadata": metadata or {},
                "created_at": now
            }
            self._access_times[key] = now
            self._insertion_times[key] = now
            
            return True
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value by key, updating access time.
        
        Args:
            key: The key to retrieve
            
        Returns:
            The stored value, or None if not found or expired
        """
        with self._lock:
            if key not in self._store:
                return None
            
            # Check if expired
            if self._is_expired(key):
                del self._store[key]
                del self._access_times[key]
                del self._insertion_times[key]
                return None
            
            # Update access time
            self._access_times[key] = datetime.now()
            
            return self._store[key]["value"]
    
    def query(self, query: Union[str, Dict[str, Any]], 
              limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Query memories based on criteria.
        
        Args:
            query: Query string or dictionary of criteria
            limit: Maximum number of results (optional)
            
        Returns:
            List of matching memory entries
        """
        with self._lock:
            self._cleanup_expired()
            
            results = []
            
            for key, item in self._store.items():
                if self._matches_query(key, item, query):
                    results.append({
                        "key": key,
                        "value": item["value"],
                        "metadata": item["metadata"],
                        "created_at": item["created_at"],
                        "last_accessed": self._access_times[key]
                    })
                    
                    if limit and len(results) >= limit:
                        break
            
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
            if key in self._store:
                del self._store[key]
                del self._access_times[key]
                del self._insertion_times[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all memories from the store."""
        with self._lock:
            self._store.clear()
            self._access_times.clear()
            self._insertion_times.clear()
    
    def size(self) -> int:
        """
        Get the number of stored memories.
        
        Returns:
            Number of memories in the store
        """
        with self._lock:
            self._cleanup_expired()
            return len(self._store)
    
    def get_capabilities(self) -> List[str]:
        """
        Get the memory store's declared capabilities.
        
        Returns:
            List of capability identifiers
        """
        return ["time_based_expiration", "capacity_limiting", "access_tracking"]
    
    def get_recent_items(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recently accessed items.
        
        Args:
            count: Number of items to retrieve
            
        Returns:
            List of recent items
        """
        with self._lock:
            self._cleanup_expired()
            
            # Sort by access time
            sorted_items = sorted(
                self._store.items(),
                key=lambda x: self._access_times[x[0]],
                reverse=True
            )
            
            results = []
            for key, item in sorted_items[:count]:
                results.append({
                    "key": key,
                    "value": item["value"],
                    "metadata": item["metadata"],
                    "created_at": item["created_at"],
                    "last_accessed": self._access_times[key]
                })
            
            return results
    
    def get_items_by_age(self, max_age_seconds: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get items within a specific age range.
        
        Args:
            max_age_seconds: Maximum age in seconds (uses default if None)
            
        Returns:
            List of items within the age range
        """
        with self._lock:
            age_limit = max_age_seconds or self.max_age_seconds
            cutoff_time = datetime.now() - timedelta(seconds=age_limit)
            
            results = []
            for key, item in self._store.items():
                if item["created_at"] >= cutoff_time:
                    results.append({
                        "key": key,
                        "value": item["value"],
                        "metadata": item["metadata"],
                        "created_at": item["created_at"],
                        "last_accessed": self._access_times[key]
                    })
            
            return results
    
    def _cleanup_expired(self) -> None:
        """Remove expired items from the store."""
        now = datetime.now()
        expired_keys = []
        
        for key in list(self._store.keys()):
            if self._is_expired(key):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._store[key]
            del self._access_times[key]
            del self._insertion_times[key]
    
    def _is_expired(self, key: str) -> bool:
        """Check if an item is expired."""
        if key not in self._insertion_times:
            return True
        
        age = datetime.now() - self._insertion_times[key]
        return age.total_seconds() > self.max_age_seconds
    
    def _evict_one(self) -> None:
        """Evict one item according to the eviction policy."""
        if not self._store:
            return
        
        if self.eviction_policy == "fifo":
            # First-in-first-out: remove oldest by insertion time
            oldest_key = min(self._insertion_times, key=self._insertion_times.get)
        elif self.eviction_policy == "lru":
            # Least-recently-used: remove oldest by access time
            oldest_key = min(self._access_times, key=self._access_times.get)
        else:  # time
            # Time-based: remove oldest by insertion time (same as FIFO)
            oldest_key = min(self._insertion_times, key=self._insertion_times.get)
        
        del self._store[oldest_key]
        del self._access_times[oldest_key]
        del self._insertion_times[oldest_key]
    
    def _matches_query(self, key: str, item: Dict[str, Any], 
                      query: Union[str, Dict[str, Any]]) -> bool:
        """Check if an item matches the query criteria."""
        if isinstance(query, str):
            # Text search in key, metadata, or string representation of value
            query_lower = query.lower()
            return (
                query_lower in key.lower() or
                any(query_lower in str(v).lower() for v in item["metadata"].values()) or
                query_lower in str(item["value"]).lower()
            )
        elif isinstance(query, dict):
            # Dictionary-based criteria matching
            for field, criteria in query.items():
                if field == "key":
                    if criteria not in key:
                        return False
                elif field == "metadata":
                    if not all(k in item["metadata"] and item["metadata"][k] == v 
                              for k, v in criteria.items()):
                        return False
                elif field == "age":
                    max_age = criteria
                    age_seconds = (datetime.now() - item["created_at"]).total_seconds()
                    if age_seconds > max_age:
                        return False
            return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        with self._lock:
            self._cleanup_expired()
            
            now = datetime.now()
            ages = [(now - item["created_at"]).total_seconds() 
                   for item in self._store.values()]
            
            return {
                "size": len(self._store),
                "max_capacity": self.max_capacity,
                "max_age_seconds": self.max_age_seconds,
                "eviction_policy": self.eviction_policy,
                "average_age": sum(ages) / ages if ages else 0,
                "oldest_item_age": max(ages) if ages else 0,
                "newest_item_age": min(ages) if ages else 0
            }
