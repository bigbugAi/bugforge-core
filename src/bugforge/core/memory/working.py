"""
Working memory implementation for active information processing.

WorkingMemory provides a limited-capacity, actively maintained memory
store for information currently being processed.
"""

import math
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from ..interfaces.memory_store import WorkingMemoryStore


class WorkingMemory(WorkingMemoryStore):
    """
    Working memory with limited capacity and active maintenance.
    
    Provides temporary storage for information currently being
    processed with various eviction policies and attention mechanisms.
    """
    
    def __init__(self, capacity: int = 7,  # Miller's magical number
                 eviction_policy: str = "lru",
                 decay_time_seconds: int = 30):
        """
        Initialize working memory.
        
        Args:
            capacity: Maximum number of items to maintain
            eviction_policy: Eviction policy ("lru", "lfu", "fifo", "random")
            decay_time_seconds: Time before items start to decay
        """
        self.capacity = capacity
        self.eviction_policy = eviction_policy
        self.decay_time_seconds = decay_time_seconds
        
        self._store: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._access_counts: Dict[str, int] = {}
        self._last_accessed: Dict[str, datetime] = {}
        self._creation_times: Dict[str, datetime] = {}
    
    def put(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store a value with capacity management.
        
        Args:
            key: Unique identifier for the memory
            value: The value to store
            metadata: Additional metadata (optional)
            
        Returns:
            True if storage was successful
        """
        now = datetime.now()
        
        # Handle existing key
        if key in self._store:
            self._store.move_to_end(key)  # Move to most recent
            self._store[key].update({
                "value": value,
                "metadata": metadata or {},
                "updated_at": now
            })
            self._access_counts[key] += 1
            self._last_accessed[key] = now
            return True
        
        # Handle capacity limit
        if len(self._store) >= self.capacity:
            self._evict_one()
        
        # Add new item
        self._store[key] = {
            "value": value,
            "metadata": metadata or {},
            "created_at": now,
            "updated_at": now
        }
        self._access_counts[key] = 1
        self._last_accessed[key] = now
        self._creation_times[key] = now
        
        return True
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value by key, updating access statistics.
        
        Args:
            key: The key to retrieve
            
        Returns:
            The stored value, or None if not found
        """
        if key not in self._store:
            return None
        
        # Update access statistics
        self._store.move_to_end(key)
        self._access_counts[key] += 1
        self._last_accessed[key] = datetime.now()
        
        return self._store[key]["value"]
    
    def query(self, query: Union[str, Dict[str, Any]], 
              limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Query working memory based on criteria.
        
        Args:
            query: Query string or dictionary of criteria
            limit: Maximum number of results (optional)
            
        Returns:
            List of matching memory entries
        """
        results = []
        
        for key, item in self._store.items():
            if self._matches_query(key, item, query):
                results.append({
                    "key": key,
                    "value": item["value"],
                    "metadata": item["metadata"],
                    "created_at": item["created_at"],
                    "updated_at": item["updated_at"],
                    "access_count": self._access_counts[key],
                    "last_accessed": self._last_accessed[key],
                    "priority": self._compute_priority(key)
                })
        
        # Sort by priority (highest first)
        results.sort(key=lambda x: x["priority"], reverse=True)
        
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
        if key in self._store:
            del self._store[key]
            del self._access_counts[key]
            del self._last_accessed[key]
            del self._creation_times[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all memories from working memory."""
        self._store.clear()
        self._access_counts.clear()
        self._last_accessed.clear()
        self._creation_times.clear()
    
    def size(self) -> int:
        """
        Get the number of stored memories.
        
        Returns:
            Number of memories in working memory
        """
        return len(self._store)
    
    def get_capacity(self) -> int:
        """
        Get the memory capacity.
        
        Returns:
            Maximum number of items that can be stored
        """
        return self.capacity
    
    def set_capacity(self, capacity: int) -> None:
        """
        Set the memory capacity.
        
        Args:
            capacity: New maximum capacity
        """
        self.capacity = capacity
        
        # Evict items if over new capacity
        while len(self._store) > self.capacity:
            self._evict_one()
    
    def get_usage(self) -> int:
        """
        Get current memory usage.
        
        Returns:
            Number of items currently stored
        """
        return len(self._store)
    
    def set_eviction_policy(self, policy: str) -> None:
        """
        Set the eviction policy.
        
        Args:
            policy: Eviction policy ("lru", "lfu", "random", "fifo")
        """
        if policy in ["lru", "lfu", "random", "fifo"]:
            self.eviction_policy = policy
        else:
            raise ValueError(f"Invalid eviction policy: {policy}")
    
    def is_full(self) -> bool:
        """
        Check if working memory is at capacity.
        
        Returns:
            True if at capacity
        """
        return len(self._store) >= self.capacity
    
    def get_utilization(self) -> float:
        """
        Get memory utilization ratio.
        
        Returns:
            Utilization ratio (0.0 to 1.0)
        """
        return len(self._store) / self.capacity if self.capacity > 0 else 0.0
    
    def get_capabilities(self) -> List[str]:
        """
        Get the memory store's declared capabilities.
        
        Returns:
            List of capability identifiers
        """
        return ["limited_capacity", "eviction_policies", "access_tracking", "priority_management"]
    
    def get_active_items(self) -> List[Dict[str, Any]]:
        """
        Get currently active items (non-decayed).
        
        Returns:
            List of active items
        """
        now = datetime.now()
        active_items = []
        
        for key, item in self._store.items():
            if not self._is_decayed(key, now):
                active_items.append({
                    "key": key,
                    "value": item["value"],
                    "metadata": item["metadata"],
                    "priority": self._compute_priority(key),
                    "decay_level": self._compute_decay_level(key, now)
                })
        
        return active_items
    
    def get_high_priority_items(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Get items above priority threshold.
        
        Args:
            threshold: Priority threshold (0.0 to 1.0)
            
        Returns:
            List of high-priority items
        """
        results = []
        
        for key, item in self._store.items():
            priority = self._compute_priority(key)
            if priority >= threshold:
                results.append({
                    "key": key,
                    "value": item["value"],
                    "metadata": item["metadata"],
                    "priority": priority
                })
        
        # Sort by priority
        results.sort(key=lambda x: x["priority"], reverse=True)
        
        return results
    
    def refresh_item(self, key: str) -> bool:
        """
        Refresh an item to prevent decay.
        
        Args:
            key: Key of the item to refresh
            
        Returns:
            True if refresh was successful
        """
        if key not in self._store:
            return False
        
        self._last_accessed[key] = datetime.now()
        self._store.move_to_end(key)
        
        return True
    
    def decay_all(self) -> None:
        """Apply decay to all items and remove heavily decayed items."""
        now = datetime.now()
        keys_to_remove = []
        
        for key in self._store.keys():
            if self._is_heavily_decayed(key, now):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self.delete(key)
    
    def get_attention_focus(self) -> List[str]:
        """
        Get items currently in attention focus.
        
        Returns:
            List of keys in attention focus (highest priority items)
        """
        # Top 3-4 items (typical attention span)
        focus_size = min(4, len(self._store))
        
        items_by_priority = sorted(
            self._store.keys(),
            key=lambda k: self._compute_priority(k),
            reverse=True
        )
        
        return items_by_priority[:focus_size]
    
    def _evict_one(self) -> None:
        """Evict one item according to the eviction policy."""
        if not self._store:
            return
        
        if self.eviction_policy == "lru":
            # Least recently used
            oldest_key = next(iter(self._store))
        elif self.eviction_policy == "lfu":
            # Least frequently used
            oldest_key = min(self._access_counts, key=self._access_counts.get)
        elif self.eviction_policy == "fifo":
            # First in, first out
            oldest_key = min(self._creation_times, key=self._creation_times.get)
        else:  # random
            import random
            oldest_key = random.choice(list(self._store.keys()))
        
        self.delete(oldest_key)
    
    def _compute_priority(self, key: str) -> float:
        """
        Compute priority score for an item.
        
        Args:
            key: Key of the item
            
        Returns:
            Priority score (0.0 to 1.0)
        """
        if key not in self._store:
            return 0.0
        
        now = datetime.now()
        
        # Factors: recency, frequency, decay
        recency_score = self._compute_recency_score(key, now)
        frequency_score = self._compute_frequency_score(key)
        decay_score = 1.0 - self._compute_decay_level(key, now)
        
        # Weighted combination
        priority = (0.4 * recency_score + 0.4 * frequency_score + 0.2 * decay_score)
        
        return min(1.0, max(0.0, priority))
    
    def _compute_recency_score(self, key: str, now: datetime) -> float:
        """Compute recency score based on last access time."""
        if key not in self._last_accessed:
            return 0.0
        
        time_since_access = (now - self._last_accessed[key]).total_seconds()
        
        # Exponential decay with time constant of 10 seconds
        time_constant = 10.0
        return math.exp(-time_since_access / time_constant)
    
    def _compute_frequency_score(self, key: str) -> float:
        """Compute frequency score based on access count."""
        if key not in self._access_counts:
            return 0.0
        
        # Normalize by total accesses
        total_accesses = sum(self._access_counts.values())
        if total_accesses == 0:
            return 0.0
        
        return self._access_counts[key] / total_accesses
    
    def _compute_decay_level(self, key: str, now: datetime) -> float:
        """
        Compute decay level (0.0 = no decay, 1.0 = fully decayed).
        
        Args:
            key: Key of the item
            now: Current time
            
        Returns:
            Decay level (0.0 to 1.0)
        """
        if key not in self._last_accessed:
            return 1.0
        
        time_since_access = (now - self._last_accessed[key]).total_seconds()
        
        if time_since_access < self.decay_time_seconds:
            return 0.0
        
        # Linear decay after decay time
        decay_duration = self.decay_time_seconds * 2  # Full decay after 2x decay time
        decay_level = min(1.0, (time_since_access - self.decay_time_seconds) / decay_duration)
        
        return decay_level
    
    def _is_decayed(self, key: str, now: datetime) -> bool:
        """Check if an item has started to decay."""
        return self._compute_decay_level(key, now) > 0.0
    
    def _is_heavily_decayed(self, key: str, now: datetime) -> bool:
        """Check if an item is heavily decayed (should be removed)."""
        return self._compute_decay_level(key, now) > 0.8
    
    def _matches_query(self, key: str, item: Dict[str, Any], 
                      query: Union[str, Dict[str, Any]]) -> bool:
        """Check if an item matches the query criteria."""
        if isinstance(query, str):
            # Text search
            query_lower = query.lower()
            return (
                query_lower in key.lower() or
                any(query_lower in str(v).lower() for v in item["metadata"].values()) or
                query_lower in str(item["value"]).lower()
            )
        elif isinstance(query, dict):
            # Criteria-based search
            for field, condition in query.items():
                if field == "key":
                    if condition not in key:
                        return False
                elif field == "metadata":
                    if not all(k in item["metadata"] and item["metadata"][k] == v 
                              for k, v in condition.items()):
                        return False
                elif field == "min_priority":
                    if self._compute_priority(key) < condition:
                        return False
            return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get working memory statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        if not self._store:
            return {
                "size": 0,
                "capacity": self.capacity,
                "utilization": 0.0,
                "eviction_policy": self.eviction_policy
            }
        
        priorities = [self._compute_priority(key) for key in self._store.keys()]
        access_counts = list(self._access_counts.values())
        
        return {
            "size": len(self._store),
            "capacity": self.capacity,
            "utilization": len(self._store) / self.capacity,
            "eviction_policy": self.eviction_policy,
            "average_priority": sum(priorities) / len(priorities),
            "min_priority": min(priorities),
            "max_priority": max(priorities),
            "total_accesses": sum(access_counts),
            "average_accesses": sum(access_counts) / len(access_counts),
            "attention_focus": self.get_attention_focus()
        }
