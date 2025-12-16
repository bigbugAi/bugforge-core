"""
MemoryStore interface - defines the contract for memory storage and retrieval.

MemoryStores are components that can store, retrieve, and manage
memories of various types and durations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from ..primitives import Observation, Action, State, Trajectory


class MemoryStore(ABC):
    """
    Abstract base class for all memory stores in BugForge.
    
    Memory stores are responsible for storing, retrieving, and managing
    memories with different retention policies and access patterns.
    """
    
    @abstractmethod
    def put(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store a value with associated key and metadata.
        
        Args:
            key: Unique identifier for the memory
            value: The value to store
            metadata: Additional metadata (optional)
            
        Returns:
            True if storage was successful
        """
        pass
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value by key.
        
        Args:
            key: The key to retrieve
            
        Returns:
            The stored value, or None if not found
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete a memory by key.
        
        Args:
            key: The key to delete
            
        Returns:
            True if deletion was successful
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all memories from the store."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """
        Get the number of stored memories.
        
        Returns:
            Number of memories in the store
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Get the memory store's declared capabilities.
        
        Returns:
            List of capability identifiers
        """
        pass
    
    def batch_put(self, items: Dict[str, Any], 
                 metadata: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
        """
        Store multiple items.
        
        Args:
            items: Dictionary of key-value pairs to store
            metadata: Additional metadata for all items (optional)
            
        Returns:
            Dictionary mapping keys to success status
        """
        results = {}
        for key, value in items.items():
            results[key] = self.put(key, value, metadata)
        return results
    
    def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """
        Retrieve multiple items by keys.
        
        Args:
            keys: List of keys to retrieve
            
        Returns:
            Dictionary mapping found keys to their values
        """
        results = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                results[key] = value
        return results
    
    def exists(self, key: str) -> bool:
        """
        Check if a key exists in the store.
        
        Args:
            key: The key to check
            
        Returns:
            True if the key exists
        """
        return self.get(key) is not None
    
    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a stored item.
        
        Args:
            key: The key to get metadata for
            
        Returns:
            Metadata dictionary, or None if not found
        """
        return None
    
    def update_metadata(self, key: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a stored item.
        
        Args:
            key: The key to update metadata for
            metadata: New metadata to merge
            
        Returns:
            True if update was successful
        """
        return False
    
    def keys(self) -> List[str]:
        """
        Get all keys in the store.
        
        Returns:
            List of all keys
        """
        return []
    
    def values(self) -> List[Any]:
        """
        Get all values in the store.
        
        Returns:
            List of all values
        """
        keys = self.keys()
        return [self.get(key) for key in keys]
    
    def items(self) -> List[tuple]:
        """
        Get all key-value pairs in the store.
        
        Returns:
            List of (key, value) tuples
        """
        keys = self.keys()
        return [(key, self.get(key)) for key in keys]
    
    def save(self, path: str) -> None:
        """
        Save memory store to disk.
        
        Args:
            path: Path where to save the store
        """
        raise NotImplementedError("MemoryStore saving not implemented")
    
    def load(self, path: str) -> None:
        """
        Load memory store from disk.
        
        Args:
            path: Path from which to load the store
        """
        raise NotImplementedError("MemoryStore loading not implemented")
    
    def __len__(self) -> int:
        """Get the number of stored memories."""
        return self.size()
    
    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the store."""
        return self.exists(key)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size()}, capabilities={self.get_capabilities()})"


class EpisodicMemoryStore(MemoryStore):
    """
    Memory store specialized for episodic memories.
    
    Stores and retrieves complete episodes/trajectories with
    temporal structure and context.
    """
    
    @abstractmethod
    def store_episode(self, episode: Trajectory, 
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a complete episode/trajectory.
        
        Args:
            episode: The episode to store
            metadata: Additional metadata (optional)
            
        Returns:
            Unique identifier for the stored episode
        """
        pass
    
    @abstractmethod
    def retrieve_episode(self, episode_id: str) -> Optional[Trajectory]:
        """
        Retrieve an episode by ID.
        
        Args:
            episode_id: The episode identifier
            
        Returns:
            The stored episode, or None if not found
        """
        pass
    
    @abstractmethod
    def query_episodes(self, criteria: Dict[str, Any], 
                      limit: Optional[int] = None) -> List[Trajectory]:
        """
        Query episodes based on criteria.
        
        Args:
            criteria: Query criteria
            limit: Maximum number of results (optional)
            
        Returns:
            List of matching episodes
        """
        pass
    
    @abstractmethod
    def get_similar_episodes(self, query_episode: Trajectory, 
                           similarity_threshold: float = 0.5,
                           limit: Optional[int] = None) -> List[Trajectory]:
        """
        Find episodes similar to a query episode.
        
        Args:
            query_episode: The episode to compare against
            similarity_threshold: Minimum similarity score
            limit: Maximum number of results (optional)
            
        Returns:
            List of similar episodes
        """
        pass
    
    def get_recent_episodes(self, count: int) -> List[Trajectory]:
        """
        Get the most recent episodes.
        
        Args:
            count: Number of recent episodes to retrieve
            
        Returns:
            List of recent episodes
        """
        return self.query_episodes({}, limit=count)
    
    def get_episodes_by_time_range(self, start_time, end_time) -> List[Trajectory]:
        """
        Get episodes within a time range.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            List of episodes in the time range
        """
        criteria = {
            "start_time_gte": start_time,
            "end_time_lte": end_time
        }
        return self.query_episodes(criteria)


class SemanticMemoryStore(MemoryStore):
    """
    Memory store with semantic search capabilities.
    
    Supports storage and retrieval based on semantic similarity
    using embeddings and vector search.
    """
    
    @abstractmethod
    def store_with_embedding(self, key: str, value: Any, 
                           embedding: List[float],
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store a value with its embedding.
        
        Args:
            key: Unique identifier
            value: Value to store
            embedding: Vector embedding of the value
            metadata: Additional metadata (optional)
            
        Returns:
            True if storage was successful
        """
        pass
    
    @abstractmethod
    def semantic_search(self, query_embedding: List[float], 
                       limit: Optional[int] = None,
                       similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Search for items by semantic similarity.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results (optional)
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of matching items with similarity scores
        """
        pass
    
    @abstractmethod
    def get_embedding(self, key: str) -> Optional[List[float]]:
        """
        Get the embedding for a stored item.
        
        Args:
            key: The key to get embedding for
            
        Returns:
            Embedding vector, or None if not found
        """
        pass
    
    def semantic_query(self, query_text: str, 
                      encoder: "Encoder",
                      limit: Optional[int] = None,
                      similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Semantic search using text query and encoder.
        
        Args:
            query_text: Text query
            encoder: Encoder to generate embeddings
            limit: Maximum number of results (optional)
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of matching items with similarity scores
        """
        query_embedding = encoder.encode(query_text)
        return self.semantic_search(query_embedding, limit, similarity_threshold)


class WorkingMemoryStore(MemoryStore):
    """
    Memory store with limited capacity and automatic eviction.
    
    Implements working memory with fixed capacity and various
    eviction policies (LRU, LFU, random, etc.).
    """
    
    @abstractmethod
    def get_capacity(self) -> int:
        """
        Get the memory capacity.
        
        Returns:
            Maximum number of items that can be stored
        """
        pass
    
    @abstractmethod
    def set_capacity(self, capacity: int) -> None:
        """
        Set the memory capacity.
        
        Args:
            capacity: New maximum capacity
        """
        pass
    
    @abstractmethod
    def get_usage(self) -> int:
        """
        Get current memory usage.
        
        Returns:
            Number of items currently stored
        """
        pass
    
    @abstractmethod
    def set_eviction_policy(self, policy: str) -> None:
        """
        Set the eviction policy.
        
        Args:
            policy: Eviction policy ("lru", "lfu", "random", "fifo")
        """
        pass
    
    def is_full(self) -> bool:
        """
        Check if the memory store is at capacity.
        
        Returns:
            True if at capacity
        """
        return self.get_usage() >= self.get_capacity()
    
    def get_utilization(self) -> float:
        """
        Get memory utilization ratio.
        
        Returns:
            Utilization ratio (0.0 to 1.0)
        """
        return self.get_usage() / self.get_capacity() if self.get_capacity() > 0 else 0.0
    
    def put(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store a value with automatic eviction if needed.
        
        Args:
            key: Unique identifier
            value: Value to store
            metadata: Additional metadata (optional)
            
        Returns:
            True if storage was successful
        """
        # If key already exists, update it
        if self.exists(key):
            return super().put(key, value, metadata)
        
        # If full, need to evict first (implementation-specific)
        if self.is_full():
            self._evict_one()
        
        return super().put(key, value, metadata)
    
    @abstractmethod
    def _evict_one(self) -> None:
        """Evict one item according to the eviction policy."""
        pass
