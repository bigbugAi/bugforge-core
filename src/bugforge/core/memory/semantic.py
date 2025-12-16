"""
Semantic memory implementation for embedding-based storage and retrieval.

SemanticMemory provides storage with semantic search capabilities
using embeddings and vector similarity.
"""

import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from ..interfaces.memory_store import SemanticMemoryStore
from ..interfaces.encoder import Encoder


class SemanticMemory(SemanticMemoryStore):
    """
    Semantic memory with embedding-based storage and retrieval.
    
    Stores items with their embeddings and provides semantic
    search capabilities using vector similarity.
    """
    
    def __init__(self, encoder: Encoder, 
                 similarity_threshold: float = 0.5,
                 max_items: Optional[int] = None):
        """
        Initialize semantic memory.
        
        Args:
            encoder: Encoder for generating embeddings
            similarity_threshold: Minimum similarity for semantic search
            max_items: Maximum number of items to store
        """
        self.encoder = encoder
        self.similarity_threshold = similarity_threshold
        self.max_items = max_items
        
        self._store: Dict[str, Dict[str, Any]] = {}
        self._embeddings: Dict[str, List[float]] = {}
    
    def put(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store a value with its embedding.
        
        Args:
            key: Unique identifier for the memory
            value: The value to store
            metadata: Additional metadata (optional)
            
        Returns:
            True if storage was successful
        """
        # Check capacity limit
        if self.max_items and len(self._store) >= self.max_items and key not in self._store:
            self._evict_one()
        
        # Generate embedding
        try:
            embedding = self.encoder.encode(value)
        except Exception as e:
            raise ValueError(f"Failed to encode value: {e}")
        
        # Store value and embedding
        self._store[key] = {
            "value": value,
            "metadata": metadata or {},
            "created_at": datetime.now()
        }
        self._embeddings[key] = embedding
        
        return True
    
    def store_with_embedding(self, key: str, value: Any, 
                           embedding: List[float],
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store a value with pre-computed embedding.
        
        Args:
            key: Unique identifier
            value: Value to store
            embedding: Vector embedding of the value
            metadata: Additional metadata (optional)
            
        Returns:
            True if storage was successful
        """
        # Validate embedding dimension
        if not self.encoder.validate_embedding(embedding):
            raise ValueError(f"Invalid embedding dimension: expected {self.encoder.get_embedding_dimension()}")
        
        # Check capacity limit
        if self.max_items and len(self._store) >= self.max_items and key not in self._store:
            self._evict_one()
        
        # Store value and embedding
        self._store[key] = {
            "value": value,
            "metadata": metadata or {},
            "created_at": datetime.now()
        }
        self._embeddings[key] = embedding
        
        return True
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value by key.
        
        Args:
            key: The key to retrieve
            
        Returns:
            The stored value, or None if not found
        """
        if key in self._store:
            return self._store[key]["value"]
        return None
    
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
        results = []
        
        if isinstance(query, str):
            # Semantic search using text query
            query_embedding = self.encoder.encode(query)
            results = self.semantic_search(query_embedding, limit, self.similarity_threshold)
        elif isinstance(query, dict):
            # Criteria-based search
            for key, item in self._store.items():
                if self._matches_criteria(key, item, query):
                    embedding = self._embeddings[key]
                    results.append({
                        "key": key,
                        "value": item["value"],
                        "metadata": item["metadata"],
                        "embedding": embedding,
                        "created_at": item["created_at"]
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
        if key in self._store:
            del self._store[key]
            del self._embeddings[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all memories from the store."""
        self._store.clear()
        self._embeddings.clear()
    
    def size(self) -> int:
        """
        Get the number of stored memories.
        
        Returns:
            Number of memories in the store
        """
        return len(self._store)
    
    def get_capabilities(self) -> List[str]:
        """
        Get the memory store's declared capabilities.
        
        Returns:
            List of capability identifiers
        """
        return ["semantic_search", "embedding_storage", "vector_similarity", "encoder_integration"]
    
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
        results = []
        
        for key, embedding in self._embeddings.items():
            similarity = self.encoder.compute_similarity(query_embedding, embedding)
            
            if similarity >= similarity_threshold:
                item = self._store[key]
                results.append({
                    "key": key,
                    "value": item["value"],
                    "metadata": item["metadata"],
                    "embedding": embedding,
                    "similarity": similarity,
                    "created_at": item["created_at"]
                })
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        if limit:
            results = results[:limit]
        
        return results
    
    def get_embedding(self, key: str) -> Optional[List[float]]:
        """
        Get the embedding for a stored item.
        
        Args:
            key: The key to get embedding for
            
        Returns:
            Embedding vector, or None if not found
        """
        return self._embeddings.get(key)
    
    def find_similar_items(self, query_key: str, 
                          limit: Optional[int] = None,
                          similarity_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Find items similar to a stored item.
        
        Args:
            query_key: Key of the item to compare against
            limit: Maximum number of results (optional)
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar items with similarity scores
        """
        if query_key not in self._embeddings:
            return []
        
        query_embedding = self._embeddings[query_key]
        threshold = similarity_threshold or self.similarity_threshold
        
        # Exclude the query item from results
        all_results = self.semantic_search(query_embedding, None, threshold)
        
        # Filter out the query item
        results = [r for r in all_results if r["key"] != query_key]
        
        if limit:
            results = results[:limit]
        
        return results
    
    def get_nearest_neighbors(self, query_key: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Get k nearest neighbors for a stored item.
        
        Args:
            query_key: Key of the item to find neighbors for
            k: Number of neighbors to find
            
        Returns:
            List of k nearest neighbors with similarity scores
        """
        if query_key not in self._embeddings:
            return []
        
        query_embedding = self._embeddings[query_key]
        
        # Compute similarities to all items
        similarities = []
        for key, embedding in self._embeddings.items():
            if key == query_key:
                continue
            
            similarity = self.encoder.compute_similarity(query_embedding, embedding)
            similarities.append((key, similarity))
        
        # Sort by similarity and take top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        neighbors = similarities[:k]
        
        # Format results
        results = []
        for key, similarity in neighbors:
            item = self._store[key]
            results.append({
                "key": key,
                "value": item["value"],
                "metadata": item["metadata"],
                "embedding": self._embeddings[key],
                "similarity": similarity,
                "created_at": item["created_at"]
            })
        
        return results
    
    def cluster_items(self, num_clusters: int = 5) -> Dict[int, List[str]]:
        """
        Cluster stored items by embedding similarity.
        
        Args:
            num_clusters: Number of clusters to create
            
        Returns:
            Dictionary mapping cluster IDs to lists of item keys
        """
        if len(self._embeddings) < num_clusters:
            # Return single cluster if not enough items
            return {0: list(self._embeddings.keys())}
        
        # Simple k-means clustering implementation
        import random
        
        keys = list(self._embeddings.keys())
        embeddings = [self._embeddings[key] for key in keys]
        
        # Initialize centroids randomly
        centroid_indices = random.sample(range(len(embeddings)), num_clusters)
        centroids = [embeddings[i] for i in centroid_indices]
        
        # Assign items to clusters
        clusters = {i: [] for i in range(num_clusters)}
        
        for key, embedding in self._embeddings.items():
            # Find nearest centroid
            max_similarity = -1
            best_cluster = 0
            
            for i, centroid in enumerate(centroids):
                similarity = self.encoder.compute_similarity(embedding, centroid)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_cluster = i
            
            clusters[best_cluster].append(key)
        
        return clusters
    
    def get_embedding_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored embeddings.
        
        Returns:
            Dictionary with embedding statistics
        """
        if not self._embeddings:
            return {}
        
        embeddings = list(self._embeddings.values())
        dimension = len(embeddings[0])
        
        # Compute statistics for each dimension
        dim_stats = []
        for dim in range(dimension):
            values = [emb[dim] for emb in embeddings]
            dim_stats.append({
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "std": math.sqrt(sum((v - sum(values)/len(values))**2 for v in values) / len(values))
            })
        
        return {
            "num_embeddings": len(embeddings),
            "embedding_dimension": dimension,
            "dimension_statistics": dim_stats
        }
    
    def _evict_one(self) -> None:
        """Evict one item to make space."""
        if not self._store:
            return
        
        # Simple FIFO eviction (oldest first)
        oldest_key = min(self._store.keys(), 
                        key=lambda k: self._store[k]["created_at"])
        
        del self._store[oldest_key]
        del self._embeddings[oldest_key]
    
    def _matches_criteria(self, key: str, item: Dict[str, Any], 
                         criteria: Dict[str, Any]) -> bool:
        """Check if an item matches the search criteria."""
        for field, condition in criteria.items():
            if field == "key":
                if condition not in key:
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
        
        return True
    
    def _matches_metadata(self, metadata: Dict[str, Any], 
                         condition: Dict[str, Any]) -> bool:
        """Check if metadata matches condition."""
        for key, value in condition.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
