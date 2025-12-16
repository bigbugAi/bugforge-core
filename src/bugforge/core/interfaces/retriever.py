"""
Retriever interface - defines the contract for information retrieval.

Retrievers are components that can find and retrieve relevant information
from various sources and memory stores.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from ..primitives import Observation, Action, State


class Retriever(ABC):
    """
    Abstract base class for all retrievers in BugForge.
    
    Retrievers are responsible for finding and retrieving relevant
    information from memory stores, databases, or external sources.
    """
    
    @abstractmethod
    def retrieve(self, query: Union[str, Dict[str, Any], Observation], 
                context: Optional[Dict[str, Any]] = None,
                limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant information based on a query.
        
        Args:
            query: The retrieval query (text, dict, or observation)
            context: Additional context for retrieval (optional)
            limit: Maximum number of results (optional)
            
        Returns:
            List of retrieved items with relevance scores
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Get the retriever's declared capabilities.
        
        Returns:
            List of capability identifiers
        """
        pass
    
    @abstractmethod
    def get_supported_query_types(self) -> List[str]:
        """
        Get the types of queries this retriever supports.
        
        Returns:
            List of supported query types
        """
        pass
    
    def batch_retrieve(self, queries: List[Union[str, Dict[str, Any], Observation]], 
                      context: Optional[Dict[str, Any]] = None,
                      limit: Optional[int] = None) -> List[List[Dict[str, Any]]]:
        """
        Retrieve information for multiple queries.
        
        Args:
            queries: List of retrieval queries
            context: Additional context for retrieval (optional)
            limit: Maximum number of results per query (optional)
            
        Returns:
            List of result lists, one for each query
        """
        return [self.retrieve(query, context, limit) for query in queries]
    
    def retrieve_with_scores(self, query: Union[str, Dict[str, Any], Observation], 
                           context: Optional[Dict[str, Any]] = None,
                           limit: Optional[int] = None,
                           min_score: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Retrieve information with relevance scores and filtering.
        
        Args:
            query: The retrieval query
            context: Additional context (optional)
            limit: Maximum number of results (optional)
            min_score: Minimum relevance score (optional)
            
        Returns:
            List of retrieved items with relevance scores
        """
        results = self.retrieve(query, context, limit)
        
        if min_score is not None:
            results = [r for r in results if r.get("score", 0) >= min_score]
        
        return results
    
    def get_relevance_score(self, query: Union[str, Dict[str, Any], Observation], 
                           item: Dict[str, Any]) -> float:
        """
        Compute relevance score for a query-item pair.
        
        Args:
            query: The retrieval query
            item: The item to score
            
        Returns:
            Relevance score (higher = more relevant)
        """
        return 0.0
    
    def explain_retrieval(self, query: Union[str, Dict[str, Any], Observation], 
                         item: Dict[str, Any]) -> Optional[str]:
        """
        Provide explanation for why an item was retrieved.
        
        Args:
            query: The retrieval query
            item: The retrieved item
            
        Returns:
            Explanation string, or None if not available
        """
        return None
    
    def reset(self) -> None:
        """
        Reset retriever state for new sessions.
        
        Called to clear any accumulated state or caches.
        """
        pass
    
    def save(self, path: str) -> None:
        """
        Save retriever state to disk.
        
        Args:
            path: Path where to save the retriever
        """
        raise NotImplementedError("Retriever saving not implemented")
    
    def load(self, path: str) -> None:
        """
        Load retriever state from disk.
        
        Args:
            path: Path from which to load the retriever
        """
        raise NotImplementedError("Retriever loading not implemented")
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(capabilities={self.get_capabilities()})"


class SemanticRetriever(Retriever):
    """
    Retriever that uses semantic similarity for information retrieval.
    
    Uses embeddings and vector similarity to find semantically related
    information from memory stores or databases.
    """
    
    @abstractmethod
    def encode_query(self, query: Union[str, Dict[str, Any], Observation]) -> List[float]:
        """
        Encode a query into an embedding vector.
        
        Args:
            query: The query to encode
            
        Returns:
            Embedding vector for the query
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
            List of similar items with similarity scores
        """
        pass
    
    def retrieve(self, query: Union[str, Dict[str, Any], Observation], 
                context: Optional[Dict[str, Any]] = None,
                limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve using semantic similarity.
        
        Args:
            query: The retrieval query
            context: Additional context (optional)
            limit: Maximum number of results (optional)
            
        Returns:
            List of semantically similar items
        """
        query_embedding = self.encode_query(query)
        similarity_threshold = context.get("similarity_threshold", 0.5) if context else 0.5
        return self.semantic_search(query_embedding, limit, similarity_threshold)
    
    def get_supported_query_types(self) -> List[str]:
        """Get supported query types."""
        return ["text", "observation", "embedding"]
    
    def get_relevance_score(self, query: Union[str, Dict[str, Any], Observation], 
                           item: Dict[str, Any]) -> float:
        """
        Get semantic similarity score.
        
        Args:
            query: The retrieval query
            item: The item to score
            
        Returns:
            Semantic similarity score
        """
        query_embedding = self.encode_query(query)
        item_embedding = item.get("embedding")
        if item_embedding is None:
            return 0.0
        
        # Compute cosine similarity
        import math
        dot_product = sum(q * i for q, i in zip(query_embedding, item_embedding))
        query_norm = math.sqrt(sum(q * q for q in query_embedding))
        item_norm = math.sqrt(sum(i * i for i in item_embedding))
        
        if query_norm == 0 or item_norm == 0:
            return 0.0
        
        return dot_product / (query_norm * item_norm)


class KeywordRetriever(Retriever):
    """
    Retriever that uses keyword matching for information retrieval.
    
    Performs keyword-based search and matching to find relevant
    information from text-based sources.
    """
    
    @abstractmethod
    def extract_keywords(self, query: Union[str, Dict[str, Any], Observation]) -> List[str]:
        """
        Extract keywords from a query.
        
        Args:
            query: The query to extract keywords from
            
        Returns:
            List of extracted keywords
        """
        pass
    
    @abstractmethod
    def keyword_search(self, keywords: List[str], 
                      limit: Optional[int] = None,
                      min_matches: int = 1) -> List[Dict[str, Any]]:
        """
        Search for items containing keywords.
        
        Args:
            keywords: List of keywords to search for
            limit: Maximum number of results (optional)
            min_matches: Minimum number of keyword matches required
            
        Returns:
            List of matching items with match counts
        """
        pass
    
    def retrieve(self, query: Union[str, Dict[str, Any], Observation], 
                context: Optional[Dict[str, Any]] = None,
                limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve using keyword matching.
        
        Args:
            query: The retrieval query
            context: Additional context (optional)
            limit: Maximum number of results (optional)
            
        Returns:
            List of keyword-matching items
        """
        keywords = self.extract_keywords(query)
        min_matches = context.get("min_matches", 1) if context else 1
        return self.keyword_search(keywords, limit, min_matches)
    
    def get_supported_query_types(self) -> List[str]:
        """Get supported query types."""
        return ["text", "keywords"]
    
    def get_relevance_score(self, query: Union[str, Dict[str, Any], Observation], 
                           item: Dict[str, Any]) -> float:
        """
        Get keyword match score.
        
        Args:
            query: The retrieval query
            item: The item to score
            
        Returns:
            Keyword match score
        """
        keywords = self.extract_keywords(query)
        item_text = str(item.get("content", "")).lower()
        
        matches = sum(1 for keyword in keywords if keyword.lower() in item_text)
        return matches / len(keywords) if keywords else 0.0


class HybridRetriever(Retriever):
    """
    Retriever that combines multiple retrieval strategies.
    
    Uses multiple retrieval methods and combines their results
    using various fusion strategies.
    """
    
    @abstractmethod
    def add_retriever(self, retriever: Retriever, weight: float = 1.0) -> None:
        """
        Add a retriever to the hybrid system.
        
        Args:
            retriever: The retriever to add
            weight: Weight for this retriever's results
        """
        pass
    
    @abstractmethod
    def remove_retriever(self, retriever: Retriever) -> None:
        """
        Remove a retriever from the hybrid system.
        
        Args:
            retriever: The retriever to remove
        """
        pass
    
    @abstractmethod
    def fuse_results(self, results_list: List[List[Dict[str, Any]]], 
                    weights: List[float]) -> List[Dict[str, Any]]:
        """
        Fuse results from multiple retrievers.
        
        Args:
            results_list: List of result lists from each retriever
            weights: Weights for each retriever's results
            
        Returns:
            Fused list of results
        """
        pass
    
    def retrieve(self, query: Union[str, Dict[str, Any], Observation], 
                context: Optional[Dict[str, Any]] = None,
                limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve using multiple retrievers and fuse results.
        
        Args:
            query: The retrieval query
            context: Additional context (optional)
            limit: Maximum number of results (optional)
            
        Returns:
            Fused list of retrieved items
        """
        # Get results from all retrievers
        results_list = []
        weights = []
        
        for retriever, weight in self.get_retrievers():
            retriever_results = retriever.retrieve(query, context, limit)
            results_list.append(retriever_results)
            weights.append(weight)
        
        return self.fuse_results(results_list, weights)
    
    @abstractmethod
    def get_retrievers(self) -> List[tuple]:
        """
        Get all retrievers and their weights.
        
        Returns:
            List of (retriever, weight) tuples
        """
        pass
    
    def get_supported_query_types(self) -> List[str]:
        """Get supported query types (union of all retrievers)."""
        all_types = set()
        for retriever, _ in self.get_retrievers():
            all_types.update(retriever.get_supported_query_types())
        return list(all_types)
    
    def get_capabilities(self) -> List[str]:
        """Get combined capabilities from all retrievers."""
        all_caps = set()
        for retriever, _ in self.get_retrievers():
            all_caps.update(retriever.get_capabilities())
        return list(all_caps)


class ContextualRetriever(Retriever):
    """
    Retriever that considers context and conversation history.
    
    Incorporates conversation context and temporal information
    to improve retrieval relevance.
    """
    
    @abstractmethod
    def update_context(self, context: Dict[str, Any]) -> None:
        """
        Update the retrieval context.
        
        Args:
            context: New context information
        """
        pass
    
    @abstractmethod
    def get_context(self) -> Dict[str, Any]:
        """
        Get the current retrieval context.
        
        Returns:
            Current context dictionary
        """
        pass
    
    @abstractmethod
    def contextualize_query(self, query: Union[str, Dict[str, Any], Observation]) -> Union[str, Dict[str, Any]]:
        """
        Contextualize a query based on conversation history.
        
        Args:
            query: The original query
            
        Returns:
            Contextualized query
        """
        pass
    
    def retrieve(self, query: Union[str, Dict[str, Any], Observation], 
                context: Optional[Dict[str, Any]] = None,
                limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve with contextual enhancement.
        
        Args:
            query: The retrieval query
            context: Additional context (optional)
            limit: Maximum number of results (optional)
            
        Returns:
            List of contextually relevant items
        """
        # Update internal context if provided
        if context:
            self.update_context(context)
        
        # Contextualize the query
        contextualized_query = self.contextualize_query(query)
        
        # Perform retrieval with contextualized query
        return self._contextual_retrieve(contextualized_query, limit)
    
    @abstractmethod
    def _contextual_retrieve(self, contextualized_query: Union[str, Dict[str, Any]], 
                           limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Perform retrieval with contextualized query.
        
        Args:
            contextualized_query: The contextualized query
            limit: Maximum number of results (optional)
            
        Returns:
            List of contextually relevant items
        """
        pass
    
    def get_supported_query_types(self) -> List[str]:
        """Get supported query types."""
        return ["text", "observation", "contextual"]
    
    def add_to_history(self, query: Union[str, Dict[str, Any], Observation], 
                      results: List[Dict[str, Any]]) -> None:
        """
        Add query-result pair to conversation history.
        
        Args:
            query: The query that was issued
            results: The results that were returned
        """
        pass
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        pass
