"""
Encoder interface - defines the contract for encoding and embedding.

Encoders are components that can convert various data types into
vector embeddings or other encoded representations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from ..primitives import Observation, Action, State


class Encoder(ABC):
    """
    Abstract base class for all encoders in BugForge.
    
    Encoders are responsible for converting data (text, observations,
    actions, etc.) into vector embeddings or other encoded forms.
    """
    
    @abstractmethod
    def encode(self, data: Union[str, Observation, Action, State, Dict[str, Any]], 
              context: Optional[Dict[str, Any]] = None) -> List[float]:
        """
        Encode data into a vector representation.
        
        Args:
            data: The data to encode
            context: Additional context for encoding (optional)
            
        Returns:
            Vector embedding of the data
        """
        pass
    
    @abstractmethod
    def batch_encode(self, data_list: List[Union[str, Observation, Action, State, Dict[str, Any]]], 
                    context: Optional[Dict[str, Any]] = None) -> List[List[float]]:
        """
        Encode multiple data items into vectors.
        
        Args:
            data_list: List of data items to encode
            context: Additional context for encoding (optional)
            
        Returns:
            List of vector embeddings
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Size of the embedding vectors
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Get the encoder's declared capabilities.
        
        Returns:
            List of capability identifiers
        """
        pass
    
    @abstractmethod
    def get_supported_data_types(self) -> List[str]:
        """
        Get the types of data this encoder can process.
        
        Returns:
            List of supported data types
        """
        pass
    
    def decode(self, embedding: List[float], 
              context: Optional[Dict[str, Any]] = None) -> Optional[Union[str, Dict[str, Any]]]:
        """
        Decode a vector embedding back to original data (if supported).
        
        Args:
            embedding: The embedding to decode
            context: Additional context for decoding (optional)
            
        Returns:
            Decoded data, or None if decoding not supported
        """
        return None
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (cosine similarity by default)
        """
        import math
        
        # Compute cosine similarity
        dot_product = sum(e1 * e2 for e1, e2 in zip(embedding1, embedding2))
        norm1 = math.sqrt(sum(e1 * e1 for e1 in embedding1))
        norm2 = math.sqrt(sum(e2 * e2 for e2 in embedding2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def compute_distance(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute distance between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Distance (Euclidean distance by default)
        """
        return math.sqrt(sum((e1 - e2) ** 2 for e1, e2 in zip(embedding1, embedding2)))
    
    def normalize_embedding(self, embedding: List[float]) -> List[float]:
        """
        Normalize an embedding vector.
        
        Args:
            embedding: The embedding to normalize
            
        Returns:
            Normalized embedding
        """
        import math
        
        norm = math.sqrt(sum(e * e for e in embedding))
        if norm == 0:
            return embedding
        
        return [e / norm for e in embedding]
    
    def validate_embedding(self, embedding: List[float]) -> bool:
        """
        Validate that an embedding has the correct dimension.
        
        Args:
            embedding: The embedding to validate
            
        Returns:
            True if embedding is valid
        """
        return len(embedding) == self.get_embedding_dimension()
    
    def save(self, path: str) -> None:
        """
        Save encoder state to disk.
        
        Args:
            path: Path where to save the encoder
        """
        raise NotImplementedError("Encoder saving not implemented")
    
    def load(self, path: str) -> None:
        """
        Load encoder state from disk.
        
        Args:
            path: Path from which to load the encoder
        """
        raise NotImplementedError("Encoder loading not implemented")
    
    def clone(self) -> "Encoder":
        """
        Create a copy of this encoder.
        
        Returns:
            A new instance of the same encoder
        """
        raise NotImplementedError("Encoder cloning not implemented")
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.get_embedding_dimension()})"


class TextEncoder(Encoder):
    """
    Encoder specialized for text data.
    
    Encodes text strings into vector embeddings using various
    text encoding techniques.
    """
    
    @abstractmethod
    def encode_text(self, text: str, 
                   context: Optional[Dict[str, Any]] = None) -> List[float]:
        """
        Encode a text string into a vector.
        
        Args:
            text: The text to encode
            context: Additional context (optional)
            
        Returns:
            Text embedding vector
        """
        pass
    
    def encode(self, data: Union[str, Observation, Action, State, Dict[str, Any]], 
              context: Optional[Dict[str, Any]] = None) -> List[float]:
        """
        Encode text data.
        
        Args:
            data: The text data to encode
            context: Additional context (optional)
            
        Returns:
            Text embedding vector
        """
        if isinstance(data, str):
            return self.encode_text(data, context)
        elif isinstance(data, dict) and "text" in data:
            return self.encode_text(data["text"], context)
        else:
            # Convert other types to string representation
            return self.encode_text(str(data), context)
    
    def batch_encode(self, data_list: List[Union[str, Observation, Action, State, Dict[str, Any]]], 
                    context: Optional[Dict[str, Any]] = None) -> List[List[float]]:
        """
        Encode multiple text items.
        
        Args:
            data_list: List of text data items
            context: Additional context (optional)
            
        Returns:
            List of text embedding vectors
        """
        return [self.encode(data, context) for data in data_list]
    
    def get_supported_data_types(self) -> List[str]:
        """Get supported data types."""
        return ["text", "string"]
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into tokens.
        
        Args:
            text: The text to tokenize
            
        Returns:
            List of tokens
        """
        # Default simple tokenization
        return text.split()
    
    def get_vocabulary_size(self) -> Optional[int]:
        """
        Get the vocabulary size.
        
        Returns:
            Size of the vocabulary, or None if not applicable
        """
        return None


class ObservationEncoder(Encoder):
    """
    Encoder specialized for observation data.
    
    Encodes observations into vector representations, handling
    various observation types and structures.
    """
    
    @abstractmethod
    def encode_observation(self, observation: Observation, 
                          context: Optional[Dict[str, Any]] = None) -> List[float]:
        """
        Encode an observation into a vector.
        
        Args:
            observation: The observation to encode
            context: Additional context (optional)
            
        Returns:
            Observation embedding vector
        """
        pass
    
    def encode(self, data: Union[str, Observation, Action, State, Dict[str, Any]], 
              context: Optional[Dict[str, Any]] = None) -> List[float]:
        """
        Encode observation data.
        
        Args:
            data: The observation data to encode
            context: Additional context (optional)
            
        Returns:
            Observation embedding vector
        """
        if isinstance(data, Observation):
            return self.encode_observation(data, context)
        else:
            raise ValueError(f"Unsupported data type for ObservationEncoder: {type(data)}")
    
    def batch_encode(self, data_list: List[Union[str, Observation, Action, State, Dict[str, Any]]], 
                    context: Optional[Dict[str, Any]] = None) -> List[List[float]]:
        """
        Encode multiple observations.
        
        Args:
            data_list: List of observations to encode
            context: Additional context (optional)
            
        Returns:
            List of observation embedding vectors
        """
        return [self.encode(data, context) for data in data_list]
    
    def get_supported_data_types(self) -> List[str]:
        """Get supported data types."""
        return ["observation"]


class MultimodalEncoder(Encoder):
    """
    Encoder that can handle multiple data types and modalities.
    
    Supports encoding of text, images, audio, and other modalities
    into a unified embedding space.
    """
    
    @abstractmethod
    def encode_multimodal(self, data_dict: Dict[str, Any], 
                         context: Optional[Dict[str, Any]] = None) -> List[float]:
        """
        Encode multimodal data into a unified vector.
        
        Args:
            data_dict: Dictionary with data from different modalities
            context: Additional context (optional)
            
        Returns:
            Unified multimodal embedding vector
        """
        pass
    
    @abstractmethod
    def get_supported_modalities(self) -> List[str]:
        """
        Get the supported modalities.
        
        Returns:
            List of supported modality names
        """
        pass
    
    def encode(self, data: Union[str, Observation, Action, State, Dict[str, Any]], 
              context: Optional[Dict[str, Any]] = None) -> List[float]:
        """
        Encode multimodal data.
        
        Args:
            data: The multimodal data to encode
            context: Additional context (optional)
            
        Returns:
            Multimodal embedding vector
        """
        if isinstance(data, dict):
            return self.encode_multimodal(data, context)
        else:
            # Single modality - wrap in dictionary
            modality = self._infer_modality(data)
            return self.encode_multimodal({modality: data}, context)
    
    def _infer_modality(self, data: Any) -> str:
        """
        Infer the modality of the data.
        
        Args:
            data: The data to infer modality for
            
        Returns:
            Inferred modality name
        """
        if isinstance(data, str):
            return "text"
        elif isinstance(data, Observation):
            return "observation"
        elif isinstance(data, Action):
            return "action"
        elif isinstance(data, State):
            return "state"
        else:
            return "unknown"
    
    def get_supported_data_types(self) -> List[str]:
        """Get supported data types."""
        return self.get_supported_modalities()


class AdaptiveEncoder(Encoder):
    """
    Encoder that can adapt and improve over time.
    
    Supports online learning and adaptation of encoding
    based on usage patterns and feedback.
    """
    
    @abstractmethod
    def update_encoding(self, data: Union[str, Observation, Action, State, Dict[str, Any]], 
                       feedback: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the encoding based on new data and feedback.
        
        Args:
            data: The data that was encoded
            feedback: Feedback about the encoding quality (optional)
        """
        pass
    
    @abstractmethod
    def get_adaptation_state(self) -> Dict[str, Any]:
        """
        Get the current adaptation state.
        
        Returns:
            Dictionary describing adaptation state
        """
        pass
    
    def encode_with_learning(self, data: Union[str, Observation, Action, State, Dict[str, Any]], 
                           context: Optional[Dict[str, Any]] = None,
                           learn: bool = True) -> List[float]:
        """
        Encode with optional learning.
        
        Args:
            data: The data to encode
            context: Additional context (optional)
            learn: Whether to update the encoder from this encoding
            
        Returns:
            Embedding vector
        """
        embedding = self.encode(data, context)
        
        if learn:
            self.update_encoding(data, context)
        
        return embedding
    
    def reset_adaptation(self) -> None:
        """Reset adaptation state to initial conditions."""
        pass
    
    def get_learning_rate(self) -> Optional[float]:
        """
        Get the current learning rate for adaptation.
        
        Returns:
            Current learning rate, or None if not applicable
        """
        return None
    
    def set_learning_rate(self, learning_rate: float) -> None:
        """
        Set the learning rate for adaptation.
        
        Args:
            learning_rate: New learning rate
        """
        pass
    
    def get_supported_data_types(self) -> List[str]:
        """Get supported data types."""
        return ["adaptive", "learnable"]
