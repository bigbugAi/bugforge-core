"""
ModelProvider interface - defines the contract for model loading and management.

ModelProviders are components that can download, load, and manage models
from various sources including local storage and model repositories.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from ..primitives import Metadata


class ModelProvider(ABC):
    """
    Abstract base class for all model providers in BugForge.
    
    Model providers are responsible for downloading, loading, and
    managing models from various sources with proper caching and
    version management.
    """
    
    @abstractmethod
    def list_available_models(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        List available models from this provider.
        
        Args:
            filters: Optional filters to apply (tags, capabilities, etc.)
            
        Returns:
            List of model information dictionaries
        """
        pass
    
    @abstractmethod
    def download_model(self, model_id: str, 
                      local_path: Optional[str] = None,
                      force_download: bool = False) -> "LocalModelArtifact":
        """
        Download a model from the provider.
        
        Args:
            model_id: Identifier of the model to download
            local_path: Local path to store the model (optional)
            force_download: Force re-download even if already cached
            
        Returns:
            LocalModelArtifact representing the downloaded model
        """
        pass
    
    @abstractmethod
    def load_model(self, model_id: str, 
                   device: Optional[str] = None,
                   **kwargs) -> "Model":
        """
        Load a model from the provider.
        
        Args:
            model_id: Identifier of the model to load
            device: Device to load the model on (optional)
            **kwargs: Additional loading parameters
            
        Returns:
            Loaded model instance
        """
        pass
    
    @abstractmethod
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_id: Identifier of the model
            
        Returns:
            Dictionary with model information
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Get the provider's declared capabilities.
        
        Returns:
            List of capability identifiers
        """
        pass
    
    def is_model_available(self, model_id: str) -> bool:
        """
        Check if a model is available from this provider.
        
        Args:
            model_id: Identifier of the model to check
            
        Returns:
            True if the model is available
        """
        try:
            self.get_model_info(model_id)
            return True
        except Exception:
            return False
    
    def get_cached_models(self) -> List[str]:
        """
        Get list of models cached locally.
        
        Returns:
            List of cached model identifiers
        """
        return []
    
    def clear_cache(self, model_id: Optional[str] = None) -> None:
        """
        Clear cached models.
        
        Args:
            model_id: Specific model to clear, or None for all
        """
        pass
    
    def validate_model_id(self, model_id: str) -> bool:
        """
        Validate a model identifier format.
        
        Args:
            model_id: The model identifier to validate
            
        Returns:
            True if the format is valid
        """
        return bool(model_id and isinstance(model_id, str))
    
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about this provider.
        
        Returns:
            Dictionary with provider information
        """
        return {
            "provider_name": self.__class__.__name__,
            "capabilities": self.get_capabilities(),
            "supported_formats": self.get_supported_formats()
        }
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Get the list of supported model formats.
        
        Returns:
            List of supported format strings
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(capabilities={self.get_capabilities()})"


class LocalModelArtifact:
    """
    Represents a locally stored model artifact.
    
    Encapsulates information about a downloaded model including
    its location, metadata, and loading capabilities.
    """
    
    def __init__(self, model_id: str, local_path: str, 
                 metadata: Optional[Dict[str, Any]] = None,
                 provider: Optional[ModelProvider] = None):
        self.model_id = model_id
        self.local_path = local_path
        self.metadata = metadata or {}
        self.provider = provider
        self._loaded_model = None
    
    def get_model_id(self) -> str:
        """Get the model identifier."""
        return self.model_id
    
    def get_local_path(self) -> str:
        """Get the local storage path."""
        return self.local_path
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        return self.metadata.copy()
    
    def get_provider(self) -> Optional[ModelProvider]:
        """Get the provider that created this artifact."""
        return self.provider
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded in memory."""
        return self._loaded_model is not None
    
    def load(self, device: Optional[str] = None, **kwargs) -> "Model":
        """
        Load the model from the local artifact.
        
        Args:
            device: Device to load on (optional)
            **kwargs: Additional loading parameters
            
        Returns:
            Loaded model instance
        """
        if self._loaded_model is not None:
            return self._loaded_model
        
        if self.provider is None:
            raise ValueError("Cannot load artifact without provider")
        
        self._loaded_model = self.provider.load_model(
            self.model_id, device=device, **kwargs
        )
        return self._loaded_model
    
    def unload(self) -> None:
        """Unload the model from memory."""
        self._loaded_model = None
    
    def get_size(self) -> Optional[int]:
        """
        Get the size of the artifact in bytes.
        
        Returns:
            Size in bytes, or None if not available
        """
        import os
        try:
            return os.path.getsize(self.local_path)
        except OSError:
            return None
    
    def get_format(self) -> Optional[str]:
        """
        Get the format of the artifact.
        
        Returns:
            Format string, or None if not available
        """
        return self.metadata.get("format")
    
    def get_version(self) -> Optional[str]:
        """
        Get the version of the model.
        
        Returns:
            Version string, or None if not available
        """
        return self.metadata.get("version")
    
    def verify_integrity(self) -> bool:
        """
        Verify the integrity of the local artifact.
        
        Returns:
            True if the artifact is intact
        """
        import os
        return os.path.exists(self.local_path) and os.path.isfile(self.local_path)
    
    def delete(self) -> bool:
        """
        Delete the local artifact.
        
        Returns:
            True if deletion was successful
        """
        import os
        try:
            if self._loaded_model is not None:
                self.unload()
            os.remove(self.local_path)
            return True
        except OSError:
            return False
    
    def __str__(self) -> str:
        return f"LocalModelArtifact(model_id={self.model_id}, path={self.local_path})"


class HuggingFaceModelProvider(ModelProvider):
    """
    Model provider for Hugging Face models.
    
    Provides access to models from the Hugging Face Hub with
    downloading, caching, and loading capabilities.
    """
    
    @abstractmethod
    def search_models(self, query: str, 
                      limit: Optional[int] = None,
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for models on Hugging Face Hub.
        
        Args:
            query: Search query
            limit: Maximum number of results (optional)
            filters: Additional search filters (optional)
            
        Returns:
            List of model information
        """
        pass
    
    @abstractmethod
    def get_model_files(self, model_id: str) -> List[str]:
        """
        Get list of files for a model.
        
        Args:
            model_id: Identifier of the model
            
        Returns:
            List of file names
        """
        pass
    
    def download_model(self, model_id: str, 
                      local_path: Optional[str] = None,
                      force_download: bool = False) -> "LocalModelArtifact":
        """
        Download a model from Hugging Face Hub.
        
        Args:
            model_id: Hugging Face model identifier
            local_path: Local path to store the model (optional)
            force_download: Force re-download even if already cached
            
        Returns:
            LocalModelArtifact representing the downloaded model
        """
        # Implementation would use huggingface_hub library
        raise NotImplementedError("HuggingFace download not implemented")
    
    def load_model(self, model_id: str, 
                   device: Optional[str] = None,
                   **kwargs) -> "Model":
        """
        Load a Hugging Face model.
        
        Args:
            model_id: Hugging Face model identifier
            device: Device to load the model on (optional)
            **kwargs: Additional loading parameters
            
        Returns:
            Loaded model instance
        """
        # Implementation would use transformers library
        raise NotImplementedError("HuggingFace loading not implemented")
    
    def get_capabilities(self) -> List[str]:
        """Get Hugging Face provider capabilities."""
        return ["huggingface_hub", "transformers", "model_download", "model_caching"]
    
    def get_supported_formats(self) -> List[str]:
        """Get supported Hugging Face formats."""
        return ["pytorch", "tensorflow", "safetensors", "gguf", "onnx"]


class LocalModelProvider(ModelProvider):
    """
    Model provider for locally stored models.
    
    Provides access to models stored on the local filesystem
    with loading and management capabilities.
    """
    
    def __init__(self, model_directory: str):
        self.model_directory = model_directory
    
    def list_available_models(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        List models in the local directory.
        
        Args:
            filters: Optional filters to apply
            
        Returns:
            List of model information
        """
        import os
        models = []
        
        if os.path.exists(self.model_directory):
            for item in os.listdir(self.model_directory):
                item_path = os.path.join(self.model_directory, item)
                if os.path.isdir(item_path):
                    models.append({
                        "model_id": item,
                        "path": item_path,
                        "source": "local"
                    })
        
        return models
    
    def download_model(self, model_id: str, 
                      local_path: Optional[str] = None,
                      force_download: bool = False) -> "LocalModelArtifact":
        """
        Create a local artifact for an existing local model.
        
        Args:
            model_id: Local model identifier
            local_path: Local path (should match model directory)
            force_download: Not applicable for local models
            
        Returns:
            LocalModelArtifact for the local model
        """
        if local_path is None:
            local_path = os.path.join(self.model_directory, model_id)
        
        return LocalModelArtifact(
            model_id=model_id,
            local_path=local_path,
            metadata={"source": "local"},
            provider=self
        )
    
    def load_model(self, model_id: str, 
                   device: Optional[str] = None,
                   **kwargs) -> "Model":
        """
        Load a local model.
        
        Args:
            model_id: Local model identifier
            device: Device to load on (optional)
            **kwargs: Additional loading parameters
            
        Returns:
            Loaded model instance
        """
        # Implementation would depend on local model format
        raise NotImplementedError("Local model loading not implemented")
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a local model."""
        import os
        model_path = os.path.join(self.model_directory, model_id)
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model {model_id} not found in {self.model_directory}")
        
        return {
            "model_id": model_id,
            "path": model_path,
            "source": "local",
            "exists": True
        }
    
    def get_capabilities(self) -> List[str]:
        """Get local provider capabilities."""
        return ["local_storage", "filesystem_access", "model_loading"]
    
    def get_supported_formats(self) -> List[str]:
        """Get supported local formats."""
        return ["pytorch", "pickle", "joblib", "custom"]
