"""
Episodic memory implementation for trajectory and episode storage.

EpisodicMemory provides specialized storage for complete episodes,
trajectories, and temporal sequences of events.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from ..interfaces.memory_store import EpisodicMemoryStore
from ..primitives import Trajectory, TrajectoryStep, Observation, Action, State


class EpisodicMemory(EpisodicMemoryStore):
    """
    Episodic memory for storing and retrieving complete episodes.
    
    Specialized memory store that handles trajectories, episodes,
    and temporal sequences with similarity search and indexing.
    """
    
    def __init__(self, max_episodes: Optional[int] = None,
                 similarity_threshold: float = 0.5):
        """
        Initialize episodic memory.
        
        Args:
            max_episodes: Maximum number of episodes to store
            similarity_threshold: Threshold for episode similarity
        """
        self.max_episodes = max_episodes
        self.similarity_threshold = similarity_threshold
        
        self._episodes: Dict[str, Trajectory] = {}
        self._episode_metadata: Dict[str, Dict[str, Any]] = {}
        self._temporal_index: List[Tuple[datetime, str]] = []  # (timestamp, episode_id)
    
    def put(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store a value (generic memory store interface).
        
        Args:
            key: Unique identifier for the memory
            value: The value to store
            metadata: Additional metadata (optional)
            
        Returns:
            True if storage was successful
        """
        if isinstance(value, Trajectory):
            return self.store_episode(value, metadata) == key
        
        # For non-trajectory values, store as generic memory
        self._episodes[key] = value
        self._episode_metadata[key] = metadata or {}
        return True
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value by key.
        
        Args:
            key: The key to retrieve
            
        Returns:
            The stored value, or None if not found
        """
        return self._episodes.get(key)
    
    def query(self, query: Union[str, Dict[str, Any]], 
              limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Query episodes based on criteria.
        
        Args:
            query: Query string or dictionary of criteria
            limit: Maximum number of results (optional)
            
        Returns:
            List of matching episodes
        """
        results = []
        
        if isinstance(query, str):
            # Text search in metadata
            query_lower = query.lower()
            for episode_id, metadata in self._episode_metadata.items():
                if any(query_lower in str(v).lower() for v in metadata.values()):
                    results.append(self._format_episode_result(episode_id))
        elif isinstance(query, dict):
            # Criteria-based search
            matching_episodes = self.query_episodes(query, limit)
            results = [self._format_episode_result(ep.trajectory_id.hex) 
                      for ep in matching_episodes]
        
        if limit:
            results = results[:limit]
        
        return results
    
    def delete(self, key: str) -> bool:
        """
        Delete an episode by key.
        
        Args:
            key: The key to delete
            
        Returns:
            True if deletion was successful
        """
        if key in self._episodes:
            del self._episodes[key]
            del self._episode_metadata[key]
            self._update_temporal_index()
            return True
        return False
    
    def clear(self) -> None:
        """Clear all episodes from the store."""
        self._episodes.clear()
        self._episode_metadata.clear()
        self._temporal_index.clear()
    
    def size(self) -> int:
        """
        Get the number of stored episodes.
        
        Returns:
            Number of episodes in the store
        """
        return len(self._episodes)
    
    def get_capabilities(self) -> List[str]:
        """
        Get the memory store's declared capabilities.
        
        Returns:
            List of capability identifiers
        """
        return ["episodic_storage", "temporal_indexing", "similarity_search", "trajectory_support"]
    
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
        episode_id = episode.trajectory_id.hex
        
        # Check capacity limit
        if self.max_episodes and len(self._episodes) >= self.max_episodes:
            self._remove_oldest_episode()
        
        # Store episode
        self._episodes[episode_id] = episode
        
        # Store metadata
        episode_metadata = {
            "length": episode.length(),
            "start_time": episode.start_time,
            "end_time": episode.end_time,
            "total_reward": episode.total_reward(),
            "stored_at": datetime.now(),
            **(metadata or {})
        }
        self._episode_metadata[episode_id] = episode_metadata
        
        # Update temporal index
        self._temporal_index.append((episode.start_time, episode_id))
        self._temporal_index.sort(key=lambda x: x[0])
        
        return episode_id
    
    def retrieve_episode(self, episode_id: str) -> Optional[Trajectory]:
        """
        Retrieve an episode by ID.
        
        Args:
            episode_id: The episode identifier
            
        Returns:
            The stored episode, or None if not found
        """
        return self._episodes.get(episode_id)
    
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
        results = []
        
        for episode_id, episode in self._episodes.items():
            if self._matches_criteria(episode_id, episode, criteria):
                results.append(episode)
                
                if limit and len(results) >= limit:
                    break
        
        return results
    
    def get_similar_episodes(self, query_episode: Trajectory, 
                           similarity_threshold: Optional[float] = None,
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
        threshold = similarity_threshold or self.similarity_threshold
        similar_episodes = []
        
        for episode_id, episode in self._episodes.items():
            if episode.trajectory_id == query_episode.trajectory_id:
                continue  # Skip self
            
            similarity = self._compute_similarity(query_episode, episode)
            if similarity >= threshold:
                similar_episodes.append((episode, similarity))
        
        # Sort by similarity (highest first)
        similar_episodes.sort(key=lambda x: x[1], reverse=True)
        
        # Return just the episodes
        result_episodes = [ep[0] for ep in similar_episodes]
        
        if limit:
            result_episodes = result_episodes[:limit]
        
        return result_episodes
    
    def get_episodes_by_time_range(self, start_time: datetime, 
                                  end_time: datetime) -> List[Trajectory]:
        """
        Get episodes within a time range.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            List of episodes in the time range
        """
        results = []
        
        # Use temporal index for efficient range queries
        for timestamp, episode_id in self._temporal_index:
            if start_time <= timestamp <= end_time:
                if episode_id in self._episodes:
                    results.append(self._episodes[episode_id])
        
        return results
    
    def get_recent_episodes(self, count: int = 10) -> List[Trajectory]:
        """
        Get the most recent episodes.
        
        Args:
            count: Number of recent episodes to retrieve
            
        Returns:
            List of recent episodes
        """
        # Get most recent from temporal index
        recent_entries = sorted(self._temporal_index, key=lambda x: x[0], reverse=True)[:count]
        
        return [self._episodes[episode_id] for _, episode_id in recent_entries if episode_id in self._episodes]
    
    def get_episodes_by_length(self, min_length: Optional[int] = None,
                             max_length: Optional[int] = None) -> List[Trajectory]:
        """
        Get episodes by length criteria.
        
        Args:
            min_length: Minimum episode length (optional)
            max_length: Maximum episode length (optional)
            
        Returns:
            List of episodes matching length criteria
        """
        results = []
        
        for episode in self._episodes.values():
            length = episode.length()
            
            if min_length and length < min_length:
                continue
            if max_length and length > max_length:
                continue
            
            results.append(episode)
        
        return results
    
    def get_episodes_by_reward(self, min_reward: Optional[float] = None,
                              max_reward: Optional[float] = None) -> List[Trajectory]:
        """
        Get episodes by total reward criteria.
        
        Args:
            min_reward: Minimum total reward (optional)
            max_reward: Maximum total total reward (optional)
            
        Returns:
            List of episodes matching reward criteria
        """
        results = []
        
        for episode in self._episodes.values():
            reward = episode.total_reward()
            
            if min_reward is not None and reward < min_reward:
                continue
            if max_reward is not None and reward > max_reward:
                continue
            
            results.append(episode)
        
        return results
    
    def compute_episode_statistics(self) -> Dict[str, Any]:
        """
        Compute statistics over all stored episodes.
        
        Returns:
            Dictionary with episode statistics
        """
        if not self._episodes:
            return {}
        
        lengths = [ep.length() for ep in self._episodes.values()]
 rewards = [ep.total_reward() for ep in self._episodes.values()]
        
        return {
            "total_episodes": len(self._episodes),
            "average_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "average_reward": sum(rewards) / len(rewards),
            "min_reward": min(rewards),
            "max_reward": max(rewards),
            "oldest_episode": min(self._temporal_index, key=lambda x: x[0])[0] if self._temporal_index else None,
            "newest_episode": max(self._temporal_index, key=lambda x: x[0])[0] if self._temporal_index else None
        }
    
    def _matches_criteria(self, episode_id: str, episode: Trajectory, 
                        criteria: Dict[str, Any]) -> bool:
        """Check if an episode matches the query criteria."""
        for field, condition in criteria.items():
            if field == "length":
                if isinstance(condition, dict):
                    if "min" in condition and episode.length() < condition["min"]:
                        return False
                    if "max" in condition and episode.length() > condition["max"]:
                        return False
                elif episode.length() != condition:
                    return False
            
            elif field == "total_reward":
                reward = episode.total_reward()
                if isinstance(condition, dict):
                    if "min" in condition and reward < condition["min"]:
                        return False
                    if "max" in condition and reward > condition["max"]:
                        return False
                elif reward != condition:
                    return False
            
            elif field == "start_time":
                if isinstance(condition, dict):
                    if "after" in condition and episode.start_time < condition["after"]:
                        return False
                    if "before" in condition and episode.start_time > condition["before"]:
                        return False
                elif episode.start_time != condition:
                    return False
            
            elif field == "metadata":
                if not self._matches_metadata(episode_id, condition):
                    return False
        
        return True
    
    def _matches_metadata(self, episode_id: str, condition: Dict[str, Any]) -> bool:
        """Check if episode metadata matches condition."""
        metadata = self._episode_metadata.get(episode_id, {})
        
        for key, value in condition.items():
            if key not in metadata or metadata[key] != value:
                return False
        
        return True
    
    def _compute_similarity(self, episode1: Trajectory, episode2: Trajectory) -> float:
        """
        Compute similarity between two episodes.
        
        Args:
            episode1: First episode
            episode2: Second episode
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Simple similarity based on length and reward
        length_diff = abs(episode1.length() - episode2.length())
        reward_diff = abs(episode1.total_reward() - episode2.total_reward())
        
        # Normalize differences (simple approach)
        max_length_diff = max(episode1.length(), episode2.length())
        max_reward_diff = max(abs(episode1.total_reward()), abs(episode2.total_reward()))
        
        if max_length_diff == 0 and max_reward_diff == 0:
            return 1.0
        
        length_similarity = 1.0 - (length_diff / max_length_diff) if max_length_diff > 0 else 1.0
        reward_similarity = 1.0 - (reward_diff / max_reward_diff) if max_reward_diff > 0 else 1.0
        
        # Combine similarities
        return (length_similarity + reward_similarity) / 2.0
    
    def _format_episode_result(self, episode_id: str) -> Dict[str, Any]:
        """Format episode for query results."""
        episode = self._episodes[episode_id]
        metadata = self._episode_metadata[episode_id]
        
        return {
            "episode_id": episode_id,
            "episode": episode,
            "metadata": metadata,
            "length": episode.length(),
            "total_reward": episode.total_reward(),
            "start_time": episode.start_time,
            "end_time": episode.end_time
        }
    
    def _remove_oldest_episode(self) -> None:
        """Remove the oldest episode to make room."""
        if not self._temporal_index:
            return
        
        oldest_timestamp, oldest_episode_id = min(self._temporal_index, key=lambda x: x[0])
        
        if oldest_episode_id in self._episodes:
            del self._episodes[oldest_episode_id]
            del self._episode_metadata[oldest_episode_id]
        
        # Remove from temporal index
        self._temporal_index = [(ts, eid) for ts, eid in self._temporal_index if eid != oldest_episode_id]
    
    def _update_temporal_index(self) -> None:
        """Update the temporal index after deletions."""
        self._temporal_index = [
            (self._episode_metadata[eid]["start_time"], eid)
            for eid in self._episodes.keys()
            if eid in self._episode_metadata
        ]
        self._temporal_index.sort(key=lambda x: x[0])
