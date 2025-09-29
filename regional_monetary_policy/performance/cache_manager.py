"""
Intelligent caching manager for API responses and intermediate results.
"""

import os
import json
import pickle
import hashlib
import sqlite3
import time
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Container for cache entry metadata."""
    key: str
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    ttl_seconds: Optional[int]
    tags: List[str]
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds

class IntelligentCacheManager:
    """Manages intelligent caching of API responses and computation results."""
    
    def __init__(self, 
                 cache_dir: str = "data/cache",
                 max_cache_size_gb: float = 5.0,
                 default_ttl_hours: int = 24,
                 enable_compression: bool = True):
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_cache_size_bytes = max_cache_size_gb * (1024**3)
        self.default_ttl_seconds = default_ttl_hours * 3600
        self.enable_compression = enable_compression
        
        # Initialize SQLite database for metadata
        self.db_path = self.cache_dir / "cache_metadata.db"
        self._init_database()
        
        logger.info(f"Cache manager initialized: {cache_dir}, {max_cache_size_gb}GB limit")
    
    def _init_database(self):
        """Initialize SQLite database for cache metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    created_at TIMESTAMP,
                    last_accessed TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    size_bytes INTEGER,
                    ttl_seconds INTEGER,
                    tags TEXT,
                    file_path TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON cache_entries(created_at)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries(last_accessed)
            """)
    
    def _generate_cache_key(self, 
                          prefix: str, 
                          params: Dict[str, Any],
                          include_timestamp: bool = False) -> str:
        """Generate cache key from parameters."""
        
        # Sort parameters for consistent hashing
        sorted_params = json.dumps(params, sort_keys=True, default=str)
        
        if include_timestamp:
            # Include date for time-sensitive data
            date_str = datetime.now().strftime("%Y-%m-%d")
            sorted_params += f"_date_{date_str}"
        
        # Create hash
        hash_obj = hashlib.md5(sorted_params.encode())
        return f"{prefix}_{hash_obj.hexdigest()}"
    
    def get(self, 
            key: str, 
            default: Any = None,
            update_access: bool = True) -> Any:
        """Retrieve item from cache."""
        
        try:
            # Check if entry exists and is not expired
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT file_path, ttl_seconds, created_at FROM cache_entries WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                
                if not row:
                    return default
                
                file_path, ttl_seconds, created_at_str = row
                created_at = datetime.fromisoformat(created_at_str)
                
                # Check expiration
                if ttl_seconds and (datetime.now() - created_at).total_seconds() > ttl_seconds:
                    self.delete(key)
                    return default
                
                # Update access statistics
                if update_access:
                    conn.execute("""
                        UPDATE cache_entries 
                        SET last_accessed = ?, access_count = access_count + 1 
                        WHERE key = ?
                    """, (datetime.now().isoformat(), key))
            
            # Load data from file
            full_path = self.cache_dir / file_path
            if not full_path.exists():
                self.delete(key)
                return default
            
            return self._load_from_file(full_path)
            
        except Exception as e:
            logger.warning(f"Failed to retrieve cache entry {key}: {str(e)}")
            return default
    
    def set(self, 
            key: str, 
            value: Any,
            ttl_seconds: Optional[int] = None,
            tags: Optional[List[str]] = None) -> bool:
        """Store item in cache."""
        
        try:
            # Use default TTL if not specified
            if ttl_seconds is None:
                ttl_seconds = self.default_ttl_seconds
            
            tags = tags or []
            
            # Generate file path
            file_name = f"{key}.pkl"
            file_path = self.cache_dir / file_name
            
            # Save data to file
            size_bytes = self._save_to_file(file_path, value)
            
            # Store metadata in database
            with sqlite3.connect(self.db_path) as conn:
                now = datetime.now().isoformat()
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries 
                    (key, created_at, last_accessed, access_count, size_bytes, ttl_seconds, tags, file_path)
                    VALUES (?, ?, ?, 0, ?, ?, ?, ?)
                """, (key, now, now, size_bytes, ttl_seconds, json.dumps(tags), file_name))
            
            # Check cache size and cleanup if necessary
            self._cleanup_if_needed()
            
            logger.debug(f"Cached {key}: {size_bytes} bytes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache {key}: {str(e)}")
            return False
    
    def _save_to_file(self, file_path: Path, value: Any) -> int:
        """Save value to file and return size in bytes."""
        
        if self.enable_compression:
            import gzip
            with gzip.open(file_path.with_suffix('.pkl.gz'), 'wb') as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            return file_path.with_suffix('.pkl.gz').stat().st_size
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            return file_path.stat().st_size
    
    def _load_from_file(self, file_path: Path) -> Any:
        """Load value from file."""
        
        # Try compressed file first
        compressed_path = file_path.with_suffix('.pkl.gz')
        if compressed_path.exists():
            import gzip
            with gzip.open(compressed_path, 'rb') as f:
                return pickle.load(f)
        
        # Fall back to uncompressed
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def delete(self, key: str) -> bool:
        """Delete item from cache."""
        
        try:
            # Get file path from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT file_path FROM cache_entries WHERE key = ?", (key,))
                row = cursor.fetchone()
                
                if row:
                    file_path = self.cache_dir / row[0]
                    
                    # Remove file
                    if file_path.exists():
                        file_path.unlink()
                    
                    # Try compressed version
                    compressed_path = file_path.with_suffix('.pkl.gz')
                    if compressed_path.exists():
                        compressed_path.unlink()
                    
                    # Remove from database
                    conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    
                    logger.debug(f"Deleted cache entry: {key}")
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Failed to delete cache entry {key}: {str(e)}")
            return False
    
    def clear_expired(self) -> int:
        """Clear all expired cache entries."""
        
        expired_count = 0
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Find expired entries
                cursor = conn.execute("""
                    SELECT key, created_at, ttl_seconds 
                    FROM cache_entries 
                    WHERE ttl_seconds IS NOT NULL
                """)
                
                expired_keys = []
                now = datetime.now()
                
                for key, created_at_str, ttl_seconds in cursor.fetchall():
                    created_at = datetime.fromisoformat(created_at_str)
                    if (now - created_at).total_seconds() > ttl_seconds:
                        expired_keys.append(key)
                
                # Delete expired entries
                for key in expired_keys:
                    if self.delete(key):
                        expired_count += 1
            
            logger.info(f"Cleared {expired_count} expired cache entries")
            
        except Exception as e:
            logger.error(f"Failed to clear expired entries: {str(e)}")
        
        return expired_count
    
    def _cleanup_if_needed(self):
        """Cleanup cache if size limit is exceeded."""
        
        current_size = self.get_cache_size()
        
        if current_size > self.max_cache_size_bytes:
            logger.info(f"Cache size ({current_size / (1024**3):.2f}GB) exceeds limit, cleaning up")
            
            # First, clear expired entries
            self.clear_expired()
            
            # If still over limit, use LRU eviction
            current_size = self.get_cache_size()
            if current_size > self.max_cache_size_bytes:
                self._lru_eviction()
    
    def _lru_eviction(self):
        """Evict least recently used entries."""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get entries sorted by last access time (oldest first)
                cursor = conn.execute("""
                    SELECT key, size_bytes 
                    FROM cache_entries 
                    ORDER BY last_accessed ASC
                """)
                
                bytes_to_free = self.get_cache_size() - (self.max_cache_size_bytes * 0.8)  # Free to 80%
                bytes_freed = 0
                
                for key, size_bytes in cursor.fetchall():
                    if bytes_freed >= bytes_to_free:
                        break
                    
                    if self.delete(key):
                        bytes_freed += size_bytes
                
                logger.info(f"LRU eviction freed {bytes_freed / (1024**2):.1f}MB")
                
        except Exception as e:
            logger.error(f"Failed LRU eviction: {str(e)}")
    
    def get_cache_size(self) -> int:
        """Get total cache size in bytes."""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT SUM(size_bytes) FROM cache_entries")
                result = cursor.fetchone()[0]
                return result or 0
        except Exception:
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Basic stats
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as entry_count,
                        SUM(size_bytes) as total_size,
                        AVG(access_count) as avg_access_count,
                        MAX(access_count) as max_access_count
                    FROM cache_entries
                """)
                
                stats = dict(zip(['entry_count', 'total_size', 'avg_access_count', 'max_access_count'], 
                               cursor.fetchone()))
                
                # Hit rate (approximate)
                cursor = conn.execute("SELECT SUM(access_count) FROM cache_entries")
                total_accesses = cursor.fetchone()[0] or 0
                
                stats.update({
                    'total_size_gb': (stats['total_size'] or 0) / (1024**3),
                    'cache_utilization': (stats['total_size'] or 0) / self.max_cache_size_bytes,
                    'total_accesses': total_accesses
                })
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get cache stats: {str(e)}")
            return {}
    
    def cache_api_response(self, 
                          api_name: str,
                          endpoint: str,
                          params: Dict[str, Any],
                          response_data: Any,
                          ttl_hours: int = 24) -> str:
        """Cache API response with intelligent key generation."""
        
        cache_key = self._generate_cache_key(
            f"api_{api_name}_{endpoint}",
            params,
            include_timestamp=True
        )
        
        tags = [f"api:{api_name}", f"endpoint:{endpoint}"]
        
        self.set(cache_key, response_data, ttl_seconds=ttl_hours * 3600, tags=tags)
        
        return cache_key
    
    def get_cached_api_response(self, 
                              api_name: str,
                              endpoint: str,
                              params: Dict[str, Any]) -> Any:
        """Retrieve cached API response."""
        
        cache_key = self._generate_cache_key(
            f"api_{api_name}_{endpoint}",
            params,
            include_timestamp=True
        )
        
        return self.get(cache_key)
    
    def cache_computation_result(self, 
                               computation_name: str,
                               input_params: Dict[str, Any],
                               result: Any,
                               ttl_hours: int = 168) -> str:  # 1 week default
        """Cache computation result."""
        
        cache_key = self._generate_cache_key(
            f"computation_{computation_name}",
            input_params
        )
        
        tags = [f"computation:{computation_name}"]
        
        self.set(cache_key, result, ttl_seconds=ttl_hours * 3600, tags=tags)
        
        return cache_key
    
    def get_cached_computation(self, 
                             computation_name: str,
                             input_params: Dict[str, Any]) -> Any:
        """Retrieve cached computation result."""
        
        cache_key = self._generate_cache_key(
            f"computation_{computation_name}",
            input_params
        )
        
        return self.get(cache_key)
    
    def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate cache entries by tags."""
        
        invalidated_count = 0
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT key, tags FROM cache_entries")
                
                keys_to_delete = []
                
                for key, tags_json in cursor.fetchall():
                    try:
                        entry_tags = json.loads(tags_json) if tags_json else []
                        if any(tag in entry_tags for tag in tags):
                            keys_to_delete.append(key)
                    except json.JSONDecodeError:
                        continue
                
                # Delete matching entries
                for key in keys_to_delete:
                    if self.delete(key):
                        invalidated_count += 1
            
            logger.info(f"Invalidated {invalidated_count} cache entries with tags: {tags}")
            
        except Exception as e:
            logger.error(f"Failed to invalidate by tags: {str(e)}")
        
        return invalidated_count
    
    def clear_all(self) -> bool:
        """Clear all cache entries."""
        
        try:
            # Remove all files
            for file_path in self.cache_dir.glob("*.pkl*"):
                file_path.unlink()
            
            # Clear database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache_entries")
            
            logger.info("Cleared all cache entries")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
            return False