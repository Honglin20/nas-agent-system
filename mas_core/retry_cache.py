"""
NAS CLI Retry and Cache System
重试机制和缓存系统
"""
import time
import hashlib
import pickle
from functools import wraps
from typing import Callable, Any, Optional, TypeVar, List, Dict
from pathlib import Path
import threading

from .exceptions import NASCLIError, ErrorCode, LLMError


T = TypeVar('T')


class RetryConfig:
    """重试配置"""
    def __init__(self,
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 retryable_errors: Optional[List[type]] = None):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.retryable_errors = retryable_errors or [Exception]


def retry_with_backoff(config: Optional[RetryConfig] = None):
    """
    带指数退避的重试装饰器
    
    使用示例:
        @retry_with_backoff(RetryConfig(max_retries=3, base_delay=1.0))
        def call_api():
            pass
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except tuple(config.retryable_errors) as e:
                    last_exception = e
                    
                    if attempt < config.max_retries:
                        # 计算延迟时间（指数退避）
                        delay = min(
                            config.base_delay * (config.exponential_base ** attempt),
                            config.max_delay
                        )
                        
                        # 添加 jitter 避免惊群效应
                        import random
                        delay = delay * (0.5 + random.random())
                        
                        print(f"[Retry] Attempt {attempt + 1}/{config.max_retries + 1} failed: {e}")
                        print(f"[Retry] Waiting {delay:.1f}s before next attempt...")
                        time.sleep(delay)
                    else:
                        break
            
            # 所有重试都失败
            raise last_exception
        
        return wrapper
    return decorator


class CacheEntry:
    """缓存条目"""
    def __init__(self, value: Any, ttl: int):
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        return time.time() - self.created_at > self.ttl


class Cache:
    """内存缓存"""
    
    def __init__(self, default_ttl: int = 3600):
        self._cache: Dict[str, CacheEntry] = {}
        self._default_ttl = default_ttl
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            
            if entry.is_expired():
                del self._cache[key]
                return None
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """设置缓存值"""
        with self._lock:
            self._cache[key] = CacheEntry(value, ttl or self._default_ttl)
    
    def delete(self, key: str):
        """删除缓存"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """获取缓存统计"""
        with self._lock:
            total = len(self._cache)
            expired = sum(1 for e in self._cache.values() if e.is_expired())
            return {
                'total_entries': total,
                'expired_entries': expired,
                'valid_entries': total - expired
            }


class FileCache:
    """文件缓存（持久化）"""
    
    def __init__(self, cache_dir: Path, default_ttl: int = 3600):
        self._cache_dir = cache_dir
        self._default_ttl = default_ttl
        self._cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, key: str) -> Path:
        """获取缓存文件路径"""
        # 使用哈希作为文件名
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self._cache_dir / f"{key_hash}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                entry = pickle.load(f)
            
            if entry.is_expired():
                cache_path.unlink(missing_ok=True)
                return None
            
            return entry.value
        except Exception:
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """设置缓存值"""
        cache_path = self._get_cache_path(key)
        entry = CacheEntry(value, ttl or self._default_ttl)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(entry, f)
        except Exception as e:
            print(f"[FileCache] Warning: Failed to write cache: {e}")
    
    def clear_expired(self):
        """清理过期缓存"""
        for cache_file in self._cache_dir.glob("*.cache"):
            try:
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                
                if entry.is_expired():
                    cache_file.unlink(missing_ok=True)
            except Exception:
                pass
    
    def clear_all(self):
        """清空所有缓存"""
        for cache_file in self._cache_dir.glob("*.cache"):
            cache_file.unlink(missing_ok=True)


def cached(cache: Cache, key_func: Optional[Callable] = None):
    """
    缓存装饰器
    
    使用示例:
        cache = Cache()
        
        @cached(cache, key_func=lambda x: f"analyze_{x}")
        def analyze_code(code: str):
            pass
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # 默认使用函数名和参数哈希
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5("|".join(key_parts).encode()).hexdigest()
            
            # 尝试从缓存获取
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                print(f"[Cache] Hit for {func.__name__}")
                return cached_value
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 存入缓存
            cache.set(cache_key, result)
            
            return result
        
        return wrapper
    return decorator


class CircuitBreaker:
    """熔断器模式"""
    
    STATE_CLOSED = 'closed'      # 正常状态
    STATE_OPEN = 'open'          # 熔断状态
    STATE_HALF_OPEN = 'half_open'  # 半开状态
    
    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 half_open_max_calls: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self._state = self.STATE_CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._lock = threading.RLock()
    
    def can_execute(self) -> bool:
        """检查是否可以执行"""
        with self._lock:
            if self._state == self.STATE_CLOSED:
                return True
            
            if self._state == self.STATE_OPEN:
                # 检查是否过了恢复时间
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._state = self.STATE_HALF_OPEN
                    self._success_count = 0
                    return True
                return False
            
            if self._state == self.STATE_HALF_OPEN:
                return self._success_count < self.half_open_max_calls
            
            return True
    
    def record_success(self):
        """记录成功"""
        with self._lock:
            if self._state == self.STATE_HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_max_calls:
                    self._state = self.STATE_CLOSED
                    self._failure_count = 0
            else:
                self._failure_count = 0
    
    def record_failure(self):
        """记录失败"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == self.STATE_HALF_OPEN:
                self._state = self.STATE_OPEN
            elif self._failure_count >= self.failure_threshold:
                self._state = self.STATE_OPEN
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """执行函数，带熔断保护"""
        if not self.can_execute():
            raise NASCLIError(
                ErrorCode.LLM_API_ERROR,
                "服务暂时不可用，请稍后重试（熔断器开启）"
            )
        
        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise e


# 全局缓存实例
_global_cache = Cache()


def get_global_cache() -> Cache:
    """获取全局缓存实例"""
    return _global_cache
