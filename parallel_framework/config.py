"""
Configuration management system for the parallel processing framework.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
from enum import Enum

from .engine import ExecutionConfig


class ConfigError(Exception):
    """Raised when configuration is invalid or cannot be loaded"""
    pass


@dataclass
class RetryPolicy:
    """Configuration for job retry behavior"""
    max_retries: int = 3
    backoff_strategy: str = "exponential"  # linear, exponential, fixed
    base_delay: float = 1.0  # seconds
    max_delay: float = 300.0  # seconds
    jitter: bool = True  # Add randomness to delays
    
    def __post_init__(self):
        valid_strategies = ["linear", "exponential", "fixed"]
        if self.backoff_strategy not in valid_strategies:
            raise ValueError(f"backoff_strategy must be one of {valid_strategies}")
        
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        
        if self.base_delay < 0:
            raise ValueError("base_delay must be non-negative")
        
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")


@dataclass 
class ResourceLimits:
    """Configuration for resource limits"""
    memory_per_job: Optional[str] = None  # e.g., "4GB", "512MB"
    timeout_default: Optional[int] = 3600  # seconds
    max_concurrent_jobs: Optional[int] = None  # None for unlimited
    cpu_limit: Optional[float] = None  # e.g., 0.5 for 50% of CPU
    
    def __post_init__(self):
        if self.timeout_default is not None and self.timeout_default <= 0:
            raise ValueError("timeout_default must be positive")
        
        if self.max_concurrent_jobs is not None and self.max_concurrent_jobs <= 0:
            raise ValueError("max_concurrent_jobs must be positive")
        
        if self.cpu_limit is not None and (self.cpu_limit <= 0 or self.cpu_limit > 1.0):
            raise ValueError("cpu_limit must be between 0 and 1")


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and logging"""
    enable_metrics: bool = True
    log_level: str = "INFO"
    progress_update_interval: float = 5.0  # seconds
    enable_dashboard: bool = False
    dashboard_port: Optional[int] = 8787
    metrics_export_format: str = "json"  # json, prometheus
    
    def __post_init__(self):
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            raise ValueError(f"log_level must be one of {valid_log_levels}")
        
        if self.progress_update_interval <= 0:
            raise ValueError("progress_update_interval must be positive")
        
        if self.dashboard_port is not None and (self.dashboard_port < 1 or self.dashboard_port > 65535):
            raise ValueError("dashboard_port must be between 1 and 65535")


@dataclass
class TaskTypeConfig:
    """Configuration for a specific task type"""
    chunk_strategy: str = "time_based"  # time_based, size_based, content_based, custom
    chunk_size: str = "10min"  # For time_based: "10min", for size_based: "100MB"
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    retry_policy: Optional[RetryPolicy] = None
    timeout: Optional[int] = None
    priority: int = 0
    
    def __post_init__(self):
        valid_strategies = ["time_based", "size_based", "content_based", "custom"]
        if self.chunk_strategy not in valid_strategies:
            raise ValueError(f"chunk_strategy must be one of {valid_strategies}")


@dataclass
class ParallelFrameworkConfig:
    """Main configuration for the parallel processing framework"""
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Storage configuration
    temp_storage: str = "/tmp/parallel_jobs"
    cleanup_temp_files: bool = True
    
    # Task-specific configurations
    task_types: Dict[str, TaskTypeConfig] = field(default_factory=dict)
    
    # Security settings
    allowed_task_types: Optional[List[str]] = None  # None for all allowed
    secure_mode: bool = False
    
    def __post_init__(self):
        # Validate temp storage path
        if not self.temp_storage:
            raise ValueError("temp_storage cannot be empty")


class ConfigManager:
    """
    Manages loading, validation, and merging of configuration from multiple sources.
    
    Supports loading from:
    - YAML files
    - Environment variables
    - Python dictionaries
    - Default values
    """
    
    def __init__(self):
        self._config: Optional[ParallelFrameworkConfig] = None
        self._config_sources: List[str] = []
    
    def load_from_file(self, config_path: Union[str, Path]) -> ParallelFrameworkConfig:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Loaded configuration
            
        Raises:
            ConfigError: If file cannot be loaded or is invalid
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
            
            if data is None:
                data = {}
            
            config = self._create_config_from_dict(data)
            self._config = config
            self._config_sources.append(f"file:{config_path}")
            
            return config
            
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in config file {config_path}: {e}")
        except Exception as e:
            raise ConfigError(f"Failed to load config file {config_path}: {e}")
    
    def load_from_dict(self, config_dict: Dict[str, Any]) -> ParallelFrameworkConfig:
        """
        Load configuration from a dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Loaded configuration
        """
        try:
            config = self._create_config_from_dict(config_dict)
            self._config = config
            self._config_sources.append("dict")
            
            return config
            
        except Exception as e:
            raise ConfigError(f"Failed to load config from dictionary: {e}")
    
    def load_from_env(self, prefix: str = "PARALLEL_FRAMEWORK_") -> Dict[str, Any]:
        """
        Load configuration values from environment variables.
        
        Args:
            prefix: Prefix for environment variables
            
        Returns:
            Dictionary of configuration values from environment
        """
        env_config = {
            'execution': {},
            'retry_policy': {},
            'resource_limits': {},
            'monitoring': {}
        }
        
        # Mapping of env var suffixes to config sections
        section_mapping = {
            'scheduler': 'execution',
            'n_workers': 'execution', 
            'threads_per_worker': 'execution',
            'memory_limit': 'execution',
            'max_retries': 'retry_policy',
            'backoff_strategy': 'retry_policy',
            'base_delay': 'retry_policy',
            'memory_per_job': 'resource_limits',
            'timeout_default': 'resource_limits',
            'max_concurrent_jobs': 'resource_limits',
            'log_level': 'monitoring',
            'enable_metrics': 'monitoring',
            'progress_update_interval': 'monitoring'
            # Note: top-level vars like temp_storage, secure_mode are handled directly
        }
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert env var name to config key
                config_key = key[len(prefix):].lower()
                
                # Convert string values to appropriate types
                if value.lower() in ('true', 'false'):
                    converted_value = value.lower() == 'true'
                elif value.isdigit():
                    converted_value = int(value)
                elif self._is_float(value):
                    converted_value = float(value)
                else:
                    converted_value = value
                
                # Place in appropriate section or top level
                if config_key in section_mapping:
                    section = section_mapping[config_key]
                    env_config[section][config_key] = converted_value
                else:
                    env_config[config_key] = converted_value
        
        # Remove empty sections
        env_config = {k: v for k, v in env_config.items() if v}
        
        if env_config:
            self._config_sources.append(f"env:{prefix}")
        
        return env_config
    
    def merge_configs(self, *configs: ParallelFrameworkConfig) -> ParallelFrameworkConfig:
        """
        Merge multiple configurations, with later configs taking precedence.
        
        Args:
            *configs: Configuration objects to merge
            
        Returns:
            Merged configuration
        """
        if not configs:
            return ParallelFrameworkConfig()
        
        # Start with first config
        merged_dict = asdict(configs[0])
        
        # Merge subsequent configs
        for config in configs[1:]:
            config_dict = asdict(config)
            merged_dict = self._deep_merge_dicts(merged_dict, config_dict)
        
        # Create new config from merged dictionary
        merged_config = self._create_config_from_dict(merged_dict)
        self._config = merged_config
        self._config_sources.append(f"merged:{len(configs)}_configs")
        
        return merged_config
    
    def load_default_config(self) -> ParallelFrameworkConfig:
        """
        Load default configuration.
        
        Returns:
            Default configuration
        """
        config = ParallelFrameworkConfig()
        self._config = config
        self._config_sources.append("default")
        
        return config
    
    def get_config(self) -> Optional[ParallelFrameworkConfig]:
        """Get currently loaded configuration"""
        return self._config
    
    def get_config_sources(self) -> List[str]:
        """Get list of configuration sources that were loaded"""
        return self._config_sources.copy()
    
    def save_to_file(self, config_path: Union[str, Path], 
                     config: Optional[ParallelFrameworkConfig] = None) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            config_path: Path to save configuration file
            config: Configuration to save (uses current config if None)
            
        Raises:
            ConfigError: If configuration cannot be saved
        """
        if config is None:
            config = self._config
        
        if config is None:
            raise ConfigError("No configuration to save")
        
        config_path = Path(config_path)
        
        try:
            # Create parent directory if it doesn't exist
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert config to dictionary and save as YAML
            config_dict = asdict(config)
            
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
        except Exception as e:
            raise ConfigError(f"Failed to save config to {config_path}: {e}")
    
    def validate_config(self, config: ParallelFrameworkConfig) -> List[str]:
        """
        Validate configuration and return list of validation errors.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        try:
            # Validate temp storage directory
            temp_path = Path(config.temp_storage)
            if temp_path.exists() and not os.access(temp_path, os.W_OK):
                errors.append(f"Temp storage directory is not writable: {config.temp_storage}")
        except Exception as e:
            errors.append(f"Invalid temp storage path: {e}")
        
        # Validate task type configurations
        for task_type, task_config in config.task_types.items():
            if not task_type:
                errors.append("Task type name cannot be empty")
            
            # Validate chunk size format
            if task_config.chunk_strategy == "time_based":
                if not self._is_valid_time_format(task_config.chunk_size):
                    errors.append(f"Invalid time format for task '{task_type}': {task_config.chunk_size}")
            elif task_config.chunk_strategy == "size_based":
                if not self._is_valid_size_format(task_config.chunk_size):
                    errors.append(f"Invalid size format for task '{task_type}': {task_config.chunk_size}")
        
        # Validate allowed task types
        if config.allowed_task_types is not None:
            for task_type in config.allowed_task_types:
                if not isinstance(task_type, str) or not task_type:
                    errors.append(f"Invalid task type in allowed list: {task_type}")
        
        return errors
    
    def _create_config_from_dict(self, data: Dict[str, Any]) -> ParallelFrameworkConfig:
        """Create configuration object from dictionary"""
        # Extract nested configurations
        execution_data = data.get('execution', {})
        retry_data = data.get('retry_policy', {})
        resource_data = data.get('resource_limits', {})
        monitoring_data = data.get('monitoring', {})
        task_types_data = data.get('task_types', {})
        
        # Create nested config objects
        execution_config = ExecutionConfig(**execution_data)
        retry_policy = RetryPolicy(**retry_data)
        resource_limits = ResourceLimits(**resource_data)
        monitoring_config = MonitoringConfig(**monitoring_data)
        
        # Create task type configs
        task_types = {}
        for task_type, task_data in task_types_data.items():
            task_retry_data = task_data.get('retry_policy')
            task_retry = RetryPolicy(**task_retry_data) if task_retry_data else None
            
            task_config = TaskTypeConfig(
                chunk_strategy=task_data.get('chunk_strategy', 'time_based'),
                chunk_size=task_data.get('chunk_size', '10min'),
                resource_requirements=task_data.get('resource_requirements', {}),
                retry_policy=task_retry,
                timeout=task_data.get('timeout'),
                priority=task_data.get('priority', 0)
            )
            task_types[task_type] = task_config
        
        # Create main config
        main_config = ParallelFrameworkConfig(
            execution=execution_config,
            retry_policy=retry_policy,
            resource_limits=resource_limits,
            monitoring=monitoring_config,
            temp_storage=data.get('temp_storage', '/tmp/parallel_jobs'),
            cleanup_temp_files=data.get('cleanup_temp_files', True),
            task_types=task_types,
            allowed_task_types=data.get('allowed_task_types'),
            secure_mode=data.get('secure_mode', False)
        )
        
        return main_config
    
    def _deep_merge_dicts(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dicts(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _is_float(self, value: str) -> bool:
        """Check if string represents a float"""
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def _is_valid_time_format(self, time_str: str) -> bool:
        """Validate time format like '10min', '2h', '30s'"""
        import re
        pattern = r'^\d+(\.\d+)?(s|min|h)$'
        return bool(re.match(pattern, time_str))
    
    def _is_valid_size_format(self, size_str: str) -> bool:
        """Validate size format like '100MB', '2GB', '500KB'"""
        import re
        pattern = r'^\d+(\.\d+)?(B|KB|MB|GB|TB)$'
        return bool(re.match(pattern, size_str))


def load_config_from_file(config_path: Union[str, Path]) -> ParallelFrameworkConfig:
    """
    Convenience function to load configuration from a file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded configuration
    """
    manager = ConfigManager()
    return manager.load_from_file(config_path)


def load_config_with_env_override(config_path: Optional[Union[str, Path]] = None,
                                 env_prefix: str = "PARALLEL_FRAMEWORK_") -> ParallelFrameworkConfig:
    """
    Load configuration from file with environment variable overrides.
    
    Args:
        config_path: Path to configuration file (optional)
        env_prefix: Prefix for environment variables
        
    Returns:
        Configuration with environment overrides applied
    """
    manager = ConfigManager()
    
    # Load base config
    if config_path:
        base_config = manager.load_from_file(config_path)
    else:
        base_config = manager.load_default_config()
    
    # Load environment overrides
    env_config_dict = manager.load_from_env(env_prefix)
    
    if env_config_dict:
        # Create config from env values and merge
        try:
            env_config_obj = manager.load_from_dict(env_config_dict)
            return manager.merge_configs(base_config, env_config_obj)
        except Exception:
            # If env config creation fails, return base config
            return base_config
    
    return base_config


def create_default_config_file(config_path: Union[str, Path]) -> None:
    """
    Create a default configuration file with example values.
    
    Args:
        config_path: Path where to create the configuration file
    """
    manager = ConfigManager()
    config = ParallelFrameworkConfig()
    
    # Add some example task configurations
    config.task_types = {
        "audio_transcription": TaskTypeConfig(
            chunk_strategy="time_based",
            chunk_size="10min",
            resource_requirements={
                "memory": "2GB",
                "cpu_cores": 1,
                "gpu_memory": "1GB"
            },
            timeout=3600,
            priority=1
        ),
        "video_processing": TaskTypeConfig(
            chunk_strategy="size_based", 
            chunk_size="100MB",
            resource_requirements={
                "memory": "4GB",
                "cpu_cores": 2
            },
            timeout=7200,
            priority=0
        )
    }
    
    manager.save_to_file(config_path, config)