"""
Tests for parallel framework configuration system.
"""

import pytest
import tempfile
import os
import yaml
from pathlib import Path
from unittest.mock import patch

from parallel_framework.config import (
    ConfigManager,
    ParallelFrameworkConfig,
    RetryPolicy,
    ResourceLimits,
    MonitoringConfig,
    TaskTypeConfig,
    ConfigError,
    load_config_from_file,
    load_config_with_env_override,
    create_default_config_file
)
from parallel_framework.engine import ExecutionConfig


class TestRetryPolicy:
    """Test RetryPolicy dataclass"""
    
    def test_retry_policy_defaults(self):
        """Test default retry policy"""
        policy = RetryPolicy()
        
        assert policy.max_retries == 3
        assert policy.backoff_strategy == "exponential"
        assert policy.base_delay == 1.0
        assert policy.max_delay == 300.0
        assert policy.jitter is True
    
    def test_retry_policy_custom(self):
        """Test custom retry policy"""
        policy = RetryPolicy(
            max_retries=5,
            backoff_strategy="linear",
            base_delay=2.0,
            max_delay=600.0,
            jitter=False
        )
        
        assert policy.max_retries == 5
        assert policy.backoff_strategy == "linear"
        assert policy.base_delay == 2.0
        assert policy.max_delay == 600.0
        assert policy.jitter is False
    
    def test_retry_policy_validation(self):
        """Test retry policy validation"""
        # Invalid backoff strategy
        with pytest.raises(ValueError, match="backoff_strategy must be one of"):
            RetryPolicy(backoff_strategy="invalid")
        
        # Negative max_retries
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            RetryPolicy(max_retries=-1)
        
        # Negative base_delay
        with pytest.raises(ValueError, match="base_delay must be non-negative"):
            RetryPolicy(base_delay=-1.0)
        
        # max_delay < base_delay
        with pytest.raises(ValueError, match="max_delay must be >= base_delay"):
            RetryPolicy(base_delay=10.0, max_delay=5.0)


class TestResourceLimits:
    """Test ResourceLimits dataclass"""
    
    def test_resource_limits_defaults(self):
        """Test default resource limits"""
        limits = ResourceLimits()
        
        assert limits.memory_per_job is None
        assert limits.timeout_default == 3600
        assert limits.max_concurrent_jobs is None
        assert limits.cpu_limit is None
    
    def test_resource_limits_custom(self):
        """Test custom resource limits"""
        limits = ResourceLimits(
            memory_per_job="4GB",
            timeout_default=7200,
            max_concurrent_jobs=10,
            cpu_limit=0.8
        )
        
        assert limits.memory_per_job == "4GB"
        assert limits.timeout_default == 7200
        assert limits.max_concurrent_jobs == 10
        assert limits.cpu_limit == 0.8
    
    def test_resource_limits_validation(self):
        """Test resource limits validation"""
        # Invalid timeout
        with pytest.raises(ValueError, match="timeout_default must be positive"):
            ResourceLimits(timeout_default=0)
        
        # Invalid max_concurrent_jobs
        with pytest.raises(ValueError, match="max_concurrent_jobs must be positive"):
            ResourceLimits(max_concurrent_jobs=0)
        
        # Invalid cpu_limit
        with pytest.raises(ValueError, match="cpu_limit must be between 0 and 1"):
            ResourceLimits(cpu_limit=1.5)
        
        with pytest.raises(ValueError, match="cpu_limit must be between 0 and 1"):
            ResourceLimits(cpu_limit=0)


class TestMonitoringConfig:
    """Test MonitoringConfig dataclass"""
    
    def test_monitoring_config_defaults(self):
        """Test default monitoring config"""
        config = MonitoringConfig()
        
        assert config.enable_metrics is True
        assert config.log_level == "INFO"
        assert config.progress_update_interval == 5.0
        assert config.enable_dashboard is False
        assert config.dashboard_port == 8787
        assert config.metrics_export_format == "json"
    
    def test_monitoring_config_validation(self):
        """Test monitoring config validation"""
        # Invalid log level
        with pytest.raises(ValueError, match="log_level must be one of"):
            MonitoringConfig(log_level="INVALID")
        
        # Invalid progress update interval
        with pytest.raises(ValueError, match="progress_update_interval must be positive"):
            MonitoringConfig(progress_update_interval=0)
        
        # Invalid dashboard port
        with pytest.raises(ValueError, match="dashboard_port must be between 1 and 65535"):
            MonitoringConfig(dashboard_port=0)
        
        with pytest.raises(ValueError, match="dashboard_port must be between 1 and 65535"):
            MonitoringConfig(dashboard_port=70000)


class TestTaskTypeConfig:
    """Test TaskTypeConfig dataclass"""
    
    def test_task_type_config_defaults(self):
        """Test default task type config"""
        config = TaskTypeConfig()
        
        assert config.chunk_strategy == "time_based"
        assert config.chunk_size == "10min"
        assert config.resource_requirements == {}
        assert config.retry_policy is None
        assert config.timeout is None
        assert config.priority == 0
    
    def test_task_type_config_custom(self):
        """Test custom task type config"""
        retry_policy = RetryPolicy(max_retries=5)
        
        config = TaskTypeConfig(
            chunk_strategy="size_based",
            chunk_size="100MB",
            resource_requirements={"memory": "4GB"},
            retry_policy=retry_policy,
            timeout=3600,
            priority=1
        )
        
        assert config.chunk_strategy == "size_based"
        assert config.chunk_size == "100MB"
        assert config.resource_requirements == {"memory": "4GB"}
        assert config.retry_policy is retry_policy
        assert config.timeout == 3600
        assert config.priority == 1
    
    def test_task_type_config_validation(self):
        """Test task type config validation"""
        # Invalid chunk strategy
        with pytest.raises(ValueError, match="chunk_strategy must be one of"):
            TaskTypeConfig(chunk_strategy="invalid")


class TestParallelFrameworkConfig:
    """Test ParallelFrameworkConfig dataclass"""
    
    def test_framework_config_defaults(self):
        """Test default framework config"""
        config = ParallelFrameworkConfig()
        
        assert isinstance(config.execution, ExecutionConfig)
        assert isinstance(config.retry_policy, RetryPolicy)
        assert isinstance(config.resource_limits, ResourceLimits)
        assert isinstance(config.monitoring, MonitoringConfig)
        assert config.temp_storage == "/tmp/parallel_jobs"
        assert config.cleanup_temp_files is True
        assert config.task_types == {}
        assert config.allowed_task_types is None
        assert config.secure_mode is False
    
    def test_framework_config_validation(self):
        """Test framework config validation"""
        # Empty temp storage
        with pytest.raises(ValueError, match="temp_storage cannot be empty"):
            ParallelFrameworkConfig(temp_storage="")


class TestConfigManager:
    """Test ConfigManager class"""
    
    def test_manager_creation(self):
        """Test creating config manager"""
        manager = ConfigManager()
        
        assert manager.get_config() is None
        assert manager.get_config_sources() == []
    
    def test_load_default_config(self):
        """Test loading default config"""
        manager = ConfigManager()
        config = manager.load_default_config()
        
        assert isinstance(config, ParallelFrameworkConfig)
        assert manager.get_config() is config
        assert "default" in manager.get_config_sources()
    
    def test_load_from_dict(self):
        """Test loading config from dictionary"""
        manager = ConfigManager()
        
        config_dict = {
            "temp_storage": "/custom/temp",
            "cleanup_temp_files": False,
            "execution": {
                "scheduler": "processes",
                "n_workers": 4
            },
            "retry_policy": {
                "max_retries": 5,
                "backoff_strategy": "linear"
            },
            "task_types": {
                "test_task": {
                    "chunk_strategy": "size_based",
                    "chunk_size": "50MB",
                    "timeout": 1800
                }
            }
        }
        
        config = manager.load_from_dict(config_dict)
        
        assert config.temp_storage == "/custom/temp"
        assert config.cleanup_temp_files is False
        assert config.execution.scheduler == "processes"
        assert config.execution.n_workers == 4
        assert config.retry_policy.max_retries == 5
        assert config.retry_policy.backoff_strategy == "linear"
        assert "test_task" in config.task_types
        assert config.task_types["test_task"].chunk_strategy == "size_based"
        assert config.task_types["test_task"].chunk_size == "50MB"
        assert config.task_types["test_task"].timeout == 1800
    
    def test_load_from_file(self):
        """Test loading config from YAML file"""
        config_data = {
            "temp_storage": "/file/temp",
            "secure_mode": True,
            "monitoring": {
                "log_level": "DEBUG",
                "enable_dashboard": True
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_file = f.name
        
        try:
            manager = ConfigManager()
            config = manager.load_from_file(temp_file)
            
            assert config.temp_storage == "/file/temp"
            assert config.secure_mode is True
            assert config.monitoring.log_level == "DEBUG"
            assert config.monitoring.enable_dashboard is True
            assert f"file:{temp_file}" in manager.get_config_sources()
            
        finally:
            os.unlink(temp_file)
    
    def test_load_from_file_not_found(self):
        """Test loading config from non-existent file"""
        manager = ConfigManager()
        
        with pytest.raises(ConfigError, match="Configuration file not found"):
            manager.load_from_file("/nonexistent/config.yaml")
    
    def test_load_from_file_invalid_yaml(self):
        """Test loading config from invalid YAML file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_file = f.name
        
        try:
            manager = ConfigManager()
            
            with pytest.raises(ConfigError, match="Invalid YAML"):
                manager.load_from_file(temp_file)
                
        finally:
            os.unlink(temp_file)
    
    def test_load_from_env(self):
        """Test loading config from environment variables"""
        env_vars = {
            "PARALLEL_FRAMEWORK_SCHEDULER": "processes",
            "PARALLEL_FRAMEWORK_N_WORKERS": "8", 
            "PARALLEL_FRAMEWORK_SECURE_MODE": "true",
            "PARALLEL_FRAMEWORK_LOG_LEVEL": "DEBUG",
            "PARALLEL_FRAMEWORK_PROGRESS_UPDATE_INTERVAL": "2.5"
        }
        
        with patch.dict(os.environ, env_vars):
            manager = ConfigManager()
            env_config = manager.load_from_env()
            
            assert env_config["execution"]["scheduler"] == "processes"
            assert env_config["execution"]["n_workers"] == 8
            assert env_config["secure_mode"] is True
            assert env_config["monitoring"]["log_level"] == "DEBUG"
            assert env_config["monitoring"]["progress_update_interval"] == 2.5
    
    def test_merge_configs(self):
        """Test merging multiple configs"""
        manager = ConfigManager()
        
        config1 = ParallelFrameworkConfig(
            temp_storage="/temp1",
            secure_mode=False
        )
        config1.execution.scheduler = "threads"
        config1.execution.n_workers = 2
        
        config2 = ParallelFrameworkConfig(
            temp_storage="/temp2",
            cleanup_temp_files=False
        )
        config2.execution.n_workers = 4  # Should override
        config2.monitoring.log_level = "DEBUG"
        
        merged = manager.merge_configs(config1, config2)
        
        # config2 values should override config1
        assert merged.temp_storage == "/temp2"
        assert merged.cleanup_temp_files is False
        assert merged.secure_mode is False  # From config1
        
        # Nested values should merge properly
        assert merged.execution.scheduler == "threads"  # From config1
        assert merged.execution.n_workers == 4  # From config2 (override)
        assert merged.monitoring.log_level == "DEBUG"  # From config2
    
    def test_save_to_file(self):
        """Test saving config to file"""
        manager = ConfigManager()
        config = ParallelFrameworkConfig(
            temp_storage="/save/test",
            secure_mode=True
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_file = f.name
        
        try:
            manager.save_to_file(temp_file, config)
            
            # Verify file was created and contains correct data
            with open(temp_file, 'r') as f:
                saved_data = yaml.safe_load(f)
            
            assert saved_data["temp_storage"] == "/save/test"
            assert saved_data["secure_mode"] is True
            
        finally:
            os.unlink(temp_file)
    
    def test_validate_config(self):
        """Test config validation"""
        manager = ConfigManager()
        
        # Valid config
        valid_config = ParallelFrameworkConfig()
        errors = manager.validate_config(valid_config)
        assert errors == []
        
        # Config with invalid task type
        invalid_config = ParallelFrameworkConfig()
        invalid_config.task_types = {
            "": TaskTypeConfig(),  # Empty task type name
            "valid_task": TaskTypeConfig(
                chunk_strategy="time_based",
                chunk_size="invalid_time"  # Invalid time format
            )
        }
        invalid_config.allowed_task_types = [123, ""]  # Invalid types
        
        errors = manager.validate_config(invalid_config)
        assert len(errors) >= 3  # Should have multiple validation errors
        assert any("Task type name cannot be empty" in error for error in errors)
        assert any("Invalid time format" in error for error in errors)
        assert any("Invalid task type in allowed list" in error for error in errors)


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_load_config_from_file(self):
        """Test load_config_from_file function"""
        config_data = {"temp_storage": "/convenience/test"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_file = f.name
        
        try:
            config = load_config_from_file(temp_file)
            assert config.temp_storage == "/convenience/test"
        finally:
            os.unlink(temp_file)
    
    def test_load_config_with_env_override(self):
        """Test load_config_with_env_override function"""
        # Create config file  
        config_data = {
            "temp_storage": "/file/storage",
            "execution": {"scheduler": "threads", "n_workers": 2}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_file = f.name
        
        # Set environment variables that should override file values
        env_vars = {
            "PARALLEL_FRAMEWORK_SCHEDULER": "processes",
            "PARALLEL_FRAMEWORK_TEMP_STORAGE": "/env/storage"
        }
        
        try:
            with patch.dict(os.environ, env_vars):
                config = load_config_with_env_override(temp_file)
                
                # Environment overrides should take precedence where they exist
                assert config.temp_storage == "/env/storage"
                assert config.execution.scheduler == "processes"
                
        finally:
            os.unlink(temp_file)
    
    def test_load_config_with_env_override_no_file(self):
        """Test load_config_with_env_override without file"""
        env_vars = {
            "PARALLEL_FRAMEWORK_SCHEDULER": "processes",
            "PARALLEL_FRAMEWORK_N_WORKERS": "6",
            "PARALLEL_FRAMEWORK_TEMP_STORAGE": "/env/temp"
        }
        
        with patch.dict(os.environ, env_vars):
            config = load_config_with_env_override()
            
            # Should have env overrides where they exist
            assert config.execution.scheduler == "processes"
            assert config.execution.n_workers == 6
            assert config.temp_storage == "/env/temp"
    
    def test_create_default_config_file(self):
        """Test create_default_config_file function"""
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            temp_file = f.name
        
        try:
            create_default_config_file(temp_file)
            
            # Verify file was created
            assert os.path.exists(temp_file)
            
            # Load and verify content
            with open(temp_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            assert "temp_storage" in config_data
            assert "task_types" in config_data
            assert "audio_transcription" in config_data["task_types"]
            assert "video_processing" in config_data["task_types"]
            
        finally:
            os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__])