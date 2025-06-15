"""
Parallel Processing Framework

A generic, reusable parallel processing system that can handle any type of job or process.
"""

from .job import Job, JobHandle, JobState, JobStatus
from .registry import TaskRegistry
from .engine import ParallelExecutionEngine, ExecutionConfig
from .state_manager import StateValidator, JobStateManager, JobStateMonitor, StateTransitionError
from .config import (
    ConfigManager, ParallelFrameworkConfig, RetryPolicy, ResourceLimits,
    MonitoringConfig, TaskTypeConfig, ConfigError, load_config_from_file,
    load_config_with_env_override, create_default_config_file
)
from .resource_pool import (
    ResourceType, ResourceSpec, ResourceAllocation, ResourcePool,
    SystemResourceMonitor, ResourceError, InsufficientResourcesError,
    ResourceAllocationError
)
from .scheduler import (
    SchedulingAlgorithm, SchedulingPolicy, JobDependency, SchedulingContext,
    ScheduledJob, JobScheduler, DependencyManager, LoadBalancer,
    SchedulingStrategy, FIFOStrategy, PriorityStrategy, ResourceAwareStrategy,
    ShortestJobFirstStrategy, FairShareStrategy
)
from .monitoring import (
    MetricType, AlertSeverity, HealthStatus, MetricPoint, MetricSeries,
    Alert, ThresholdRule, ResourceMetrics, AlertManager, SystemHealthMonitor,
    MetricsCollector
)
from .scaling import (
    ScalingDirection, ScalingTrigger, WorkerState, ScalingEvent, ScalingMetrics,
    WorkerInstance, ScalingPolicy, UtilizationBasedPolicy, QueueBasedPolicy,
    PredictivePolicy, ResourceScaler, WorkerPoolManager, AutoScalingController
)

__version__ = "0.1.0"

__all__ = [
    "Job",
    "JobHandle", 
    "JobState",
    "JobStatus",
    "TaskRegistry",
    "ParallelExecutionEngine",
    "ExecutionConfig",
    "StateValidator",
    "JobStateManager", 
    "JobStateMonitor",
    "StateTransitionError",
    "ConfigManager",
    "ParallelFrameworkConfig",
    "RetryPolicy",
    "ResourceLimits", 
    "MonitoringConfig",
    "TaskTypeConfig",
    "ConfigError",
    "load_config_from_file",
    "load_config_with_env_override", 
    "create_default_config_file",
    "ResourceType",
    "ResourceSpec",
    "ResourceAllocation", 
    "ResourcePool",
    "SystemResourceMonitor",
    "ResourceError",
    "InsufficientResourcesError",
    "ResourceAllocationError",
    "SchedulingAlgorithm",
    "SchedulingPolicy",
    "JobDependency",
    "SchedulingContext",
    "ScheduledJob",
    "JobScheduler",
    "DependencyManager",
    "LoadBalancer",
    "SchedulingStrategy",
    "FIFOStrategy",
    "PriorityStrategy",
    "ResourceAwareStrategy",
    "ShortestJobFirstStrategy",
    "FairShareStrategy",
    "MetricType",
    "AlertSeverity",
    "HealthStatus",
    "MetricPoint",
    "MetricSeries",
    "Alert",
    "ThresholdRule",
    "ResourceMetrics",
    "AlertManager",
    "SystemHealthMonitor",
    "MetricsCollector",
    "ScalingDirection",
    "ScalingTrigger",
    "WorkerState",
    "ScalingEvent",
    "ScalingMetrics",
    "WorkerInstance",
    "ScalingPolicy",
    "UtilizationBasedPolicy",
    "QueueBasedPolicy",
    "PredictivePolicy",
    "ResourceScaler",
    "WorkerPoolManager",
    "AutoScalingController",
]