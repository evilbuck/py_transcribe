"""
Dynamic resource scaling system for the parallel processing framework.

This module provides intelligent resource scaling capabilities that automatically
adjust system capacity based on workload demand, performance metrics, and
resource utilization patterns.
"""

import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any, Tuple, Union
import logging
import statistics

from .resource_pool import ResourcePool, ResourceType, ResourceSpec
from .monitoring import ResourceMetrics, AlertManager, SystemHealthMonitor
from .scheduler import JobScheduler

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Direction of scaling operation"""
    UP = "up"
    DOWN = "down"
    MAINTAIN = "maintain"


class ScalingTrigger(Enum):
    """Triggers that can initiate scaling actions"""
    UTILIZATION = "utilization"
    QUEUE_DEPTH = "queue_depth"
    RESPONSE_TIME = "response_time"
    FAILURE_RATE = "failure_rate"
    SCHEDULED = "scheduled"
    MANUAL = "manual"


class WorkerState(Enum):
    """States of worker instances"""
    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class ScalingEvent:
    """Record of a scaling event"""
    id: str
    timestamp: datetime
    direction: ScalingDirection
    trigger: ScalingTrigger
    resource_type: ResourceType
    requested_change: float
    actual_change: float
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """Check if scaling event was successful"""
        return abs(self.actual_change - self.requested_change) < 0.1


@dataclass
class ScalingMetrics:
    """Metrics for scaling decision making"""
    timestamp: datetime
    cpu_utilization: float
    memory_utilization: float
    queue_depth: int
    average_response_time: float
    job_failure_rate: float
    active_workers: int
    pending_jobs: int
    system_load: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp,
            "cpu_utilization": self.cpu_utilization,
            "memory_utilization": self.memory_utilization,
            "queue_depth": self.queue_depth,
            "average_response_time": self.average_response_time,
            "job_failure_rate": self.job_failure_rate,
            "active_workers": self.active_workers,
            "pending_jobs": self.pending_jobs,
            "system_load": self.system_load
        }


@dataclass
class WorkerInstance:
    """Represents a worker instance"""
    id: str
    resource_type: ResourceType
    capacity: Dict[ResourceType, float]
    state: WorkerState
    created_at: datetime
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def uptime(self) -> Optional[timedelta]:
        """Get worker uptime"""
        if self.started_at:
            end_time = self.stopped_at or datetime.now(timezone.utc)
            return end_time - self.started_at
        return None
    
    @property
    def is_healthy(self) -> bool:
        """Check if worker is healthy"""
        if self.state not in [WorkerState.RUNNING]:
            return False
        
        if self.last_heartbeat:
            # Consider unhealthy if no heartbeat in last 5 minutes
            threshold = datetime.now(timezone.utc) - timedelta(minutes=5)
            return self.last_heartbeat > threshold
        
        return False  # No heartbeat means unhealthy


class ScalingPolicy(ABC):
    """Abstract base class for scaling policies"""
    
    @abstractmethod
    def should_scale(self, metrics: ScalingMetrics, history: List[ScalingMetrics]) -> Tuple[ScalingDirection, float, str]:
        """
        Determine if scaling is needed and by how much.
        
        Args:
            metrics: Current system metrics
            history: Historical metrics for trend analysis
            
        Returns:
            Tuple of (direction, amount, reason)
        """
        pass
    
    @abstractmethod
    def get_cooldown_period(self) -> timedelta:
        """Get cooldown period after scaling events"""
        pass


class UtilizationBasedPolicy(ScalingPolicy):
    """Scaling policy based on resource utilization"""
    
    def __init__(self, 
                 cpu_target: float = 70.0,
                 memory_target: float = 80.0,
                 scale_up_threshold: float = 15.0,
                 scale_down_threshold: float = 30.0,
                 min_workers: int = 1,
                 max_workers: int = 10,
                 cooldown_minutes: int = 5):
        self.cpu_target = cpu_target
        self.memory_target = memory_target
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.cooldown_period = timedelta(minutes=cooldown_minutes)
    
    def should_scale(self, metrics: ScalingMetrics, history: List[ScalingMetrics]) -> Tuple[ScalingDirection, float, str]:
        """Determine scaling based on CPU and memory utilization"""
        cpu_util = metrics.cpu_utilization
        memory_util = metrics.memory_utilization
        current_workers = metrics.active_workers
        
        # Calculate utilization deviation from targets
        cpu_deviation = abs(cpu_util - self.cpu_target)
        memory_deviation = abs(memory_util - self.memory_target)
        
        # Use the resource that deviates most from target
        if cpu_deviation > memory_deviation:
            primary_resource = "CPU"
            primary_util = cpu_util
            primary_target = self.cpu_target
            max_deviation = cpu_deviation
        else:
            primary_resource = "Memory"
            primary_util = memory_util
            primary_target = self.memory_target
            max_deviation = memory_deviation
        
        # Scale up if utilization is significantly above target
        if max_deviation > self.scale_up_threshold and primary_util > primary_target:
            if current_workers < self.max_workers:
                # Calculate scale amount based on deviation
                scale_factor = min(max_deviation / 50.0, 2.0)  # Cap at 2x
                workers_to_add = max(1, int(current_workers * scale_factor * 0.2))
                workers_to_add = min(workers_to_add, self.max_workers - current_workers)
                
                reason = f"High {primary_resource.lower()} utilization: {primary_util:.1f}% (target: {primary_target:.1f}%)"
                return ScalingDirection.UP, float(workers_to_add), reason
        
        # Scale down if utilization is significantly below target
        elif max_deviation > self.scale_down_threshold and primary_util < primary_target:
            if current_workers > self.min_workers:
                # Calculate scale down amount
                under_utilization = primary_target - primary_util
                scale_factor = min(under_utilization / 50.0, 0.5)  # Cap at 50% reduction
                workers_to_remove = max(1, int(current_workers * scale_factor * 0.3))
                workers_to_remove = min(workers_to_remove, current_workers - self.min_workers)
                
                reason = f"Low {primary_resource.lower()} utilization: {primary_util:.1f}% (target: {primary_target:.1f}%)"
                return ScalingDirection.DOWN, float(workers_to_remove), reason
        
        return ScalingDirection.MAINTAIN, 0.0, "Utilization within target range"
    
    def get_cooldown_period(self) -> timedelta:
        """Get cooldown period"""
        return self.cooldown_period


class QueueBasedPolicy(ScalingPolicy):
    """Scaling policy based on job queue depth"""
    
    def __init__(self,
                 target_queue_per_worker: int = 5,
                 scale_up_threshold: int = 10,
                 scale_down_threshold: int = 2,
                 min_workers: int = 1,
                 max_workers: int = 10,
                 cooldown_minutes: int = 3):
        self.target_queue_per_worker = target_queue_per_worker
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.cooldown_period = timedelta(minutes=cooldown_minutes)
    
    def should_scale(self, metrics: ScalingMetrics, history: List[ScalingMetrics]) -> Tuple[ScalingDirection, float, str]:
        """Determine scaling based on queue depth"""
        queue_depth = metrics.queue_depth
        current_workers = max(metrics.active_workers, 1)  # Avoid division by zero
        
        queue_per_worker = queue_depth / current_workers
        
        # Scale up if queue per worker exceeds threshold
        if queue_per_worker > self.scale_up_threshold and current_workers < self.max_workers:
            # Calculate workers needed to reach target
            ideal_workers = max(1, queue_depth / self.target_queue_per_worker)
            workers_to_add = min(
                int(ideal_workers - current_workers),
                self.max_workers - current_workers
            )
            workers_to_add = max(1, workers_to_add)
            
            reason = f"High queue depth: {queue_depth} jobs ({queue_per_worker:.1f} per worker)"
            return ScalingDirection.UP, float(workers_to_add), reason
        
        # Scale down if queue per worker is very low
        elif queue_per_worker < self.scale_down_threshold and current_workers > self.min_workers:
            # Calculate workers that can be removed
            if queue_depth == 0:
                workers_to_remove = max(1, (current_workers - self.min_workers) // 2)
            else:
                ideal_workers = max(self.min_workers, queue_depth / self.target_queue_per_worker)
                workers_to_remove = max(1, int(current_workers - ideal_workers))
            
            workers_to_remove = min(workers_to_remove, current_workers - self.min_workers)
            
            reason = f"Low queue depth: {queue_depth} jobs ({queue_per_worker:.1f} per worker)"
            return ScalingDirection.DOWN, float(workers_to_remove), reason
        
        return ScalingDirection.MAINTAIN, 0.0, "Queue depth within acceptable range"
    
    def get_cooldown_period(self) -> timedelta:
        """Get cooldown period"""
        return self.cooldown_period


class PredictivePolicy(ScalingPolicy):
    """Predictive scaling policy based on trends and patterns"""
    
    def __init__(self,
                 prediction_window: int = 10,
                 trend_threshold: float = 0.1,
                 min_workers: int = 1,
                 max_workers: int = 10,
                 cooldown_minutes: int = 8):
        self.prediction_window = prediction_window
        self.trend_threshold = trend_threshold
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.cooldown_period = timedelta(minutes=cooldown_minutes)
    
    def should_scale(self, metrics: ScalingMetrics, history: List[ScalingMetrics]) -> Tuple[ScalingDirection, float, str]:
        """Determine scaling based on trend prediction"""
        if len(history) < self.prediction_window:
            return ScalingDirection.MAINTAIN, 0.0, "Insufficient historical data for prediction"
        
        recent_history = history[-self.prediction_window:]
        current_workers = metrics.active_workers
        
        # Analyze trends in multiple metrics
        cpu_trend = self._calculate_trend([m.cpu_utilization for m in recent_history])
        memory_trend = self._calculate_trend([m.memory_utilization for m in recent_history])
        queue_trend = self._calculate_trend([m.queue_depth for m in recent_history])
        
        # Predict future load
        future_cpu = metrics.cpu_utilization + (cpu_trend * 3)  # 3 periods ahead
        future_memory = metrics.memory_utilization + (memory_trend * 3)
        future_queue = max(0, metrics.queue_depth + (queue_trend * 3))
        
        # Decision based on predicted values
        if (future_cpu > 85 or future_memory > 90 or future_queue > current_workers * 8) and current_workers < self.max_workers:
            workers_to_add = 1
            if future_cpu > 95 or future_memory > 95:
                workers_to_add = 2
            
            workers_to_add = min(workers_to_add, self.max_workers - current_workers)
            reason = f"Predicted overload: CPU={future_cpu:.1f}%, Memory={future_memory:.1f}%, Queue={future_queue:.0f}"
            return ScalingDirection.UP, float(workers_to_add), reason
        
        elif (future_cpu < 40 and future_memory < 50 and future_queue < current_workers * 2) and current_workers > self.min_workers:
            workers_to_remove = 1
            workers_to_remove = min(workers_to_remove, current_workers - self.min_workers)
            
            reason = f"Predicted underutilization: CPU={future_cpu:.1f}%, Memory={future_memory:.1f}%, Queue={future_queue:.0f}"
            return ScalingDirection.DOWN, float(workers_to_remove), reason
        
        return ScalingDirection.MAINTAIN, 0.0, "Predicted load within acceptable range"
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend slope using linear regression"""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_vals = list(range(n))
        
        # Simple linear regression
        x_mean = statistics.mean(x_vals)
        y_mean = statistics.mean(values)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, values))
        denominator = sum((x - x_mean) ** 2 for x in x_vals)
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def get_cooldown_period(self) -> timedelta:
        """Get cooldown period"""
        return self.cooldown_period


class ResourceScaler:
    """
    Manages dynamic resource scaling for the parallel processing framework.
    
    Coordinates with resource pools to dynamically adjust capacity based on
    workload demand and system performance metrics.
    """
    
    def __init__(self, resource_pool: ResourcePool, metrics: ResourceMetrics):
        self.resource_pool = resource_pool
        self.metrics = metrics
        
        self._lock = threading.RLock()
        self._scaling_events: deque = deque(maxlen=1000)
        self._last_scaling_time: Dict[ResourceType, datetime] = {}
        
        # Scaling configuration
        self._min_capacity: Dict[ResourceType, float] = {}
        self._max_capacity: Dict[ResourceType, float] = {}
        self._scaling_step: Dict[ResourceType, float] = {}
        
        # Initialize default scaling parameters
        self._initialize_scaling_parameters()
    
    def _initialize_scaling_parameters(self):
        """Initialize default scaling parameters"""
        self._min_capacity = {
            ResourceType.CPU: 2.0,
            ResourceType.MEMORY: 4.0,
            ResourceType.GPU: 0.0,
            ResourceType.DISK: 10.0
        }
        
        self._max_capacity = {
            ResourceType.CPU: 32.0,
            ResourceType.MEMORY: 64.0,
            ResourceType.GPU: 8.0,
            ResourceType.DISK: 1000.0
        }
        
        self._scaling_step = {
            ResourceType.CPU: 2.0,
            ResourceType.MEMORY: 4.0,
            ResourceType.GPU: 1.0,
            ResourceType.DISK: 50.0
        }
    
    def set_scaling_limits(self, resource_type: ResourceType, 
                          min_capacity: float, max_capacity: float, 
                          scaling_step: float):
        """Set scaling limits for a resource type"""
        with self._lock:
            self._min_capacity[resource_type] = min_capacity
            self._max_capacity[resource_type] = max_capacity
            self._scaling_step[resource_type] = scaling_step
            
            logger.info(f"Set scaling limits for {resource_type.value}: "
                       f"min={min_capacity}, max={max_capacity}, step={scaling_step}")
    
    def scale_up(self, resource_type: ResourceType, amount: float, 
                 reason: str, trigger: ScalingTrigger = ScalingTrigger.MANUAL) -> bool:
        """Scale up resources by specified amount"""
        with self._lock:
            current_capacity = self.resource_pool.get_capacity(resource_type)
            if not current_capacity:
                logger.error(f"Cannot scale {resource_type.value}: no current capacity defined")
                return False
            
            # Calculate new capacity
            current_amount = current_capacity.amount
            max_allowed = self._max_capacity.get(resource_type, float('inf'))
            new_amount = min(current_amount + amount, max_allowed)
            actual_increase = new_amount - current_amount
            
            if actual_increase <= 0:
                logger.warning(f"Cannot scale up {resource_type.value}: already at maximum capacity")
                return False
            
            # Update resource pool capacity
            try:
                self.resource_pool.set_capacity(resource_type, new_amount, current_capacity.unit)
                
                # Record scaling event
                event = ScalingEvent(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc),
                    direction=ScalingDirection.UP,
                    trigger=trigger,
                    resource_type=resource_type,
                    requested_change=amount,
                    actual_change=actual_increase,
                    reason=reason
                )
                self._scaling_events.append(event)
                self._last_scaling_time[resource_type] = event.timestamp
                
                logger.info(f"Scaled up {resource_type.value} by {actual_increase:.1f} {current_capacity.unit}: {reason}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to scale up {resource_type.value}: {e}")
                return False
    
    def scale_down(self, resource_type: ResourceType, amount: float, 
                   reason: str, trigger: ScalingTrigger = ScalingTrigger.MANUAL) -> bool:
        """Scale down resources by specified amount"""
        with self._lock:
            current_capacity = self.resource_pool.get_capacity(resource_type)
            if not current_capacity:
                logger.error(f"Cannot scale {resource_type.value}: no current capacity defined")
                return False
            
            # Calculate new capacity
            current_amount = current_capacity.amount
            min_allowed = self._min_capacity.get(resource_type, 0.0)
            new_amount = max(current_amount - amount, min_allowed)
            actual_decrease = current_amount - new_amount
            
            if actual_decrease <= 0:
                logger.warning(f"Cannot scale down {resource_type.value}: already at minimum capacity")
                return False
            
            # Check if resources are available to scale down
            available = self.resource_pool.get_available_amount(resource_type, current_capacity.unit)
            if available < actual_decrease:
                logger.warning(f"Cannot scale down {resource_type.value}: insufficient free resources")
                return False
            
            # Update resource pool capacity
            try:
                self.resource_pool.set_capacity(resource_type, new_amount, current_capacity.unit)
                
                # Record scaling event
                event = ScalingEvent(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc),
                    direction=ScalingDirection.DOWN,
                    trigger=trigger,
                    resource_type=resource_type,
                    requested_change=amount,
                    actual_change=actual_decrease,
                    reason=reason
                )
                self._scaling_events.append(event)
                self._last_scaling_time[resource_type] = event.timestamp
                
                logger.info(f"Scaled down {resource_type.value} by {actual_decrease:.1f} {current_capacity.unit}: {reason}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to scale down {resource_type.value}: {e}")
                return False
    
    def can_scale(self, resource_type: ResourceType, direction: ScalingDirection, 
                  cooldown_period: timedelta = timedelta(minutes=5)) -> bool:
        """Check if scaling is allowed for a resource type"""
        with self._lock:
            # Check cooldown period
            last_scaling = self._last_scaling_time.get(resource_type)
            if last_scaling:
                time_since_last = datetime.now(timezone.utc) - last_scaling
                if time_since_last < cooldown_period:
                    return False
            
            # Check capacity limits
            current_capacity = self.resource_pool.get_capacity(resource_type)
            if not current_capacity:
                return False
            
            current_amount = current_capacity.amount
            
            if direction == ScalingDirection.UP:
                max_allowed = self._max_capacity.get(resource_type, float('inf'))
                return current_amount < max_allowed
            elif direction == ScalingDirection.DOWN:
                min_allowed = self._min_capacity.get(resource_type, 0.0)
                return current_amount > min_allowed
            
            return True
    
    def get_scaling_recommendations(self, metrics: ScalingMetrics) -> List[Dict[str, Any]]:
        """Get scaling recommendations based on current metrics"""
        recommendations = []
        
        # CPU scaling recommendation
        if metrics.cpu_utilization > 85:
            recommendations.append({
                "resource_type": ResourceType.CPU,
                "direction": ScalingDirection.UP,
                "urgency": "high" if metrics.cpu_utilization > 95 else "medium",
                "reason": f"High CPU utilization: {metrics.cpu_utilization:.1f}%"
            })
        elif metrics.cpu_utilization < 30 and metrics.active_workers > 1:
            recommendations.append({
                "resource_type": ResourceType.CPU,
                "direction": ScalingDirection.DOWN,
                "urgency": "low",
                "reason": f"Low CPU utilization: {metrics.cpu_utilization:.1f}%"
            })
        
        # Memory scaling recommendation
        if metrics.memory_utilization > 90:
            recommendations.append({
                "resource_type": ResourceType.MEMORY,
                "direction": ScalingDirection.UP,
                "urgency": "critical" if metrics.memory_utilization > 95 else "high",
                "reason": f"High memory utilization: {metrics.memory_utilization:.1f}%"
            })
        
        # Queue-based recommendations
        if metrics.queue_depth > metrics.active_workers * 10:
            recommendations.append({
                "resource_type": ResourceType.CPU,
                "direction": ScalingDirection.UP,
                "urgency": "medium",
                "reason": f"High queue depth: {metrics.queue_depth} jobs for {metrics.active_workers} workers"
            })
        
        return recommendations
    
    def get_scaling_history(self, hours: int = 24) -> List[ScalingEvent]:
        """Get scaling event history"""
        with self._lock:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            return [event for event in self._scaling_events if event.timestamp >= cutoff]
    
    def get_scaling_statistics(self) -> Dict[str, Any]:
        """Get scaling statistics"""
        with self._lock:
            history = list(self._scaling_events)
            
            if not history:
                return {
                    "total_events": 0,
                    "scale_up_events": 0,
                    "scale_down_events": 0,
                    "success_rate": 100.0,
                    "avg_time_between_events": 0,
                    "most_scaled_resource": None
                }
            
            # Calculate statistics
            total_events = len(history)
            scale_up_events = sum(1 for e in history if e.direction == ScalingDirection.UP)
            scale_down_events = sum(1 for e in history if e.direction == ScalingDirection.DOWN)
            successful_events = sum(1 for e in history if e.success)
            
            success_rate = (successful_events / total_events * 100) if total_events > 0 else 100.0
            
            # Calculate average time between events
            if len(history) > 1:
                time_diffs = [
                    (history[i].timestamp - history[i-1].timestamp).total_seconds()
                    for i in range(1, len(history))
                ]
                avg_time_between = statistics.mean(time_diffs) / 60  # Convert to minutes
            else:
                avg_time_between = 0
            
            # Find most scaled resource
            resource_counts = defaultdict(int)
            for event in history:
                resource_counts[event.resource_type.value] += 1
            
            most_scaled_resource = max(resource_counts.items(), key=lambda x: x[1])[0] if resource_counts else None
            
            return {
                "total_events": total_events,
                "scale_up_events": scale_up_events,
                "scale_down_events": scale_down_events,
                "success_rate": success_rate,
                "avg_time_between_events_minutes": avg_time_between,
                "most_scaled_resource": most_scaled_resource,
                "resource_event_counts": dict(resource_counts)
            }


class WorkerPoolManager:
    """
    Manages a pool of worker instances for dynamic scaling.
    
    Provides functionality to start, stop, and monitor worker instances
    based on scaling decisions and resource requirements.
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self._workers: Dict[str, WorkerInstance] = {}
        self._lock = threading.RLock()
        
        # Worker configuration
        self._default_worker_capacity = {
            ResourceType.CPU: 2.0,
            ResourceType.MEMORY: 4.0,
            ResourceType.GPU: 0.0
        }
        
        # Lifecycle callbacks
        self._worker_started_callbacks: List[Callable[[WorkerInstance], None]] = []
        self._worker_stopped_callbacks: List[Callable[[WorkerInstance], None]] = []
        self._worker_failed_callbacks: List[Callable[[WorkerInstance], None]] = []
    
    def set_default_worker_capacity(self, capacity: Dict[ResourceType, float]):
        """Set default capacity for new workers"""
        with self._lock:
            self._default_worker_capacity.update(capacity)
    
    def add_worker_started_callback(self, callback: Callable[[WorkerInstance], None]):
        """Add callback for when worker starts"""
        with self._lock:
            self._worker_started_callbacks.append(callback)
    
    def add_worker_stopped_callback(self, callback: Callable[[WorkerInstance], None]):
        """Add callback for when worker stops"""
        with self._lock:
            self._worker_stopped_callbacks.append(callback)
    
    def add_worker_failed_callback(self, callback: Callable[[WorkerInstance], None]):
        """Add callback for when worker fails"""
        with self._lock:
            self._worker_failed_callbacks.append(callback)
    
    def start_workers(self, count: int, resource_type: ResourceType = ResourceType.CPU) -> List[str]:
        """Start specified number of workers"""
        started_workers = []
        
        for _ in range(count):
            worker_id = self._start_single_worker(resource_type)
            if worker_id:
                started_workers.append(worker_id)
        
        logger.info(f"Started {len(started_workers)}/{count} workers in pool '{self.name}'")
        return started_workers
    
    def _start_single_worker(self, resource_type: ResourceType) -> Optional[str]:
        """Start a single worker"""
        with self._lock:
            worker_id = f"worker-{uuid.uuid4().hex[:8]}"
            
            try:
                worker = WorkerInstance(
                    id=worker_id,
                    resource_type=resource_type,
                    capacity=self._default_worker_capacity.copy(),
                    state=WorkerState.STARTING,
                    created_at=datetime.now(timezone.utc)
                )
                
                self._workers[worker_id] = worker
                
                # Simulate worker startup process
                success = self._simulate_worker_startup(worker)
                
                if success:
                    worker.state = WorkerState.RUNNING
                    worker.started_at = datetime.now(timezone.utc)
                    worker.last_heartbeat = worker.started_at
                    
                    # Notify callbacks
                    for callback in self._worker_started_callbacks:
                        try:
                            callback(worker)
                        except Exception as e:
                            logger.error(f"Error in worker started callback: {e}")
                    
                    logger.debug(f"Worker {worker_id} started successfully")
                    return worker_id
                else:
                    worker.state = WorkerState.FAILED
                    
                    # Notify failure callbacks
                    for callback in self._worker_failed_callbacks:
                        try:
                            callback(worker)
                        except Exception as e:
                            logger.error(f"Error in worker failed callback: {e}")
                    
                    logger.error(f"Failed to start worker {worker_id}")
                    return None
                    
            except Exception as e:
                logger.error(f"Exception starting worker: {e}")
                return None
    
    def _simulate_worker_startup(self, worker: WorkerInstance) -> bool:
        """Simulate worker startup process (in real implementation, this would start actual worker)"""
        # Simulate startup time
        time.sleep(0.1)
        
        # Simulate 95% success rate
        import random
        return random.random() < 0.95
    
    def stop_workers(self, count: int) -> List[str]:
        """Stop specified number of workers"""
        with self._lock:
            running_workers = [
                worker for worker in self._workers.values()
                if worker.state == WorkerState.RUNNING
            ]
            
            # Sort by uptime (stop newest first to preserve experienced workers)
            running_workers.sort(key=lambda w: w.uptime or timedelta(0))
            
            workers_to_stop = running_workers[:count]
            stopped_workers = []
            
            for worker in workers_to_stop:
                if self._stop_single_worker(worker.id):
                    stopped_workers.append(worker.id)
            
            logger.info(f"Stopped {len(stopped_workers)}/{count} workers in pool '{self.name}'")
            return stopped_workers
    
    def _stop_single_worker(self, worker_id: str) -> bool:
        """Stop a single worker"""
        with self._lock:
            worker = self._workers.get(worker_id)
            if not worker:
                return False
            
            if worker.state not in [WorkerState.RUNNING, WorkerState.STARTING]:
                return False
            
            try:
                worker.state = WorkerState.STOPPING
                
                # Simulate worker shutdown process
                success = self._simulate_worker_shutdown(worker)
                
                if success:
                    worker.state = WorkerState.STOPPED
                    worker.stopped_at = datetime.now(timezone.utc)
                    
                    # Notify callbacks
                    for callback in self._worker_stopped_callbacks:
                        try:
                            callback(worker)
                        except Exception as e:
                            logger.error(f"Error in worker stopped callback: {e}")
                    
                    logger.debug(f"Worker {worker_id} stopped successfully")
                    return True
                else:
                    worker.state = WorkerState.FAILED
                    logger.error(f"Failed to stop worker {worker_id}")
                    return False
                    
            except Exception as e:
                logger.error(f"Exception stopping worker {worker_id}: {e}")
                worker.state = WorkerState.FAILED
                return False
    
    def _simulate_worker_shutdown(self, worker: WorkerInstance) -> bool:
        """Simulate worker shutdown process"""
        time.sleep(0.05)
        return True  # Assume shutdown always succeeds
    
    def update_worker_heartbeat(self, worker_id: str) -> bool:
        """Update worker heartbeat"""
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker and worker.state == WorkerState.RUNNING:
                worker.last_heartbeat = datetime.now(timezone.utc)
                return True
            return False
    
    def get_worker(self, worker_id: str) -> Optional[WorkerInstance]:
        """Get worker by ID"""
        with self._lock:
            return self._workers.get(worker_id)
    
    def get_workers_by_state(self, state: WorkerState) -> List[WorkerInstance]:
        """Get workers by state"""
        with self._lock:
            return [worker for worker in self._workers.values() if worker.state == state]
    
    def get_healthy_workers(self) -> List[WorkerInstance]:
        """Get all healthy workers"""
        with self._lock:
            return [worker for worker in self._workers.values() if worker.is_healthy]
    
    def cleanup_failed_workers(self) -> int:
        """Remove failed workers from the pool"""
        with self._lock:
            failed_workers = [
                worker_id for worker_id, worker in self._workers.items()
                if worker.state == WorkerState.FAILED
            ]
            
            for worker_id in failed_workers:
                del self._workers[worker_id]
            
            if failed_workers:
                logger.info(f"Cleaned up {len(failed_workers)} failed workers")
            
            return len(failed_workers)
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get worker pool statistics"""
        with self._lock:
            workers_by_state = defaultdict(int)
            total_capacity = defaultdict(float)
            healthy_workers = 0
            
            for worker in self._workers.values():
                workers_by_state[worker.state.value] += 1
                
                if worker.is_healthy:
                    healthy_workers += 1
                    for resource_type, capacity in worker.capacity.items():
                        total_capacity[resource_type.value] += capacity
            
            return {
                "pool_name": self.name,
                "total_workers": len(self._workers),
                "healthy_workers": healthy_workers,
                "workers_by_state": dict(workers_by_state),
                "total_capacity": dict(total_capacity),
                "worker_ids": list(self._workers.keys())
            }


class AutoScalingController:
    """
    Intelligent auto-scaling controller that makes scaling decisions.
    
    Combines multiple scaling policies, monitors system metrics, and
    orchestrates scaling actions across resource pools and worker managers.
    """
    
    def __init__(self, 
                 resource_scaler: ResourceScaler,
                 worker_pool_manager: WorkerPoolManager,
                 metrics: ResourceMetrics,
                 health_monitor: SystemHealthMonitor):
        self.resource_scaler = resource_scaler
        self.worker_pool_manager = worker_pool_manager
        self.metrics = metrics
        self.health_monitor = health_monitor
        
        self._lock = threading.RLock()
        self._running = False
        self._controller_thread: Optional[threading.Thread] = None
        
        # Scaling policies
        self._policies: List[ScalingPolicy] = []
        self._policy_weights: Dict[str, float] = {}
        
        # Controller configuration
        self._evaluation_interval = 30.0  # seconds
        self._metrics_history: deque = deque(maxlen=100)
        self._scaling_decisions: deque = deque(maxlen=500)
        
        # Safety configuration
        self._emergency_stop = False
        self._max_scaling_events_per_hour = 20
        
        # Initialize default policies
        self._initialize_default_policies()
    
    def _initialize_default_policies(self):
        """Initialize default scaling policies"""
        utilization_policy = UtilizationBasedPolicy(
            cpu_target=70.0,
            memory_target=80.0,
            scale_up_threshold=15.0,
            scale_down_threshold=25.0
        )
        
        queue_policy = QueueBasedPolicy(
            target_queue_per_worker=5,
            scale_up_threshold=8,
            scale_down_threshold=2
        )
        
        predictive_policy = PredictivePolicy(
            prediction_window=10,
            trend_threshold=0.1
        )
        
        self.add_policy(utilization_policy, weight=0.5, name="utilization")
        self.add_policy(queue_policy, weight=0.3, name="queue")
        self.add_policy(predictive_policy, weight=0.2, name="predictive")
    
    def add_policy(self, policy: ScalingPolicy, weight: float = 1.0, name: Optional[str] = None):
        """Add a scaling policy with optional weight"""
        with self._lock:
            self._policies.append(policy)
            policy_name = name or f"policy_{len(self._policies)}"
            self._policy_weights[policy_name] = weight
            
            logger.info(f"Added scaling policy '{policy_name}' with weight {weight}")
    
    def remove_policy(self, index: int) -> bool:
        """Remove a scaling policy by index"""
        with self._lock:
            if 0 <= index < len(self._policies):
                removed_policy = self._policies.pop(index)
                logger.info(f"Removed scaling policy at index {index}")
                return True
            return False
    
    def set_evaluation_interval(self, interval_seconds: float):
        """Set the evaluation interval"""
        with self._lock:
            self._evaluation_interval = max(10.0, interval_seconds)
            logger.info(f"Set evaluation interval to {self._evaluation_interval} seconds")
    
    def start_controller(self):
        """Start the auto-scaling controller"""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._emergency_stop = False
            self._controller_thread = threading.Thread(
                target=self._controller_loop,
                name="AutoScalingController",
                daemon=True
            )
            self._controller_thread.start()
            logger.info("Auto-scaling controller started")
    
    def stop_controller(self):
        """Stop the auto-scaling controller"""
        with self._lock:
            if not self._running:
                return
            
            self._running = False
            if self._controller_thread and self._controller_thread.is_alive():
                self._controller_thread.join(timeout=10.0)
            
            logger.info("Auto-scaling controller stopped")
    
    def emergency_stop(self):
        """Emergency stop all scaling operations"""
        with self._lock:
            self._emergency_stop = True
            logger.warning("Auto-scaling controller emergency stop activated")
    
    def _controller_loop(self):
        """Main controller loop"""
        while self._running and not self._emergency_stop:
            try:
                self._evaluate_and_scale()
                time.sleep(self._evaluation_interval)
            except Exception as e:
                logger.error(f"Error in auto-scaling controller loop: {e}")
                time.sleep(self._evaluation_interval)
    
    def _evaluate_and_scale(self):
        """Evaluate metrics and make scaling decisions"""
        with self._lock:
            # Collect current metrics
            current_metrics = self._collect_current_metrics()
            self._metrics_history.append(current_metrics)
            
            # Check safety limits
            if not self._is_scaling_safe():
                logger.debug("Scaling temporarily disabled due to safety limits")
                return
            
            # Get policy recommendations
            recommendations = self._get_policy_recommendations(current_metrics, list(self._metrics_history))
            
            # Make scaling decision
            scaling_decision = self._make_scaling_decision(recommendations)
            
            if scaling_decision:
                self._execute_scaling_decision(scaling_decision, current_metrics)
    
    def _collect_current_metrics(self) -> ScalingMetrics:
        """Collect current system metrics"""
        try:
            # Get resource utilization
            utilization = self.resource_scaler.resource_pool.get_resource_utilization()
            cpu_util = utilization.get(ResourceType.CPU, {}).get("utilization_percent", 0.0)
            memory_util = utilization.get(ResourceType.MEMORY, {}).get("utilization_percent", 0.0)
            
            # Get job metrics
            queue_metric = self.metrics.get_metric("queue_depth")
            queue_depth = int(queue_metric.get_latest_value() or 0)
            
            response_time_metric = self.metrics.get_metric("average_job_duration")
            avg_response_time = response_time_metric.get_latest_value() or 0.0
            
            failure_rate = self._calculate_failure_rate()
            
            # Get worker info
            healthy_workers = len(self.worker_pool_manager.get_healthy_workers())
            
            # Get system load
            system_load = self.metrics.calculate_system_load()
            
            return ScalingMetrics(
                timestamp=datetime.now(timezone.utc),
                cpu_utilization=cpu_util,
                memory_utilization=memory_util,
                queue_depth=queue_depth,
                average_response_time=avg_response_time,
                job_failure_rate=failure_rate,
                active_workers=healthy_workers,
                pending_jobs=queue_depth,
                system_load=system_load
            )
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return ScalingMetrics(
                timestamp=datetime.now(timezone.utc),
                cpu_utilization=0.0,
                memory_utilization=0.0,
                queue_depth=0,
                average_response_time=0.0,
                job_failure_rate=0.0,
                active_workers=0,
                pending_jobs=0,
                system_load=0.0
            )
    
    def _calculate_failure_rate(self) -> float:
        """Calculate job failure rate"""
        try:
            completed_metric = self.metrics.get_metric("total_jobs_completed")
            failed_metric = self.metrics.get_metric("total_jobs_failed")
            
            completed = completed_metric.get_latest_value() or 0
            failed = failed_metric.get_latest_value() or 0
            
            total = completed + failed
            return (failed / total * 100) if total > 0 else 0.0
        except:
            return 0.0
    
    def _is_scaling_safe(self) -> bool:
        """Check if scaling is safe to perform"""
        # Check emergency stop
        if self._emergency_stop:
            return False
        
        # Check system health
        health_status = self.health_monitor.get_current_health_status()
        if health_status.get("overall_status") == "critical":
            return False
        
        # Check scaling rate limits
        recent_events = self.resource_scaler.get_scaling_history(hours=1)
        if len(recent_events) >= self._max_scaling_events_per_hour:
            return False
        
        return True
    
    def _get_policy_recommendations(self, current_metrics: ScalingMetrics, history: List[ScalingMetrics]) -> List[Dict[str, Any]]:
        """Get recommendations from all policies"""
        recommendations = []
        
        for i, policy in enumerate(self._policies):
            try:
                direction, amount, reason = policy.should_scale(current_metrics, history)
                
                if direction != ScalingDirection.MAINTAIN:
                    policy_name = list(self._policy_weights.keys())[i]
                    weight = self._policy_weights[policy_name]
                    
                    recommendations.append({
                        "policy_name": policy_name,
                        "direction": direction,
                        "amount": amount,
                        "reason": reason,
                        "weight": weight,
                        "cooldown": policy.get_cooldown_period()
                    })
                    
            except Exception as e:
                logger.error(f"Error getting recommendation from policy {i}: {e}")
        
        return recommendations
    
    def _make_scaling_decision(self, recommendations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Make final scaling decision based on policy recommendations"""
        if not recommendations:
            return None
        
        # Group recommendations by direction
        scale_up_votes = []
        scale_down_votes = []
        
        for rec in recommendations:
            if rec["direction"] == ScalingDirection.UP:
                scale_up_votes.append(rec)
            elif rec["direction"] == ScalingDirection.DOWN:
                scale_down_votes.append(rec)
        
        # Calculate weighted scores
        scale_up_score = sum(rec["weight"] * rec["amount"] for rec in scale_up_votes)
        scale_down_score = sum(rec["weight"] * rec["amount"] for rec in scale_down_votes)
        
        # Make decision
        if scale_up_score > scale_down_score and scale_up_score > 0.5:
            # Scale up decision
            avg_amount = statistics.mean([rec["amount"] for rec in scale_up_votes])
            reasons = [rec["reason"] for rec in scale_up_votes]
            
            return {
                "direction": ScalingDirection.UP,
                "amount": avg_amount,
                "resource_type": ResourceType.CPU,  # Default to CPU scaling
                "reasons": reasons,
                "confidence": min(scale_up_score, 1.0)
            }
        
        elif scale_down_score > scale_up_score and scale_down_score > 0.3:
            # Scale down decision
            avg_amount = statistics.mean([rec["amount"] for rec in scale_down_votes])
            reasons = [rec["reason"] for rec in scale_down_votes]
            
            return {
                "direction": ScalingDirection.DOWN,
                "amount": avg_amount,
                "resource_type": ResourceType.CPU,
                "reasons": reasons,
                "confidence": min(scale_down_score, 1.0)
            }
        
        return None
    
    def _execute_scaling_decision(self, decision: Dict[str, Any], metrics: ScalingMetrics):
        """Execute a scaling decision"""
        direction = decision["direction"]
        amount = decision["amount"]
        resource_type = decision["resource_type"]
        reasons = decision["reasons"]
        confidence = decision["confidence"]
        
        # Record the decision
        decision_record = {
            "timestamp": datetime.now(timezone.utc),
            "decision": decision,
            "metrics": metrics.to_dict(),
            "executed": False
        }
        
        try:
            # Check if scaling is allowed for this resource
            if not self.resource_scaler.can_scale(resource_type, direction):
                decision_record["failure_reason"] = "Scaling not allowed (cooldown or limits)"
                self._scaling_decisions.append(decision_record)
                return
            
            # Execute scaling
            reason_summary = "; ".join(reasons[:3])  # Limit to first 3 reasons
            
            if direction == ScalingDirection.UP:
                # Scale up resources and workers
                resource_success = self.resource_scaler.scale_up(
                    resource_type, amount, reason_summary, ScalingTrigger.UTILIZATION
                )
                
                if resource_success:
                    # Start additional workers
                    workers_to_add = max(1, int(amount))
                    worker_ids = self.worker_pool_manager.start_workers(workers_to_add, resource_type)
                    
                    decision_record["executed"] = True
                    decision_record["workers_added"] = len(worker_ids)
                    logger.info(f"Scaled up: {amount:.1f} {resource_type.value}, started {len(worker_ids)} workers")
                
            elif direction == ScalingDirection.DOWN:
                # Scale down workers and resources
                workers_to_remove = max(1, int(amount))
                stopped_workers = self.worker_pool_manager.stop_workers(workers_to_remove)
                
                if stopped_workers:
                    # Scale down resources
                    resource_success = self.resource_scaler.scale_down(
                        resource_type, amount, reason_summary, ScalingTrigger.UTILIZATION
                    )
                    
                    decision_record["executed"] = True
                    decision_record["workers_removed"] = len(stopped_workers)
                    logger.info(f"Scaled down: {amount:.1f} {resource_type.value}, stopped {len(stopped_workers)} workers")
            
        except Exception as e:
            logger.error(f"Error executing scaling decision: {e}")
            decision_record["failure_reason"] = str(e)
        
        finally:
            self._scaling_decisions.append(decision_record)
    
    def get_controller_status(self) -> Dict[str, Any]:
        """Get controller status"""
        with self._lock:
            recent_decisions = [
                decision for decision in self._scaling_decisions
                if decision["timestamp"] > datetime.now(timezone.utc) - timedelta(hours=24)
            ]
            
            executed_decisions = [d for d in recent_decisions if d.get("executed", False)]
            
            return {
                "running": self._running,
                "emergency_stop": self._emergency_stop,
                "evaluation_interval": self._evaluation_interval,
                "policies_count": len(self._policies),
                "metrics_history_size": len(self._metrics_history),
                "recent_decisions_24h": len(recent_decisions),
                "executed_decisions_24h": len(executed_decisions),
                "last_evaluation": self._metrics_history[-1].timestamp if self._metrics_history else None
            }
    
    def get_scaling_decisions_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get scaling decisions history"""
        with self._lock:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            return [
                decision for decision in self._scaling_decisions
                if decision["timestamp"] >= cutoff
            ]