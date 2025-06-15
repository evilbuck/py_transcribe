"""
Tests for parallel framework dynamic scaling system.
"""

import pytest
import time
import threading
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock

from parallel_framework.resource_pool import ResourcePool, ResourceType, ResourceSpec
from parallel_framework.monitoring import ResourceMetrics, SystemHealthMonitor, AlertManager
from parallel_framework.scaling import (
    ScalingDirection, ScalingTrigger, WorkerState, ScalingEvent, ScalingMetrics,
    WorkerInstance, ScalingPolicy, UtilizationBasedPolicy, QueueBasedPolicy,
    PredictivePolicy, ResourceScaler, WorkerPoolManager, AutoScalingController
)


class TestScalingDirection:
    """Test ScalingDirection enum"""
    
    def test_scaling_directions(self):
        """Test all scaling directions are available"""
        assert ScalingDirection.UP.value == "up"
        assert ScalingDirection.DOWN.value == "down"
        assert ScalingDirection.MAINTAIN.value == "maintain"


class TestScalingTrigger:
    """Test ScalingTrigger enum"""
    
    def test_scaling_triggers(self):
        """Test all scaling triggers are available"""
        assert ScalingTrigger.UTILIZATION.value == "utilization"
        assert ScalingTrigger.QUEUE_DEPTH.value == "queue_depth"
        assert ScalingTrigger.RESPONSE_TIME.value == "response_time"
        assert ScalingTrigger.FAILURE_RATE.value == "failure_rate"
        assert ScalingTrigger.SCHEDULED.value == "scheduled"
        assert ScalingTrigger.MANUAL.value == "manual"


class TestWorkerState:
    """Test WorkerState enum"""
    
    def test_worker_states(self):
        """Test all worker states are available"""
        assert WorkerState.PENDING.value == "pending"
        assert WorkerState.STARTING.value == "starting"
        assert WorkerState.RUNNING.value == "running"
        assert WorkerState.STOPPING.value == "stopping"
        assert WorkerState.STOPPED.value == "stopped"
        assert WorkerState.FAILED.value == "failed"


class TestScalingEvent:
    """Test ScalingEvent dataclass"""
    
    def test_scaling_event_creation(self):
        """Test creating scaling event"""
        timestamp = datetime.now(timezone.utc)
        event = ScalingEvent(
            id="event1",
            timestamp=timestamp,
            direction=ScalingDirection.UP,
            trigger=ScalingTrigger.UTILIZATION,
            resource_type=ResourceType.CPU,
            requested_change=2.0,
            actual_change=2.0,
            reason="High CPU utilization",
            metadata={"policy": "utilization"}
        )
        
        assert event.id == "event1"
        assert event.timestamp == timestamp
        assert event.direction == ScalingDirection.UP
        assert event.trigger == ScalingTrigger.UTILIZATION
        assert event.resource_type == ResourceType.CPU
        assert event.requested_change == 2.0
        assert event.actual_change == 2.0
        assert event.reason == "High CPU utilization"
        assert event.metadata == {"policy": "utilization"}
        assert event.success  # actual_change == requested_change
    
    def test_scaling_event_success(self):
        """Test scaling event success detection"""
        # Successful event
        event = ScalingEvent(
            "id1", datetime.now(timezone.utc), ScalingDirection.UP,
            ScalingTrigger.MANUAL, ResourceType.CPU, 2.0, 2.0, "test"
        )
        assert event.success
        
        # Partial success (still considered success if close)
        event.actual_change = 1.95
        assert event.success
        
        # Failed event
        event.actual_change = 0.5
        assert not event.success


class TestScalingMetrics:
    """Test ScalingMetrics dataclass"""
    
    def test_scaling_metrics_creation(self):
        """Test creating scaling metrics"""
        timestamp = datetime.now(timezone.utc)
        metrics = ScalingMetrics(
            timestamp=timestamp,
            cpu_utilization=75.0,
            memory_utilization=60.0,
            queue_depth=10,
            average_response_time=120.0,
            job_failure_rate=5.0,
            active_workers=3,
            pending_jobs=10,
            system_load=65.0
        )
        
        assert metrics.timestamp == timestamp
        assert metrics.cpu_utilization == 75.0
        assert metrics.memory_utilization == 60.0
        assert metrics.queue_depth == 10
        assert metrics.average_response_time == 120.0
        assert metrics.job_failure_rate == 5.0
        assert metrics.active_workers == 3
        assert metrics.pending_jobs == 10
        assert metrics.system_load == 65.0
    
    def test_scaling_metrics_to_dict(self):
        """Test converting scaling metrics to dictionary"""
        timestamp = datetime.now(timezone.utc)
        metrics = ScalingMetrics(
            timestamp=timestamp,
            cpu_utilization=50.0,
            memory_utilization=40.0,
            queue_depth=5,
            average_response_time=60.0,
            job_failure_rate=2.0,
            active_workers=2,
            pending_jobs=5,
            system_load=45.0
        )
        
        metrics_dict = metrics.to_dict()
        assert isinstance(metrics_dict, dict)
        assert metrics_dict["timestamp"] == timestamp
        assert metrics_dict["cpu_utilization"] == 50.0
        assert metrics_dict["system_load"] == 45.0


class TestWorkerInstance:
    """Test WorkerInstance dataclass"""
    
    def test_worker_instance_creation(self):
        """Test creating worker instance"""
        created_at = datetime.now(timezone.utc)
        worker = WorkerInstance(
            id="worker1",
            resource_type=ResourceType.CPU,
            capacity={ResourceType.CPU: 2.0, ResourceType.MEMORY: 4.0},
            state=WorkerState.RUNNING,
            created_at=created_at,
            metadata={"version": "1.0"}
        )
        
        assert worker.id == "worker1"
        assert worker.resource_type == ResourceType.CPU
        assert worker.capacity[ResourceType.CPU] == 2.0
        assert worker.state == WorkerState.RUNNING
        assert worker.created_at == created_at
        assert worker.metadata == {"version": "1.0"}
    
    def test_worker_uptime(self):
        """Test worker uptime calculation"""
        created_at = datetime.now(timezone.utc)
        worker = WorkerInstance(
            "worker1", ResourceType.CPU, {}, WorkerState.PENDING, created_at
        )
        
        # No uptime if not started
        assert worker.uptime is None
        
        # Uptime calculation when running
        started_at = datetime.now(timezone.utc)
        worker.started_at = started_at
        worker.state = WorkerState.RUNNING
        
        uptime = worker.uptime
        assert uptime is not None
        assert uptime.total_seconds() >= 0
        
        # Uptime calculation when stopped
        stopped_at = started_at + timedelta(minutes=5)
        worker.stopped_at = stopped_at
        worker.state = WorkerState.STOPPED
        
        uptime = worker.uptime
        assert uptime.total_seconds() == 300  # 5 minutes = 300 seconds
    
    def test_worker_health(self):
        """Test worker health check"""
        worker = WorkerInstance(
            "worker1", ResourceType.CPU, {}, WorkerState.RUNNING,
            datetime.now(timezone.utc)
        )
        
        # Healthy if running with recent heartbeat
        worker.last_heartbeat = datetime.now(timezone.utc)
        assert worker.is_healthy
        
        # Unhealthy if no heartbeat
        worker.last_heartbeat = None
        assert not worker.is_healthy
        
        # Unhealthy if old heartbeat
        worker.last_heartbeat = datetime.now(timezone.utc) - timedelta(minutes=10)
        assert not worker.is_healthy
        
        # Unhealthy if not running
        worker.state = WorkerState.FAILED
        worker.last_heartbeat = datetime.now(timezone.utc)
        assert not worker.is_healthy


class TestUtilizationBasedPolicy:
    """Test UtilizationBasedPolicy class"""
    
    def setup_method(self):
        """Setup for each test"""
        self.policy = UtilizationBasedPolicy(
            cpu_target=70.0,
            memory_target=80.0,
            scale_up_threshold=15.0,
            scale_down_threshold=30.0,
            min_workers=1,
            max_workers=10
        )
    
    def test_policy_creation(self):
        """Test creating utilization policy"""
        assert self.policy.cpu_target == 70.0
        assert self.policy.memory_target == 80.0
        assert self.policy.scale_up_threshold == 15.0
        assert self.policy.scale_down_threshold == 30.0
        assert self.policy.min_workers == 1
        assert self.policy.max_workers == 10
    
    def test_should_scale_up_cpu(self):
        """Test scaling up based on CPU utilization"""
        metrics = ScalingMetrics(
            timestamp=datetime.now(timezone.utc),
            cpu_utilization=90.0,  # High CPU (20 points above target of 70)
            memory_utilization=75.0,  # Close to memory target of 80
            queue_depth=5,
            average_response_time=60.0,
            job_failure_rate=0.0,
            active_workers=2,
            pending_jobs=5,
            system_load=70.0
        )
        
        direction, amount, reason = self.policy.should_scale(metrics, [])
        
        # CPU deviation: abs(90 - 70) = 20, which is > scale_up_threshold (15)
        # Memory deviation: abs(75 - 80) = 5, which is < scale_up_threshold (15)
        # CPU deviation is higher, so CPU should be primary resource
        # CPU utilization (90) > CPU target (70), so should scale up
        assert direction == ScalingDirection.UP
        assert amount > 0
        assert "cpu" in reason.lower()
    
    def test_should_scale_up_memory(self):
        """Test scaling up based on memory utilization"""
        metrics = ScalingMetrics(
            timestamp=datetime.now(timezone.utc),
            cpu_utilization=65.0,  # Close to CPU target of 70
            memory_utilization=98.0,  # High memory (18 points above target of 80)
            queue_depth=5,
            average_response_time=60.0,
            job_failure_rate=0.0,
            active_workers=2,
            pending_jobs=5,
            system_load=70.0
        )
        
        direction, amount, reason = self.policy.should_scale(metrics, [])
        
        # CPU deviation: abs(65 - 70) = 5, which is < scale_up_threshold (15)
        # Memory deviation: abs(98 - 80) = 18, which is > scale_up_threshold (15)
        # Memory deviation is higher, so memory should be primary resource
        # Memory utilization (98) > memory target (80), so should scale up
        assert direction == ScalingDirection.UP
        assert amount > 0
        assert "memory" in reason.lower()
    
    def test_should_scale_down(self):
        """Test scaling down based on low utilization"""
        metrics = ScalingMetrics(
            timestamp=datetime.now(timezone.utc),
            cpu_utilization=30.0,  # Low CPU
            memory_utilization=40.0,  # Low memory
            queue_depth=1,
            average_response_time=30.0,
            job_failure_rate=0.0,
            active_workers=5,
            pending_jobs=1,
            system_load=35.0
        )
        
        direction, amount, reason = self.policy.should_scale(metrics, [])
        
        assert direction == ScalingDirection.DOWN
        assert amount > 0
        assert "utilization" in reason.lower()
    
    def test_should_maintain(self):
        """Test maintaining current scale"""
        metrics = ScalingMetrics(
            timestamp=datetime.now(timezone.utc),
            cpu_utilization=70.0,  # At target
            memory_utilization=80.0,  # At target
            queue_depth=5,
            average_response_time=60.0,
            job_failure_rate=0.0,
            active_workers=3,
            pending_jobs=5,
            system_load=75.0
        )
        
        direction, amount, reason = self.policy.should_scale(metrics, [])
        
        assert direction == ScalingDirection.MAINTAIN
        assert amount == 0.0
    
    def test_worker_limits(self):
        """Test worker count limits"""
        # Test max workers limit
        metrics = ScalingMetrics(
            timestamp=datetime.now(timezone.utc),
            cpu_utilization=95.0,
            memory_utilization=95.0,
            queue_depth=20,
            average_response_time=200.0,
            job_failure_rate=0.0,
            active_workers=10,  # At max
            pending_jobs=20,
            system_load=95.0
        )
        
        direction, amount, reason = self.policy.should_scale(metrics, [])
        assert direction == ScalingDirection.MAINTAIN
        
        # Test min workers limit
        metrics.cpu_utilization = 10.0
        metrics.memory_utilization = 10.0
        metrics.active_workers = 1  # At min
        
        direction, amount, reason = self.policy.should_scale(metrics, [])
        assert direction == ScalingDirection.MAINTAIN
    
    def test_get_cooldown_period(self):
        """Test getting cooldown period"""
        cooldown = self.policy.get_cooldown_period()
        assert isinstance(cooldown, timedelta)
        assert cooldown.total_seconds() > 0


class TestQueueBasedPolicy:
    """Test QueueBasedPolicy class"""
    
    def setup_method(self):
        """Setup for each test"""
        self.policy = QueueBasedPolicy(
            target_queue_per_worker=5,
            scale_up_threshold=10,
            scale_down_threshold=2,
            min_workers=1,
            max_workers=10
        )
    
    def test_should_scale_up_queue(self):
        """Test scaling up based on high queue depth"""
        metrics = ScalingMetrics(
            timestamp=datetime.now(timezone.utc),
            cpu_utilization=50.0,
            memory_utilization=50.0,
            queue_depth=25,  # High queue
            average_response_time=60.0,
            job_failure_rate=0.0,
            active_workers=2,  # 25/2 = 12.5 per worker > 10
            pending_jobs=25,
            system_load=50.0
        )
        
        direction, amount, reason = self.policy.should_scale(metrics, [])
        
        assert direction == ScalingDirection.UP
        assert amount > 0
        assert "queue depth" in reason.lower()
    
    def test_should_scale_down_queue(self):
        """Test scaling down based on low queue depth"""
        metrics = ScalingMetrics(
            timestamp=datetime.now(timezone.utc),
            cpu_utilization=50.0,
            memory_utilization=50.0,
            queue_depth=5,  # Low queue
            average_response_time=30.0,
            job_failure_rate=0.0,
            active_workers=5,  # 5/5 = 1 per worker < 2
            pending_jobs=5,
            system_load=30.0
        )
        
        direction, amount, reason = self.policy.should_scale(metrics, [])
        
        assert direction == ScalingDirection.DOWN
        assert amount > 0
        assert "queue depth" in reason.lower()
    
    def test_empty_queue_scaling_down(self):
        """Test scaling down with empty queue"""
        metrics = ScalingMetrics(
            timestamp=datetime.now(timezone.utc),
            cpu_utilization=20.0,
            memory_utilization=20.0,
            queue_depth=0,  # Empty queue
            average_response_time=10.0,
            job_failure_rate=0.0,
            active_workers=5,
            pending_jobs=0,
            system_load=20.0
        )
        
        direction, amount, reason = self.policy.should_scale(metrics, [])
        
        assert direction == ScalingDirection.DOWN
        assert amount > 0


class TestPredictivePolicy:
    """Test PredictivePolicy class"""
    
    def setup_method(self):
        """Setup for each test"""
        self.policy = PredictivePolicy(
            prediction_window=5,
            trend_threshold=0.1,
            min_workers=1,
            max_workers=10
        )
    
    def test_insufficient_history(self):
        """Test behavior with insufficient historical data"""
        metrics = ScalingMetrics(
            timestamp=datetime.now(timezone.utc),
            cpu_utilization=70.0,
            memory_utilization=70.0,
            queue_depth=10,
            average_response_time=60.0,
            job_failure_rate=0.0,
            active_workers=2,
            pending_jobs=10,
            system_load=70.0
        )
        
        # Empty history
        direction, amount, reason = self.policy.should_scale(metrics, [])
        assert direction == ScalingDirection.MAINTAIN
        assert "insufficient" in reason.lower()
    
    def test_predict_scale_up(self):
        """Test predicting need to scale up"""
        # Create increasing trend in CPU utilization
        base_time = datetime.now(timezone.utc)
        history = []
        
        for i in range(10):
            timestamp = base_time - timedelta(minutes=10-i)
            metrics = ScalingMetrics(
                timestamp=timestamp,
                cpu_utilization=50.0 + (i * 5),  # Increasing trend
                memory_utilization=60.0,
                queue_depth=5 + i,  # Increasing queue
                average_response_time=60.0,
                job_failure_rate=0.0,
                active_workers=2,
                pending_jobs=5 + i,
                system_load=50.0 + (i * 3)
            )
            history.append(metrics)
        
        current_metrics = history[-1]
        direction, amount, reason = self.policy.should_scale(current_metrics, history)
        
        # Should predict need to scale up
        assert direction in [ScalingDirection.UP, ScalingDirection.MAINTAIN]
        if direction == ScalingDirection.UP:
            assert "predicted" in reason.lower()
    
    def test_predict_scale_down(self):
        """Test predicting opportunity to scale down"""
        # Create decreasing trend
        base_time = datetime.now(timezone.utc)
        history = []
        
        for i in range(10):
            timestamp = base_time - timedelta(minutes=10-i)
            metrics = ScalingMetrics(
                timestamp=timestamp,
                cpu_utilization=80.0 - (i * 5),  # Decreasing trend
                memory_utilization=70.0 - (i * 3),
                queue_depth=20 - i,  # Decreasing queue
                average_response_time=60.0,
                job_failure_rate=0.0,
                active_workers=5,
                pending_jobs=20 - i,
                system_load=75.0 - (i * 4)
            )
            history.append(metrics)
        
        current_metrics = history[-1]
        direction, amount, reason = self.policy.should_scale(current_metrics, history)
        
        # Should predict opportunity to scale down or maintain
        assert direction in [ScalingDirection.DOWN, ScalingDirection.MAINTAIN]
    
    def test_calculate_trend(self):
        """Test trend calculation"""
        # Test positive trend
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        trend = self.policy._calculate_trend(values)
        assert trend > 0
        
        # Test negative trend
        values = [5.0, 4.0, 3.0, 2.0, 1.0]
        trend = self.policy._calculate_trend(values)
        assert trend < 0
        
        # Test flat trend
        values = [3.0, 3.0, 3.0, 3.0, 3.0]
        trend = self.policy._calculate_trend(values)
        assert abs(trend) < 0.1
        
        # Test insufficient data
        values = [1.0]
        trend = self.policy._calculate_trend(values)
        assert trend == 0.0


class TestResourceScaler:
    """Test ResourceScaler class"""
    
    def setup_method(self):
        """Setup for each test"""
        self.resource_pool = ResourcePool(auto_detect=False)
        self.resource_pool.set_capacity(ResourceType.CPU, 8, "cores")
        self.resource_pool.set_capacity(ResourceType.MEMORY, 16, "GB")
        
        self.metrics = ResourceMetrics(self.resource_pool)
        self.scaler = ResourceScaler(self.resource_pool, self.metrics)
    
    def test_scaler_creation(self):
        """Test creating resource scaler"""
        assert self.scaler.resource_pool is self.resource_pool
        assert self.scaler.metrics is self.metrics
        assert len(self.scaler._scaling_events) == 0
    
    def test_set_scaling_limits(self):
        """Test setting scaling limits"""
        self.scaler.set_scaling_limits(
            ResourceType.CPU, min_capacity=1.0, max_capacity=16.0, scaling_step=1.0
        )
        
        assert self.scaler._min_capacity[ResourceType.CPU] == 1.0
        assert self.scaler._max_capacity[ResourceType.CPU] == 16.0
        assert self.scaler._scaling_step[ResourceType.CPU] == 1.0
    
    def test_scale_up_success(self):
        """Test successful scale up operation"""
        success = self.scaler.scale_up(
            ResourceType.CPU, 2.0, "Test scale up", ScalingTrigger.MANUAL
        )
        
        assert success
        
        # Check capacity was increased
        new_capacity = self.resource_pool.get_capacity(ResourceType.CPU)
        assert new_capacity.amount == 10.0  # 8 + 2
        
        # Check scaling event was recorded
        assert len(self.scaler._scaling_events) == 1
        event = self.scaler._scaling_events[0]
        assert event.direction == ScalingDirection.UP
        assert event.resource_type == ResourceType.CPU
        assert event.actual_change == 2.0
    
    def test_scale_up_at_maximum(self):
        """Test scale up when at maximum capacity"""
        # Set low maximum
        self.scaler.set_scaling_limits(ResourceType.CPU, 0.0, 8.0, 1.0)
        
        success = self.scaler.scale_up(
            ResourceType.CPU, 2.0, "Test scale up", ScalingTrigger.MANUAL
        )
        
        assert not success
        
        # Capacity should remain unchanged
        capacity = self.resource_pool.get_capacity(ResourceType.CPU)
        assert capacity.amount == 8.0
    
    def test_scale_down_success(self):
        """Test successful scale down operation"""
        success = self.scaler.scale_down(
            ResourceType.CPU, 2.0, "Test scale down", ScalingTrigger.MANUAL
        )
        
        assert success
        
        # Check capacity was decreased
        new_capacity = self.resource_pool.get_capacity(ResourceType.CPU)
        assert new_capacity.amount == 6.0  # 8 - 2
        
        # Check scaling event was recorded
        assert len(self.scaler._scaling_events) == 1
        event = self.scaler._scaling_events[0]
        assert event.direction == ScalingDirection.DOWN
        assert event.resource_type == ResourceType.CPU
        assert event.actual_change == 2.0
    
    def test_scale_down_at_minimum(self):
        """Test scale down when at minimum capacity"""
        # Set high minimum
        self.scaler.set_scaling_limits(ResourceType.CPU, 8.0, 32.0, 1.0)
        
        success = self.scaler.scale_down(
            ResourceType.CPU, 2.0, "Test scale down", ScalingTrigger.MANUAL
        )
        
        assert not success
        
        # Capacity should remain unchanged
        capacity = self.resource_pool.get_capacity(ResourceType.CPU)
        assert capacity.amount == 8.0
    
    def test_can_scale_cooldown(self):
        """Test scaling cooldown functionality"""
        # Initially can scale
        assert self.scaler.can_scale(ResourceType.CPU, ScalingDirection.UP)
        
        # Scale up
        self.scaler.scale_up(ResourceType.CPU, 1.0, "Test", ScalingTrigger.MANUAL)
        
        # Should not be able to scale immediately (cooldown)
        cooldown = timedelta(minutes=1)
        assert not self.scaler.can_scale(ResourceType.CPU, ScalingDirection.UP, cooldown)
        
        # Should be able to scale with no cooldown
        assert self.scaler.can_scale(ResourceType.CPU, ScalingDirection.UP, timedelta(0))
    
    def test_get_scaling_recommendations(self):
        """Test getting scaling recommendations"""
        metrics = ScalingMetrics(
            timestamp=datetime.now(timezone.utc),
            cpu_utilization=95.0,  # High CPU - should recommend scale up
            memory_utilization=60.0,
            queue_depth=20,  # High queue
            average_response_time=120.0,
            job_failure_rate=0.0,
            active_workers=2,
            pending_jobs=20,
            system_load=85.0
        )
        
        recommendations = self.scaler.get_scaling_recommendations(metrics)
        
        assert len(recommendations) > 0
        cpu_rec = next((r for r in recommendations if r["resource_type"] == ResourceType.CPU), None)
        assert cpu_rec is not None
        assert cpu_rec["direction"] == ScalingDirection.UP
        assert cpu_rec["urgency"] in ["high", "medium"]
    
    def test_get_scaling_history(self):
        """Test getting scaling history"""
        # Perform some scaling operations
        self.scaler.scale_up(ResourceType.CPU, 1.0, "Test 1", ScalingTrigger.MANUAL)
        self.scaler.scale_down(ResourceType.CPU, 0.5, "Test 2", ScalingTrigger.UTILIZATION)
        
        history = self.scaler.get_scaling_history(24)
        assert len(history) == 2
        assert history[0].reason == "Test 1"
        assert history[1].reason == "Test 2"
    
    def test_get_scaling_statistics(self):
        """Test getting scaling statistics"""
        # Perform some scaling operations
        self.scaler.scale_up(ResourceType.CPU, 1.0, "Test up", ScalingTrigger.MANUAL)
        self.scaler.scale_down(ResourceType.CPU, 0.5, "Test down", ScalingTrigger.UTILIZATION)
        
        stats = self.scaler.get_scaling_statistics()
        
        assert stats["total_events"] == 2
        assert stats["scale_up_events"] == 1
        assert stats["scale_down_events"] == 1
        assert stats["success_rate"] == 100.0
        assert stats["most_scaled_resource"] == "cpu"


class TestWorkerPoolManager:
    """Test WorkerPoolManager class"""
    
    def setup_method(self):
        """Setup for each test"""
        self.manager = WorkerPoolManager("test_pool")
    
    def test_manager_creation(self):
        """Test creating worker pool manager"""
        assert self.manager.name == "test_pool"
        assert len(self.manager._workers) == 0
    
    def test_set_default_worker_capacity(self):
        """Test setting default worker capacity"""
        capacity = {
            ResourceType.CPU: 4.0,
            ResourceType.MEMORY: 8.0
        }
        self.manager.set_default_worker_capacity(capacity)
        
        assert self.manager._default_worker_capacity[ResourceType.CPU] == 4.0
        assert self.manager._default_worker_capacity[ResourceType.MEMORY] == 8.0
    
    def test_add_callbacks(self):
        """Test adding worker lifecycle callbacks"""
        started_calls = []
        stopped_calls = []
        failed_calls = []
        
        def on_started(worker):
            started_calls.append(worker.id)
        
        def on_stopped(worker):
            stopped_calls.append(worker.id)
        
        def on_failed(worker):
            failed_calls.append(worker.id)
        
        self.manager.add_worker_started_callback(on_started)
        self.manager.add_worker_stopped_callback(on_stopped)
        self.manager.add_worker_failed_callback(on_failed)
        
        assert len(self.manager._worker_started_callbacks) == 1
        assert len(self.manager._worker_stopped_callbacks) == 1
        assert len(self.manager._worker_failed_callbacks) == 1
    
    def test_start_workers(self):
        """Test starting workers"""
        started_ids = self.manager.start_workers(3, ResourceType.CPU)
        
        # Should start some workers (success rate is 95%)
        assert len(started_ids) >= 2  # Allow for some failures
        assert len(self.manager._workers) >= len(started_ids)
        
        # Check worker states
        for worker_id in started_ids:
            worker = self.manager.get_worker(worker_id)
            assert worker is not None
            assert worker.state == WorkerState.RUNNING
            assert worker.resource_type == ResourceType.CPU
    
    def test_stop_workers(self):
        """Test stopping workers"""
        # Start some workers first
        started_ids = self.manager.start_workers(3, ResourceType.CPU)
        
        # Stop workers
        stopped_ids = self.manager.stop_workers(2)
        
        assert len(stopped_ids) <= 2
        
        # Check worker states
        for worker_id in stopped_ids:
            worker = self.manager.get_worker(worker_id)
            assert worker.state == WorkerState.STOPPED
    
    def test_update_worker_heartbeat(self):
        """Test updating worker heartbeat"""
        started_ids = self.manager.start_workers(1, ResourceType.CPU)
        
        if started_ids:
            worker_id = started_ids[0]
            
            # Update heartbeat
            success = self.manager.update_worker_heartbeat(worker_id)
            assert success
            
            worker = self.manager.get_worker(worker_id)
            assert worker.last_heartbeat is not None
    
    def test_get_workers_by_state(self):
        """Test getting workers by state"""
        # Start some workers
        self.manager.start_workers(3, ResourceType.CPU)
        
        running_workers = self.manager.get_workers_by_state(WorkerState.RUNNING)
        assert len(running_workers) >= 2  # Account for potential failures
        
        stopped_workers = self.manager.get_workers_by_state(WorkerState.STOPPED)
        assert len(stopped_workers) == 0
    
    def test_get_healthy_workers(self):
        """Test getting healthy workers"""
        started_ids = self.manager.start_workers(2, ResourceType.CPU)
        
        healthy_workers = self.manager.get_healthy_workers()
        
        # Workers should be healthy if they started successfully
        assert len(healthy_workers) == len(started_ids)
    
    def test_cleanup_failed_workers(self):
        """Test cleaning up failed workers"""
        # Start some workers
        self.manager.start_workers(5, ResourceType.CPU)
        
        # Manually mark some as failed
        workers = list(self.manager._workers.values())
        if workers:
            workers[0].state = WorkerState.FAILED
            if len(workers) > 1:
                workers[1].state = WorkerState.FAILED
        
        # Cleanup failed workers
        cleaned_count = self.manager.cleanup_failed_workers()
        
        # Should have cleaned up failed workers
        assert cleaned_count >= 0
        
        # Failed workers should be removed
        remaining_workers = list(self.manager._workers.values())
        failed_workers = [w for w in remaining_workers if w.state == WorkerState.FAILED]
        assert len(failed_workers) == 0
    
    def test_get_pool_statistics(self):
        """Test getting pool statistics"""
        # Start some workers
        self.manager.start_workers(3, ResourceType.CPU)
        
        stats = self.manager.get_pool_statistics()
        
        assert stats["pool_name"] == "test_pool"
        assert stats["total_workers"] >= 2  # Account for potential failures
        assert "workers_by_state" in stats
        assert "total_capacity" in stats
        assert "worker_ids" in stats


class TestAutoScalingController:
    """Test AutoScalingController class"""
    
    def setup_method(self):
        """Setup for each test"""
        self.resource_pool = ResourcePool(auto_detect=False)
        self.resource_pool.set_capacity(ResourceType.CPU, 8, "cores")
        self.resource_pool.set_capacity(ResourceType.MEMORY, 16, "GB")
        
        self.metrics = ResourceMetrics(self.resource_pool)
        self.alert_manager = AlertManager(self.metrics)
        self.health_monitor = SystemHealthMonitor(
            self.resource_pool, self.metrics, self.alert_manager
        )
        self.resource_scaler = ResourceScaler(self.resource_pool, self.metrics)
        self.worker_manager = WorkerPoolManager("test_pool")
        
        self.controller = AutoScalingController(
            self.resource_scaler, self.worker_manager, self.metrics, self.health_monitor
        )
    
    def test_controller_creation(self):
        """Test creating auto-scaling controller"""
        assert self.controller.resource_scaler is self.resource_scaler
        assert self.controller.worker_pool_manager is self.worker_manager
        assert self.controller.metrics is self.metrics
        assert self.controller.health_monitor is self.health_monitor
        assert not self.controller._running
        assert len(self.controller._policies) == 3  # Default policies
    
    def test_add_remove_policy(self):
        """Test adding and removing policies"""
        initial_count = len(self.controller._policies)
        
        # Add custom policy
        custom_policy = UtilizationBasedPolicy()
        self.controller.add_policy(custom_policy, weight=0.8, name="custom")
        
        assert len(self.controller._policies) == initial_count + 1
        assert "custom" in self.controller._policy_weights
        assert self.controller._policy_weights["custom"] == 0.8
        
        # Remove policy
        success = self.controller.remove_policy(0)
        assert success
        assert len(self.controller._policies) == initial_count
    
    def test_set_evaluation_interval(self):
        """Test setting evaluation interval"""
        self.controller.set_evaluation_interval(60.0)
        assert self.controller._evaluation_interval == 60.0
        
        # Test minimum interval
        self.controller.set_evaluation_interval(5.0)
        assert self.controller._evaluation_interval == 10.0  # Minimum
    
    def test_controller_lifecycle(self):
        """Test controller start/stop lifecycle"""
        assert not self.controller._running
        
        self.controller.start_controller()
        assert self.controller._running
        assert self.controller._controller_thread is not None
        
        time.sleep(0.1)  # Let it run briefly
        
        self.controller.stop_controller()
        assert not self.controller._running
    
    def test_emergency_stop(self):
        """Test emergency stop functionality"""
        self.controller.start_controller()
        assert self.controller._running
        assert not self.controller._emergency_stop
        
        self.controller.emergency_stop()
        assert self.controller._emergency_stop
        
        self.controller.stop_controller()
    
    def test_collect_current_metrics(self):
        """Test collecting current metrics"""
        # Add some metrics data
        self.metrics.record_metric("queue_depth", 10.0)
        self.metrics.record_metric("average_job_duration", 60.0)
        
        current_metrics = self.controller._collect_current_metrics()
        
        assert isinstance(current_metrics, ScalingMetrics)
        assert current_metrics.queue_depth == 10
        assert current_metrics.average_response_time == 60.0
        assert current_metrics.timestamp is not None
    
    def test_calculate_failure_rate(self):
        """Test calculating job failure rate"""
        # No jobs initially
        failure_rate = self.controller._calculate_failure_rate()
        assert failure_rate == 0.0
        
        # Add some job metrics
        self.metrics.record_metric("total_jobs_completed", 80.0)
        self.metrics.record_metric("total_jobs_failed", 20.0)
        
        failure_rate = self.controller._calculate_failure_rate()
        assert failure_rate == 20.0  # 20/(80+20) * 100
    
    def test_is_scaling_safe(self):
        """Test scaling safety checks"""
        # Should be safe initially
        assert self.controller._is_scaling_safe()
        
        # Emergency stop should make it unsafe
        self.controller._emergency_stop = True
        assert not self.controller._is_scaling_safe()
        self.controller._emergency_stop = False
        
        # Too many recent events should make it unsafe
        for _ in range(25):  # Exceed max events per hour
            event = ScalingEvent(
                f"event_{_}", datetime.now(timezone.utc), ScalingDirection.UP,
                ScalingTrigger.MANUAL, ResourceType.CPU, 1.0, 1.0, "test"
            )
            self.controller.resource_scaler._scaling_events.append(event)
        
        assert not self.controller._is_scaling_safe()
    
    def test_get_policy_recommendations(self):
        """Test getting policy recommendations"""
        # Create metrics that should trigger scale up
        metrics = ScalingMetrics(
            timestamp=datetime.now(timezone.utc),
            cpu_utilization=90.0,  # High CPU
            memory_utilization=60.0,
            queue_depth=20,  # High queue
            average_response_time=120.0,
            job_failure_rate=0.0,
            active_workers=2,
            pending_jobs=20,
            system_load=85.0
        )
        
        recommendations = self.controller._get_policy_recommendations(metrics, [])
        
        # Should get recommendations from policies
        assert len(recommendations) > 0
        
        # Check recommendation structure
        rec = recommendations[0]
        assert "policy_name" in rec
        assert "direction" in rec
        assert "amount" in rec
        assert "reason" in rec
        assert "weight" in rec
    
    def test_make_scaling_decision(self):
        """Test making scaling decisions"""
        # Create scale up recommendations
        recommendations = [
            {
                "policy_name": "policy1",
                "direction": ScalingDirection.UP,
                "amount": 2.0,
                "reason": "High utilization",
                "weight": 0.5
            },
            {
                "policy_name": "policy2",
                "direction": ScalingDirection.UP,
                "amount": 1.0,
                "reason": "High queue",
                "weight": 0.3
            }
        ]
        
        decision = self.controller._make_scaling_decision(recommendations)
        
        assert decision is not None
        assert decision["direction"] == ScalingDirection.UP
        assert decision["amount"] > 0
        assert "reasons" in decision
        assert "confidence" in decision
    
    def test_get_controller_status(self):
        """Test getting controller status"""
        status = self.controller.get_controller_status()
        
        assert "running" in status
        assert "emergency_stop" in status
        assert "evaluation_interval" in status
        assert "policies_count" in status
        assert "metrics_history_size" in status
        assert "recent_decisions_24h" in status
        assert "executed_decisions_24h" in status
    
    def test_get_scaling_decisions_history(self):
        """Test getting scaling decisions history"""
        # Initially no decisions
        history = self.controller.get_scaling_decisions_history(24)
        assert len(history) == 0
        
        # Add a decision manually for testing
        decision_record = {
            "timestamp": datetime.now(timezone.utc),
            "decision": {"direction": ScalingDirection.UP, "amount": 1.0},
            "metrics": {},
            "executed": True
        }
        self.controller._scaling_decisions.append(decision_record)
        
        history = self.controller.get_scaling_decisions_history(24)
        assert len(history) == 1
        assert history[0]["executed"]


if __name__ == "__main__":
    pytest.main([__file__])