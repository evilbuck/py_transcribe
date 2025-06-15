"""
Tests for parallel framework job scheduling system.
"""

import pytest
import time
import threading
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

from parallel_framework.job import Job, JobHandle, JobState
from parallel_framework.resource_pool import ResourcePool, ResourceType, ResourceSpec
from parallel_framework.scheduler import (
    SchedulingAlgorithm, SchedulingPolicy, JobDependency, SchedulingContext,
    ScheduledJob, JobScheduler, DependencyManager, LoadBalancer,
    FIFOStrategy, PriorityStrategy, ResourceAwareStrategy,
    ShortestJobFirstStrategy, FairShareStrategy
)


class TestSchedulingAlgorithm:
    """Test SchedulingAlgorithm enum"""
    
    def test_scheduling_algorithms(self):
        """Test all scheduling algorithms are available"""
        assert SchedulingAlgorithm.FIFO.value == "fifo"
        assert SchedulingAlgorithm.PRIORITY.value == "priority"
        assert SchedulingAlgorithm.RESOURCE_AWARE.value == "resource_aware"
        assert SchedulingAlgorithm.SHORTEST_JOB_FIRST.value == "sjf"
        assert SchedulingAlgorithm.FAIR_SHARE.value == "fair_share"


class TestSchedulingPolicy:
    """Test SchedulingPolicy enum"""
    
    def test_scheduling_policies(self):
        """Test all scheduling policies are available"""
        assert SchedulingPolicy.IMMEDIATE.value == "immediate"
        assert SchedulingPolicy.BATCH.value == "batch"
        assert SchedulingPolicy.BACKFILL.value == "backfill"


class TestJobDependency:
    """Test JobDependency dataclass"""
    
    def test_dependency_creation(self):
        """Test creating job dependency"""
        dependency = JobDependency(
            dependent_job_id="job2",
            prerequisite_job_id="job1",
            dependency_type="success",
            metadata={"priority": "high"}
        )
        
        assert dependency.dependent_job_id == "job2"
        assert dependency.prerequisite_job_id == "job1"
        assert dependency.dependency_type == "success"
        assert dependency.metadata == {"priority": "high"}
    
    def test_dependency_defaults(self):
        """Test dependency with default values"""
        dependency = JobDependency("job2", "job1")
        
        assert dependency.dependency_type == "completion"
        assert dependency.metadata == {}


class TestSchedulingContext:
    """Test SchedulingContext dataclass"""
    
    def test_context_creation(self):
        """Test creating scheduling context"""
        context = SchedulingContext(
            available_resources={ResourceType.CPU: 8.0, ResourceType.MEMORY: 16.0},
            queue_depth=5,
            running_jobs=2,
            failed_jobs_last_hour=1,
            average_job_duration=120.0,
            current_load=0.2
        )
        
        assert context.available_resources[ResourceType.CPU] == 8.0
        assert context.queue_depth == 5
        assert context.running_jobs == 2
        assert context.current_load == 0.2
        assert context.timestamp is not None


class TestScheduledJob:
    """Test ScheduledJob dataclass"""
    
    def test_scheduled_job_creation(self):
        """Test creating scheduled job"""
        job = Job("test", "data")
        handle = JobHandle(job)
        
        scheduled_job = ScheduledJob(
            job=job,
            handle=handle,
            priority=3,
            estimated_duration=60.0,
            estimated_resources={ResourceType.CPU: 2.0},
            dependencies=["job1"],
            user_id="user123",
            queue_name="priority"
        )
        
        assert scheduled_job.job == job
        assert scheduled_job.handle == handle
        assert scheduled_job.priority == 3
        assert scheduled_job.estimated_duration == 60.0
        assert scheduled_job.dependencies == ["job1"]
        assert scheduled_job.user_id == "user123"
        assert scheduled_job.queue_name == "priority"
    
    def test_scheduled_job_defaults(self):
        """Test scheduled job with defaults"""
        job = Job("test", "data")
        handle = JobHandle(job)
        
        scheduled_job = ScheduledJob(job=job, handle=handle)
        
        assert scheduled_job.priority == 0
        assert scheduled_job.estimated_duration is None
        assert scheduled_job.estimated_resources == {}
        assert scheduled_job.dependencies == []
        assert scheduled_job.queue_name == "default"
    
    def test_scheduled_job_comparison(self):
        """Test scheduled job comparison for priority queue"""
        job1 = Job("test1", "data")
        job2 = Job("test2", "data")
        handle1 = JobHandle(job1)
        handle2 = JobHandle(job2)
        
        # Higher priority (lower number) comes first
        sj1 = ScheduledJob(job=job1, handle=handle1, priority=1)
        sj2 = ScheduledJob(job=job2, handle=handle2, priority=2)
        assert sj1 < sj2
        
        # Same priority, earlier submission comes first
        time.sleep(0.001)  # Ensure different timestamps
        sj3 = ScheduledJob(job=job1, handle=handle1, priority=1)
        assert sj1 < sj3


class TestFIFOStrategy:
    """Test FIFO scheduling strategy"""
    
    def test_fifo_selection(self):
        """Test FIFO job selection"""
        strategy = FIFOStrategy()
        
        # Create jobs with different submission times
        jobs = []
        for i in range(5):
            job = Job(f"test{i}", "data")
            handle = JobHandle(job)
            scheduled_job = ScheduledJob(job=job, handle=handle)
            scheduled_job.submitted_at = datetime.now(timezone.utc) + timedelta(seconds=i)
            jobs.append(scheduled_job)
        
        context = SchedulingContext(
            available_resources={}, queue_depth=5, running_jobs=0,
            failed_jobs_last_hour=0, average_job_duration=60.0, current_load=0.0
        )
        
        selected = strategy.select_jobs(jobs, context, max_jobs=3)
        
        assert len(selected) == 3
        # Should select jobs in submission order
        assert selected[0].job.task_type == "test0"
        assert selected[1].job.task_type == "test1"
        assert selected[2].job.task_type == "test2"
    
    def test_fifo_priority(self):
        """Test FIFO priority calculation"""
        strategy = FIFOStrategy()
        job = Job("test", "data")
        handle = JobHandle(job)
        scheduled_job = ScheduledJob(job=job, handle=handle)
        
        context = SchedulingContext(
            available_resources={}, queue_depth=1, running_jobs=0,
            failed_jobs_last_hour=0, average_job_duration=60.0, current_load=0.0
        )
        
        priority = strategy.get_job_priority(scheduled_job, context)
        assert priority == scheduled_job.submitted_at.timestamp()


class TestPriorityStrategy:
    """Test Priority scheduling strategy"""
    
    def test_priority_selection(self):
        """Test priority-based job selection"""
        strategy = PriorityStrategy()
        
        jobs = []
        priorities = [5, 1, 3, 2, 4]
        for i, priority in enumerate(priorities):
            job = Job(f"test{i}", "data")
            handle = JobHandle(job)
            scheduled_job = ScheduledJob(job=job, handle=handle, priority=priority)
            jobs.append(scheduled_job)
        
        context = SchedulingContext(
            available_resources={}, queue_depth=5, running_jobs=0,
            failed_jobs_last_hour=0, average_job_duration=60.0, current_load=0.0
        )
        
        selected = strategy.select_jobs(jobs, context, max_jobs=3)
        
        assert len(selected) == 3
        # Should select highest priority jobs (lowest numbers)
        assert selected[0].priority == 1
        assert selected[1].priority == 2
        assert selected[2].priority == 3


class TestResourceAwareStrategy:
    """Test Resource-aware scheduling strategy"""
    
    def test_resource_aware_selection(self):
        """Test resource-aware job selection"""
        strategy = ResourceAwareStrategy()
        
        jobs = []
        # Create jobs with different resource requirements
        for i in range(3):
            job = Job(f"test{i}", "data")
            handle = JobHandle(job)
            
            # Different resource patterns
            if i == 0:
                resources = {ResourceType.CPU: 2.0}  # Low CPU
            elif i == 1:
                resources = {ResourceType.CPU: 8.0}  # High CPU
            else:
                resources = {ResourceType.CPU: 4.0, ResourceType.MEMORY: 8.0}  # Mixed
            
            scheduled_job = ScheduledJob(
                job=job, handle=handle, 
                estimated_resources=resources,
                estimated_duration=60.0
            )
            jobs.append(scheduled_job)
        
        context = SchedulingContext(
            available_resources={ResourceType.CPU: 8.0, ResourceType.MEMORY: 16.0},
            queue_depth=3, running_jobs=0,
            failed_jobs_last_hour=0, average_job_duration=60.0, current_load=0.0
        )
        
        selected = strategy.select_jobs(jobs, context, max_jobs=2)
        assert len(selected) == 2


class TestShortestJobFirstStrategy:
    """Test Shortest Job First scheduling strategy"""
    
    def test_sjf_selection(self):
        """Test shortest job first selection"""
        strategy = ShortestJobFirstStrategy()
        
        jobs = []
        durations = [300, 60, 180, 30, 120]  # Different durations
        for i, duration in enumerate(durations):
            job = Job(f"test{i}", "data")
            handle = JobHandle(job)
            scheduled_job = ScheduledJob(
                job=job, handle=handle, 
                estimated_duration=duration
            )
            jobs.append(scheduled_job)
        
        context = SchedulingContext(
            available_resources={}, queue_depth=5, running_jobs=0,
            failed_jobs_last_hour=0, average_job_duration=60.0, current_load=0.0
        )
        
        selected = strategy.select_jobs(jobs, context, max_jobs=3)
        
        assert len(selected) == 3
        # Should select shortest jobs first
        assert selected[0].estimated_duration == 30
        assert selected[1].estimated_duration == 60
        assert selected[2].estimated_duration == 120


class TestFairShareStrategy:
    """Test Fair Share scheduling strategy"""
    
    def test_fair_share_selection(self):
        """Test fair share job selection"""
        strategy = FairShareStrategy()
        
        # Simulate some prior usage for user1 and user2
        strategy.user_usage["user1"] = 1.0
        strategy.user_usage["user2"] = 0.5
        
        jobs = []
        users = ["user1", "user2", "user1", "user3", "user2"]  # user3 is new
        base_time = datetime.now(timezone.utc)
        
        for i, user in enumerate(users):
            job = Job(f"test{i}", "data")
            handle = JobHandle(job)
            scheduled_job = ScheduledJob(job=job, handle=handle, user_id=user)
            # Set distinct submission times
            scheduled_job.submitted_at = base_time + timedelta(seconds=i)
            jobs.append(scheduled_job)
        
        context = SchedulingContext(
            available_resources={}, queue_depth=5, running_jobs=0,
            failed_jobs_last_hour=0, average_job_duration=60.0, current_load=0.0
        )
        
        selected = strategy.select_jobs(jobs, context, max_jobs=3)
        assert len(selected) == 3
        
        # Should favor users with lower usage
        selected_users = [job.user_id for job in selected]
        assert "user3" in selected_users  # New user should be prioritized


class TestDependencyManager:
    """Test DependencyManager class"""
    
    def test_manager_creation(self):
        """Test creating dependency manager"""
        manager = DependencyManager()
        assert len(manager._dependencies) == 0
        assert len(manager._dependents) == 0
        assert len(manager._completed_jobs) == 0
        assert len(manager._failed_jobs) == 0
    
    def test_add_dependency(self):
        """Test adding job dependency"""
        manager = DependencyManager()
        dependency = JobDependency("job2", "job1")
        
        manager.add_dependency(dependency)
        
        assert "job2" in manager._dependencies
        assert len(manager._dependencies["job2"]) == 1
        assert manager._dependencies["job2"][0] == dependency
        assert "job1" in manager._dependents
        assert "job2" in manager._dependents["job1"]
    
    def test_remove_dependency(self):
        """Test removing job dependency"""
        manager = DependencyManager()
        dependency = JobDependency("job2", "job1")
        manager.add_dependency(dependency)
        
        manager.remove_dependency("job2", "job1")
        
        assert len(manager._dependencies["job2"]) == 0
        assert "job2" not in manager._dependents["job1"]
    
    def test_mark_job_completed(self):
        """Test marking job as completed"""
        manager = DependencyManager()
        
        manager.mark_job_completed("job1")
        assert "job1" in manager._completed_jobs
        assert "job1" not in manager._failed_jobs
    
    def test_mark_job_failed(self):
        """Test marking job as failed"""
        manager = DependencyManager()
        
        manager.mark_job_failed("job1")
        assert "job1" in manager._failed_jobs
        assert "job1" not in manager._completed_jobs
    
    def test_can_schedule_job_no_dependencies(self):
        """Test scheduling job with no dependencies"""
        manager = DependencyManager()
        assert manager.can_schedule_job("job1")
    
    def test_can_schedule_job_with_completed_dependency(self):
        """Test scheduling job with completed dependency"""
        manager = DependencyManager()
        dependency = JobDependency("job2", "job1", "completion")
        manager.add_dependency(dependency)
        
        # Should not be schedulable initially
        assert not manager.can_schedule_job("job2")
        
        # Should be schedulable after prerequisite completes
        manager.mark_job_completed("job1")
        assert manager.can_schedule_job("job2")
    
    def test_can_schedule_job_with_success_dependency(self):
        """Test scheduling job with success dependency"""
        manager = DependencyManager()
        dependency = JobDependency("job2", "job1", "success")
        manager.add_dependency(dependency)
        
        # Should not be schedulable if prerequisite failed
        manager.mark_job_failed("job1")
        assert not manager.can_schedule_job("job2")
        
        # Should be schedulable if prerequisite succeeded
        manager.mark_job_completed("job1")
        assert manager.can_schedule_job("job2")
    
    def test_get_ready_jobs(self):
        """Test getting jobs with satisfied dependencies"""
        manager = DependencyManager()
        
        # Add dependencies
        manager.add_dependency(JobDependency("job2", "job1"))
        manager.add_dependency(JobDependency("job3", "job1"))
        
        # Initially only job1 is ready
        ready = manager.get_ready_jobs(["job1", "job2", "job3"])
        assert ready == ["job1"]
        
        # After job1 completes, job2 and job3 are ready
        manager.mark_job_completed("job1")
        ready = manager.get_ready_jobs(["job1", "job2", "job3"])
        assert set(ready) == {"job1", "job2", "job3"}
    
    def test_get_blocked_jobs(self):
        """Test getting jobs blocked by dependencies"""
        manager = DependencyManager()
        manager.add_dependency(JobDependency("job2", "job1"))
        
        blocked = manager.get_blocked_jobs(["job1", "job2"])
        assert blocked == ["job2"]
    
    def test_circular_dependency_detection(self):
        """Test circular dependency detection"""
        manager = DependencyManager()
        
        # Create circular dependency: job1 -> job2 -> job3 -> job1
        manager.add_dependency(JobDependency("job2", "job1"))
        manager.add_dependency(JobDependency("job3", "job2"))
        manager.add_dependency(JobDependency("job1", "job3"))
        
        assert manager.has_circular_dependency("job1")
        assert manager.has_circular_dependency("job2")
        assert manager.has_circular_dependency("job3")
    
    def test_no_circular_dependency(self):
        """Test non-circular dependency detection"""
        manager = DependencyManager()
        
        # Create linear dependency: job1 -> job2 -> job3
        manager.add_dependency(JobDependency("job2", "job1"))
        manager.add_dependency(JobDependency("job3", "job2"))
        
        assert not manager.has_circular_dependency("job1")
        assert not manager.has_circular_dependency("job2")
        assert not manager.has_circular_dependency("job3")


class TestLoadBalancer:
    """Test LoadBalancer class"""
    
    def setup_method(self):
        """Setup for each test"""
        self.resource_pool = ResourcePool(auto_detect=False)
        self.load_balancer = LoadBalancer(self.resource_pool)
    
    def test_load_balancer_creation(self):
        """Test creating load balancer"""
        assert self.load_balancer.resource_pool is self.resource_pool
        assert len(self.load_balancer._worker_loads) == 0
    
    def test_update_worker_load(self):
        """Test updating worker load"""
        self.load_balancer.update_worker_load("worker1", 0.5)
        self.load_balancer.update_worker_load("worker2", 0.8)
        
        loads = self.load_balancer.get_worker_loads()
        assert loads["worker1"] == 0.5
        assert loads["worker2"] == 0.8
    
    def test_get_least_loaded_worker(self):
        """Test getting least loaded worker"""
        # No workers initially
        assert self.load_balancer.get_least_loaded_worker() is None
        
        # Add workers with different loads
        self.load_balancer.update_worker_load("worker1", 0.8)
        self.load_balancer.update_worker_load("worker2", 0.3)
        self.load_balancer.update_worker_load("worker3", 0.6)
        
        least_loaded = self.load_balancer.get_least_loaded_worker()
        assert least_loaded == "worker2"
    
    def test_balance_queues(self):
        """Test queue balancing"""
        # Create unbalanced queues
        job1 = Job("test1", "data")
        job2 = Job("test2", "data")
        job3 = Job("test3", "data")
        job4 = Job("test4", "data")
        
        scheduled_jobs = [
            ScheduledJob(job=job1, handle=JobHandle(job1), queue_name="queue1"),
            ScheduledJob(job=job2, handle=JobHandle(job2), queue_name="queue1"),
            ScheduledJob(job=job3, handle=JobHandle(job3), queue_name="queue1"),
            ScheduledJob(job=job4, handle=JobHandle(job4), queue_name="queue2")
        ]
        
        queues = {
            "queue1": scheduled_jobs[:3],  # 3 jobs
            "queue2": scheduled_jobs[3:],  # 1 job
            "queue3": []                   # 0 jobs
        }
        
        balanced = self.load_balancer.balance_queues(queues)
        
        # Should distribute jobs more evenly
        queue_sizes = [len(jobs) for jobs in balanced.values()]
        assert max(queue_sizes) - min(queue_sizes) <= 1


class TestJobScheduler:
    """Test JobScheduler class"""
    
    def setup_method(self):
        """Setup for each test"""
        self.resource_pool = ResourcePool(auto_detect=False)
        self.resource_pool.set_capacity(ResourceType.CPU, 8, "cores")
        self.resource_pool.set_capacity(ResourceType.MEMORY, 16, "GB")
        
        self.scheduler = JobScheduler(self.resource_pool)
    
    def test_scheduler_creation(self):
        """Test creating job scheduler"""
        assert self.scheduler.resource_pool is self.resource_pool
        assert self.scheduler.algorithm == SchedulingAlgorithm.PRIORITY
        assert not self.scheduler._running
        assert len(self.scheduler._queues) == 0
        assert len(self.scheduler._running_jobs) == 0
    
    def test_submit_job(self):
        """Test submitting job to scheduler"""
        # Set policy to BATCH to prevent immediate scheduling
        self.scheduler.set_policy(SchedulingPolicy.BATCH)
        
        job = Job("test_task", "input_data")
        
        handle = self.scheduler.submit_job(
            job, 
            priority=3,
            estimated_duration=120.0,
            estimated_resources={ResourceType.CPU: 2.0},
            user_id="user123",
            queue_name="priority"
        )
        
        assert isinstance(handle, JobHandle)
        assert handle.job is job
        
        # Check job was added to queue
        assert "priority" in self.scheduler._queues
        assert len(self.scheduler._queues["priority"]) == 1
        
        scheduled_job = self.scheduler._queues["priority"][0]
        assert scheduled_job.job is job
        assert scheduled_job.priority == 3
        assert scheduled_job.estimated_duration == 120.0
        assert scheduled_job.user_id == "user123"
    
    def test_submit_job_with_dependencies(self):
        """Test submitting job with dependencies"""
        job1 = Job("task1", "data1")
        job2 = Job("task2", "data2")
        
        handle1 = self.scheduler.submit_job(job1)
        handle2 = self.scheduler.submit_job(job2, dependencies=[job1.id])
        
        # Check dependency was added
        deps = self.scheduler.dependency_manager.get_job_dependencies(job2.id)
        assert len(deps) == 1
        assert deps[0].prerequisite_job_id == job1.id
    
    def test_cancel_job(self):
        """Test cancelling a job"""
        job = Job("test_task", "input_data")
        handle = self.scheduler.submit_job(job)
        
        # Should be able to cancel queued job
        success = self.scheduler.cancel_job(job.id)
        assert success
        
        # Job should be removed from queue
        assert len(self.scheduler._queues["default"]) == 0
        assert handle.status.state == JobState.CANCELLED
    
    def test_get_queue_status(self):
        """Test getting queue status"""
        # Set policy to BATCH to prevent immediate scheduling
        self.scheduler.set_policy(SchedulingPolicy.BATCH)
        
        job1 = Job("task1", "data1")
        job2 = Job("task2", "data2")
        
        self.scheduler.submit_job(job1, queue_name="test_queue")
        self.scheduler.submit_job(job2, dependencies=[job1.id], queue_name="test_queue")
        
        status = self.scheduler.get_queue_status("test_queue")
        
        assert status["name"] == "test_queue"
        assert status["total_jobs"] == 2
        assert status["ready_jobs"] == 1  # job1 is ready, job2 is blocked
        assert status["blocked_jobs"] == 1
        assert len(status["jobs"]) == 2
    
    def test_get_scheduler_status(self):
        """Test getting scheduler status"""
        # Set policy to BATCH to prevent immediate scheduling
        self.scheduler.set_policy(SchedulingPolicy.BATCH)
        
        job = Job("test_task", "input_data")
        self.scheduler.submit_job(job)
        
        status = self.scheduler.get_scheduler_status()
        
        assert status["algorithm"] == "priority"
        assert not status["running"]
        assert status["total_queued"] == 1
        assert status["running_jobs"] == 0
        assert status["max_concurrent"] == 10
        assert "default" in status["queues"]
    
    def test_set_algorithm(self):
        """Test changing scheduling algorithm"""
        assert self.scheduler.algorithm == SchedulingAlgorithm.PRIORITY
        
        self.scheduler.set_algorithm(SchedulingAlgorithm.FIFO)
        assert self.scheduler.algorithm == SchedulingAlgorithm.FIFO
        assert isinstance(self.scheduler._current_strategy, FIFOStrategy)
    
    def test_set_policy(self):
        """Test setting scheduling policy"""
        self.scheduler.set_policy(SchedulingPolicy.BATCH)
        assert self.scheduler._policy == SchedulingPolicy.BATCH
    
    def test_set_max_concurrent_jobs(self):
        """Test setting max concurrent jobs"""
        self.scheduler.set_max_concurrent_jobs(5)
        assert self.scheduler._max_concurrent_jobs == 5
    
    def test_scheduler_lifecycle(self):
        """Test scheduler start/stop lifecycle"""
        assert not self.scheduler._running
        
        self.scheduler.start_scheduler()
        assert self.scheduler._running
        assert self.scheduler._scheduler_thread is not None
        
        time.sleep(0.1)  # Let it run briefly
        
        self.scheduler.stop_scheduler()
        assert not self.scheduler._running
    
    def test_scheduling_with_dependencies(self):
        """Test job scheduling respects dependencies"""
        job1 = Job("task1", "data1")
        job2 = Job("task2", "data2")
        
        # Submit jobs with dependency
        handle1 = self.scheduler.submit_job(job1, priority=5)
        handle2 = self.scheduler.submit_job(job2, priority=1, dependencies=[job1.id])
        
        # Start scheduler
        self.scheduler.start_scheduler()
        time.sleep(0.1)
        
        # job2 should not be scheduled yet (higher priority but blocked)
        assert len(self.scheduler._running_jobs) <= 1
        if self.scheduler._running_jobs:
            running_job_id = list(self.scheduler._running_jobs.keys())[0]
            assert running_job_id == job1.id
        
        self.scheduler.stop_scheduler()
    
    def test_thread_safety(self):
        """Test thread safety of scheduler operations"""
        # Set policy to BATCH to prevent immediate scheduling
        self.scheduler.set_policy(SchedulingPolicy.BATCH)
        
        results = []
        errors = []
        
        def submit_worker():
            try:
                for i in range(5):
                    job = Job(f"task_{threading.current_thread().ident}_{i}", "data")
                    handle = self.scheduler.submit_job(job)
                    results.append(handle)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=submit_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0
        assert len(results) == 15  # 3 threads * 5 jobs each
        
        # Check all jobs were queued
        total_queued = sum(len(jobs) for jobs in self.scheduler._queues.values())
        assert total_queued == 15


if __name__ == "__main__":
    pytest.main([__file__])