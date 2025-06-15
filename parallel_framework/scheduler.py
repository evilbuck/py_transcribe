"""
Job scheduling and prioritization system for the parallel processing framework.

This module provides classes for scheduling jobs with different algorithms,
managing priorities, handling dependencies, and load balancing across resources.
"""

import heapq
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

from .job import Job, JobHandle, JobState
from .resource_pool import ResourcePool, ResourceSpec, ResourceType

logger = logging.getLogger(__name__)


class SchedulingAlgorithm(Enum):
    """Available job scheduling algorithms"""
    FIFO = "fifo"  # First In, First Out
    PRIORITY = "priority"  # Priority-based scheduling
    RESOURCE_AWARE = "resource_aware"  # Resource-aware scheduling
    SHORTEST_JOB_FIRST = "sjf"  # Shortest Job First
    FAIR_SHARE = "fair_share"  # Fair share scheduling


class SchedulingPolicy(Enum):
    """Job scheduling policies"""
    IMMEDIATE = "immediate"  # Schedule immediately when resources available
    BATCH = "batch"  # Schedule in batches at intervals
    BACKFILL = "backfill"  # Backfill with smaller jobs


@dataclass
class JobDependency:
    """Represents a dependency between jobs"""
    dependent_job_id: str
    prerequisite_job_id: str
    dependency_type: str = "completion"  # completion, success, data
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulingContext:
    """Context information for scheduling decisions"""
    available_resources: Dict[ResourceType, float]
    queue_depth: int
    running_jobs: int
    failed_jobs_last_hour: int
    average_job_duration: float
    current_load: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ScheduledJob:
    """A job with scheduling metadata"""
    job: Job
    handle: JobHandle
    priority: int = 0
    estimated_duration: Optional[float] = None
    estimated_resources: Dict[ResourceType, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    scheduled_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    user_id: Optional[str] = None
    queue_name: str = "default"
    
    def __lt__(self, other):
        """Comparison for priority queue (higher priority = lower number)"""
        if not isinstance(other, ScheduledJob):
            return NotImplemented
        
        # Primary sort by priority (lower number = higher priority)
        if self.priority != other.priority:
            return self.priority < other.priority
        
        # Secondary sort by submission time (earlier = higher priority)
        return self.submitted_at < other.submitted_at


class SchedulingStrategy(ABC):
    """Abstract base class for scheduling strategies"""
    
    @abstractmethod
    def select_jobs(self, jobs: List[ScheduledJob], context: SchedulingContext, 
                   max_jobs: int) -> List[ScheduledJob]:
        """
        Select jobs to schedule based on the strategy.
        
        Args:
            jobs: Available jobs to schedule
            context: Current scheduling context
            max_jobs: Maximum number of jobs to schedule
            
        Returns:
            List of selected jobs to schedule
        """
        pass
    
    @abstractmethod
    def get_job_priority(self, job: ScheduledJob, context: SchedulingContext) -> float:
        """Calculate priority score for a job"""
        pass


class FIFOStrategy(SchedulingStrategy):
    """First In, First Out scheduling strategy"""
    
    def select_jobs(self, jobs: List[ScheduledJob], context: SchedulingContext,
                   max_jobs: int) -> List[ScheduledJob]:
        """Select jobs in submission order"""
        sorted_jobs = sorted(jobs, key=lambda j: j.submitted_at)
        return sorted_jobs[:max_jobs]
    
    def get_job_priority(self, job: ScheduledJob, context: SchedulingContext) -> float:
        """Priority based on submission time"""
        return job.submitted_at.timestamp()


class PriorityStrategy(SchedulingStrategy):
    """Priority-based scheduling strategy"""
    
    def select_jobs(self, jobs: List[ScheduledJob], context: SchedulingContext,
                   max_jobs: int) -> List[ScheduledJob]:
        """Select jobs by priority and submission time"""
        sorted_jobs = sorted(jobs)  # Uses ScheduledJob.__lt__
        return sorted_jobs[:max_jobs]
    
    def get_job_priority(self, job: ScheduledJob, context: SchedulingContext) -> float:
        """Priority based on job priority and age"""
        age_hours = (context.timestamp - job.submitted_at).total_seconds() / 3600
        return job.priority - (age_hours * 0.1)  # Age bonus


class ResourceAwareStrategy(SchedulingStrategy):
    """Resource-aware scheduling strategy"""
    
    def select_jobs(self, jobs: List[ScheduledJob], context: SchedulingContext,
                   max_jobs: int) -> List[ScheduledJob]:
        """Select jobs based on resource efficiency"""
        # Calculate resource efficiency score for each job
        scored_jobs = []
        for job in jobs:
            score = self._calculate_resource_score(job, context)
            scored_jobs.append((score, job))
        
        # Sort by score (higher is better)
        scored_jobs.sort(key=lambda x: x[0], reverse=True)
        return [job for _, job in scored_jobs[:max_jobs]]
    
    def get_job_priority(self, job: ScheduledJob, context: SchedulingContext) -> float:
        """Priority based on resource efficiency"""
        return self._calculate_resource_score(job, context)
    
    def _calculate_resource_score(self, job: ScheduledJob, context: SchedulingContext) -> float:
        """Calculate resource efficiency score"""
        if not job.estimated_resources:
            return job.priority
        
        # Calculate resource utilization efficiency
        efficiency = 0.0
        for resource_type, required in job.estimated_resources.items():
            available = context.available_resources.get(resource_type, 0)
            if available > 0:
                utilization = required / available
                efficiency += min(utilization, 1.0)  # Cap at 100%
        
        # Combine with priority and estimated duration
        duration_factor = 1.0 / max(job.estimated_duration or 1.0, 1.0)
        return efficiency * duration_factor * (10 - job.priority)


class ShortestJobFirstStrategy(SchedulingStrategy):
    """Shortest Job First scheduling strategy"""
    
    def select_jobs(self, jobs: List[ScheduledJob], context: SchedulingContext,
                   max_jobs: int) -> List[ScheduledJob]:
        """Select shortest jobs first"""
        def job_duration(job):
            return job.estimated_duration or float('inf')
        
        sorted_jobs = sorted(jobs, key=job_duration)
        return sorted_jobs[:max_jobs]
    
    def get_job_priority(self, job: ScheduledJob, context: SchedulingContext) -> float:
        """Priority based on estimated duration"""
        duration = job.estimated_duration or float('inf')
        return 1.0 / duration if duration > 0 else 0.0


class FairShareStrategy(SchedulingStrategy):
    """Fair share scheduling strategy"""
    
    def __init__(self):
        self.user_usage: Dict[str, float] = defaultdict(float)
        self.last_reset = datetime.now(timezone.utc)
    
    def select_jobs(self, jobs: List[ScheduledJob], context: SchedulingContext,
                   max_jobs: int) -> List[ScheduledJob]:
        """Select jobs to maintain fair resource sharing"""
        self._maybe_reset_usage()
        
        # Calculate fair share scores
        scored_jobs = []
        for job in jobs:
            score = self._calculate_fair_share_score(job)
            scored_jobs.append((score, job.submitted_at.timestamp(), job))
        
        # Sort by fair share score (higher is better), then by submission time
        scored_jobs.sort(key=lambda x: (-x[0], x[1]))
        return [job for _, _, job in scored_jobs[:max_jobs]]
    
    def get_job_priority(self, job: ScheduledJob, context: SchedulingContext) -> float:
        """Priority based on fair share"""
        return self._calculate_fair_share_score(job)
    
    def _calculate_fair_share_score(self, job: ScheduledJob) -> float:
        """Calculate fair share score for a job"""
        user_id = job.user_id or "anonymous"
        user_usage = self.user_usage[user_id]
        
        # Users with lower usage get higher priority
        # New users (usage = 0) get highest priority
        if user_usage == 0:
            base_score = 10.0  # High score for new users
        else:
            base_score = 1.0 / user_usage
        
        priority_bonus = (10 - job.priority) * 0.1
        
        return base_score + priority_bonus
    
    def _maybe_reset_usage(self):
        """Reset usage tracking periodically"""
        now = datetime.now(timezone.utc)
        if (now - self.last_reset).total_seconds() > 3600:  # Reset every hour
            self.user_usage.clear()
            self.last_reset = now


class DependencyManager:
    """Manages job dependencies and prerequisite resolution"""
    
    def __init__(self):
        self._dependencies: Dict[str, List[JobDependency]] = defaultdict(list)
        self._dependents: Dict[str, List[str]] = defaultdict(list)
        self._completed_jobs: Set[str] = set()
        self._failed_jobs: Set[str] = set()
        self._lock = threading.RLock()
    
    def add_dependency(self, dependency: JobDependency):
        """Add a job dependency"""
        with self._lock:
            self._dependencies[dependency.dependent_job_id].append(dependency)
            self._dependents[dependency.prerequisite_job_id].append(dependency.dependent_job_id)
    
    def remove_dependency(self, dependent_job_id: str, prerequisite_job_id: str):
        """Remove a specific dependency"""
        with self._lock:
            deps = self._dependencies.get(dependent_job_id, [])
            self._dependencies[dependent_job_id] = [
                dep for dep in deps 
                if dep.prerequisite_job_id != prerequisite_job_id
            ]
            
            dependents = self._dependents.get(prerequisite_job_id, [])
            if dependent_job_id in dependents:
                dependents.remove(dependent_job_id)
    
    def mark_job_completed(self, job_id: str):
        """Mark a job as completed"""
        with self._lock:
            self._completed_jobs.add(job_id)
            if job_id in self._failed_jobs:
                self._failed_jobs.remove(job_id)
    
    def mark_job_failed(self, job_id: str):
        """Mark a job as failed"""
        with self._lock:
            self._failed_jobs.add(job_id)
            if job_id in self._completed_jobs:
                self._completed_jobs.remove(job_id)
    
    def can_schedule_job(self, job_id: str) -> bool:
        """Check if a job's dependencies are satisfied"""
        with self._lock:
            dependencies = self._dependencies.get(job_id, [])
            
            for dep in dependencies:
                prereq_id = dep.prerequisite_job_id
                
                if dep.dependency_type == "completion":
                    if prereq_id not in self._completed_jobs:
                        return False
                elif dep.dependency_type == "success":
                    if prereq_id not in self._completed_jobs or prereq_id in self._failed_jobs:
                        return False
            
            return True
    
    def get_ready_jobs(self, job_ids: List[str]) -> List[str]:
        """Get jobs that have their dependencies satisfied"""
        with self._lock:
            return [job_id for job_id in job_ids if self.can_schedule_job(job_id)]
    
    def get_blocked_jobs(self, job_ids: List[str]) -> List[str]:
        """Get jobs that are blocked by dependencies"""
        with self._lock:
            return [job_id for job_id in job_ids if not self.can_schedule_job(job_id)]
    
    def get_job_dependencies(self, job_id: str) -> List[JobDependency]:
        """Get all dependencies for a job"""
        with self._lock:
            return self._dependencies.get(job_id, []).copy()
    
    def get_dependent_jobs(self, job_id: str) -> List[str]:
        """Get jobs that depend on this job"""
        with self._lock:
            return self._dependents.get(job_id, []).copy()
    
    def has_circular_dependency(self, job_id: str, visited: Optional[Set[str]] = None) -> bool:
        """Check for circular dependencies"""
        if visited is None:
            visited = set()
        
        if job_id in visited:
            return True
        
        visited.add(job_id)
        
        with self._lock:
            dependencies = self._dependencies.get(job_id, [])
            for dep in dependencies:
                if self.has_circular_dependency(dep.prerequisite_job_id, visited.copy()):
                    return True
        
        return False


class LoadBalancer:
    """Load balancing for distributing jobs across resources"""
    
    def __init__(self, resource_pool: ResourcePool):
        self.resource_pool = resource_pool
        self._worker_loads: Dict[str, float] = defaultdict(float)
        self._lock = threading.RLock()
    
    def select_best_resources(self, job: ScheduledJob) -> Dict[ResourceType, str]:
        """Select the best resources for a job"""
        with self._lock:
            # For now, return default resource mapping
            # In a real implementation, this would consider worker nodes, GPU devices, etc.
            return {
                ResourceType.CPU: "default_cpu_pool",
                ResourceType.MEMORY: "default_memory_pool",
                ResourceType.GPU: "default_gpu_pool"
            }
    
    def update_worker_load(self, worker_id: str, load: float):
        """Update the load for a specific worker"""
        with self._lock:
            self._worker_loads[worker_id] = load
    
    def get_least_loaded_worker(self) -> Optional[str]:
        """Get the worker with the lowest load"""
        with self._lock:
            if not self._worker_loads:
                return None
            
            return min(self._worker_loads.items(), key=lambda x: x[1])[0]
    
    def get_worker_loads(self) -> Dict[str, float]:
        """Get current worker loads"""
        with self._lock:
            return self._worker_loads.copy()
    
    def balance_queues(self, queues: Dict[str, List[ScheduledJob]]) -> Dict[str, List[ScheduledJob]]:
        """Balance jobs across queues based on load"""
        with self._lock:
            # Simple load balancing - move jobs from overloaded queues
            balanced_queues = queues.copy()
            
            total_jobs = sum(len(jobs) for jobs in queues.values())
            target_per_queue = total_jobs // len(queues) if queues else 0
            
            # Find overloaded and underloaded queues
            overloaded = []
            underloaded = []
            
            for queue_name, jobs in balanced_queues.items():
                if len(jobs) > target_per_queue + 1:
                    overloaded.append((queue_name, jobs))
                elif len(jobs) < target_per_queue:
                    underloaded.append((queue_name, jobs))
            
            # Move jobs from overloaded to underloaded queues
            for over_queue, over_jobs in overloaded:
                while len(over_jobs) > target_per_queue + 1 and underloaded:
                    under_queue, under_jobs = underloaded[0]
                    if len(under_jobs) >= target_per_queue:
                        underloaded.pop(0)
                        continue
                    
                    # Move a job
                    job_to_move = over_jobs.pop()
                    job_to_move.queue_name = under_queue
                    under_jobs.append(job_to_move)
            
            return balanced_queues


class JobScheduler:
    """
    Main job scheduler that manages job queues, priorities, and scheduling decisions.
    
    Supports multiple scheduling algorithms, job dependencies, resource awareness,
    and load balancing across available resources.
    """
    
    def __init__(self, resource_pool: ResourcePool, algorithm: SchedulingAlgorithm = SchedulingAlgorithm.PRIORITY):
        self.resource_pool = resource_pool
        self.algorithm = algorithm
        
        # Job queues
        self._queues: Dict[str, List[ScheduledJob]] = defaultdict(list)
        self._running_jobs: Dict[str, ScheduledJob] = {}
        self._completed_jobs: Dict[str, ScheduledJob] = {}
        
        # Scheduling components
        self.dependency_manager = DependencyManager()
        self.load_balancer = LoadBalancer(resource_pool)
        
        # Scheduling strategy
        self._strategies = {
            SchedulingAlgorithm.FIFO: FIFOStrategy(),
            SchedulingAlgorithm.PRIORITY: PriorityStrategy(),
            SchedulingAlgorithm.RESOURCE_AWARE: ResourceAwareStrategy(),
            SchedulingAlgorithm.SHORTEST_JOB_FIRST: ShortestJobFirstStrategy(),
            SchedulingAlgorithm.FAIR_SHARE: FairShareStrategy()
        }
        self._current_strategy = self._strategies[algorithm]
        
        # Scheduling configuration
        self._policy = SchedulingPolicy.IMMEDIATE
        self._max_concurrent_jobs = 10
        self._scheduling_interval = 1.0  # seconds
        
        # Thread safety
        self._lock = threading.RLock()
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        
        # Statistics
        self._total_scheduled = 0
        self._total_completed = 0
        self._total_failed = 0
        self._scheduling_history: List[Dict[str, Any]] = []
    
    def set_algorithm(self, algorithm: SchedulingAlgorithm):
        """Change the scheduling algorithm"""
        with self._lock:
            self.algorithm = algorithm
            self._current_strategy = self._strategies[algorithm]
            logger.info(f"Switched to {algorithm.value} scheduling algorithm")
    
    def set_policy(self, policy: SchedulingPolicy):
        """Set the scheduling policy"""
        with self._lock:
            self._policy = policy
            logger.info(f"Set scheduling policy to {policy.value}")
    
    def set_max_concurrent_jobs(self, max_jobs: int):
        """Set maximum number of concurrent jobs"""
        with self._lock:
            self._max_concurrent_jobs = max_jobs
            logger.info(f"Set max concurrent jobs to {max_jobs}")
    
    def submit_job(self, job: Job, priority: int = 5, 
                   estimated_duration: Optional[float] = None,
                   estimated_resources: Optional[Dict[ResourceType, float]] = None,
                   dependencies: Optional[List[str]] = None,
                   deadline: Optional[datetime] = None,
                   user_id: Optional[str] = None,
                   queue_name: str = "default") -> JobHandle:
        """
        Submit a job for scheduling.
        
        Args:
            job: Job to schedule
            priority: Job priority (0 = highest, 9 = lowest)
            estimated_duration: Estimated job duration in seconds
            estimated_resources: Estimated resource requirements
            dependencies: List of job IDs this job depends on
            deadline: Optional job deadline
            user_id: User ID for fair share scheduling
            queue_name: Queue to submit job to
            
        Returns:
            JobHandle for tracking the job
        """
        with self._lock:
            handle = JobHandle(job)
            
            scheduled_job = ScheduledJob(
                job=job,
                handle=handle,
                priority=priority,
                estimated_duration=estimated_duration,
                estimated_resources=estimated_resources or {},
                dependencies=dependencies or [],
                deadline=deadline,
                user_id=user_id,
                queue_name=queue_name
            )
            
            # Add dependencies to dependency manager
            for dep_job_id in (dependencies or []):
                dependency = JobDependency(
                    dependent_job_id=job.id,
                    prerequisite_job_id=dep_job_id
                )
                self.dependency_manager.add_dependency(dependency)
            
            # Add to appropriate queue
            self._queues[queue_name].append(scheduled_job)
            
            logger.info(f"Submitted job {job.id} to queue '{queue_name}' with priority {priority}")
            
            # Trigger immediate scheduling if policy allows
            if self._policy == SchedulingPolicy.IMMEDIATE and not self._running:
                self._schedule_jobs()
            
            return handle
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a scheduled or running job"""
        with self._lock:
            # Check running jobs
            if job_id in self._running_jobs:
                scheduled_job = self._running_jobs[job_id]
                if scheduled_job.handle.cancel():
                    del self._running_jobs[job_id]
                    logger.info(f"Cancelled running job {job_id}")
                    return True
            
            # Check queued jobs
            for queue_name, jobs in self._queues.items():
                for i, scheduled_job in enumerate(jobs):
                    if scheduled_job.job.id == job_id:
                        scheduled_job.handle.cancel()
                        jobs.pop(i)
                        logger.info(f"Cancelled queued job {job_id}")
                        return True
            
            return False
    
    def get_queue_status(self, queue_name: str = "default") -> Dict[str, Any]:
        """Get status of a specific queue"""
        with self._lock:
            jobs = self._queues.get(queue_name, [])
            ready_jobs = self.dependency_manager.get_ready_jobs([j.job.id for j in jobs])
            
            return {
                "name": queue_name,
                "total_jobs": len(jobs),
                "ready_jobs": len(ready_jobs),
                "blocked_jobs": len(jobs) - len(ready_jobs),
                "jobs": [
                    {
                        "id": job.job.id,
                        "priority": job.priority,
                        "submitted_at": job.submitted_at,
                        "ready": job.job.id in ready_jobs
                    }
                    for job in jobs
                ]
            }
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get overall scheduler status"""
        with self._lock:
            total_queued = sum(len(jobs) for jobs in self._queues.values())
            
            return {
                "algorithm": self.algorithm.value,
                "policy": self._policy.value,
                "running": self._running,
                "total_queued": total_queued,
                "running_jobs": len(self._running_jobs),
                "completed_jobs": len(self._completed_jobs),
                "max_concurrent": self._max_concurrent_jobs,
                "queues": list(self._queues.keys()),
                "total_scheduled": self._total_scheduled,
                "total_completed": self._total_completed,
                "total_failed": self._total_failed
            }
    
    def start_scheduler(self):
        """Start the background scheduler thread"""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._scheduler_thread = threading.Thread(
                target=self._scheduler_loop,
                name="JobScheduler",
                daemon=True
            )
            self._scheduler_thread.start()
            logger.info("Job scheduler started")
    
    def stop_scheduler(self):
        """Stop the background scheduler thread"""
        with self._lock:
            if not self._running:
                return
            
            self._running = False
            if self._scheduler_thread and self._scheduler_thread.is_alive():
                self._scheduler_thread.join(timeout=5.0)
            
            logger.info("Job scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self._running:
            try:
                self._schedule_jobs()
                self._cleanup_completed_jobs()
                time.sleep(self._scheduling_interval)
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(self._scheduling_interval)
    
    def _schedule_jobs(self):
        """Main scheduling logic"""
        with self._lock:
            if len(self._running_jobs) >= self._max_concurrent_jobs:
                return
            
            # Create scheduling context
            context = self._create_scheduling_context()
            
            # Get all ready jobs from all queues
            ready_jobs = []
            for queue_name, jobs in self._queues.items():
                queue_ready = []
                for job in jobs:
                    if self.dependency_manager.can_schedule_job(job.job.id):
                        queue_ready.append(job)
                ready_jobs.extend(queue_ready)
            
            if not ready_jobs:
                return
            
            # Apply load balancing
            if len(self._queues) > 1:
                balanced_queues = self.load_balancer.balance_queues(self._queues)
                self._queues.update(balanced_queues)
            
            # Select jobs to schedule
            max_to_schedule = self._max_concurrent_jobs - len(self._running_jobs)
            selected_jobs = self._current_strategy.select_jobs(ready_jobs, context, max_to_schedule)
            
            # Schedule selected jobs
            for scheduled_job in selected_jobs:
                if self._try_schedule_job(scheduled_job, context):
                    self._total_scheduled += 1
    
    def _try_schedule_job(self, scheduled_job: ScheduledJob, context: SchedulingContext) -> bool:
        """Try to schedule a single job"""
        try:
            # Check if we can allocate resources
            resource_specs = []
            for resource_type, amount in scheduled_job.estimated_resources.items():
                # Convert to ResourceSpec (assuming default units)
                unit_map = {
                    ResourceType.CPU: "cores",
                    ResourceType.MEMORY: "GB",
                    ResourceType.GPU: "devices"
                }
                unit = unit_map.get(resource_type, "units")
                resource_specs.append(ResourceSpec(resource_type, amount, unit))
            
            if resource_specs and not self.resource_pool.can_allocate(resource_specs):
                return False
            
            # Remove from queue
            queue_jobs = self._queues[scheduled_job.queue_name]
            if scheduled_job in queue_jobs:
                queue_jobs.remove(scheduled_job)
            
            # Mark as scheduled
            scheduled_job.scheduled_at = datetime.now(timezone.utc)
            
            # Add to running jobs
            self._running_jobs[scheduled_job.job.id] = scheduled_job
            
            # Update job handle state
            scheduled_job.handle.update_status(state=JobState.QUEUED)
            
            logger.info(f"Scheduled job {scheduled_job.job.id} with {self.algorithm.value} algorithm")
            
            # Record scheduling decision
            self._record_scheduling_decision(scheduled_job, context)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to schedule job {scheduled_job.job.id}: {e}")
            return False
    
    def _create_scheduling_context(self) -> SchedulingContext:
        """Create scheduling context for decision making"""
        with self._lock:
            total_queued = sum(len(jobs) for jobs in self._queues.values())
            
            # Get available resources from resource pool
            available_resources = {}
            try:
                utilization = self.resource_pool.get_resource_utilization()
                for resource_type, util_info in utilization.items():
                    available_resources[resource_type] = util_info.get("available_amount", 0)
            except Exception:
                available_resources = {rt: 0 for rt in ResourceType}
            
            # Calculate average job duration from completed jobs
            avg_duration = 60.0  # Default 1 minute
            if self._completed_jobs:
                durations = []
                for completed_job in self._completed_jobs.values():
                    if completed_job.estimated_duration:
                        durations.append(completed_job.estimated_duration)
                if durations:
                    avg_duration = sum(durations) / len(durations)
            
            # Calculate current load
            current_load = len(self._running_jobs) / max(self._max_concurrent_jobs, 1)
            
            return SchedulingContext(
                available_resources=available_resources,
                queue_depth=total_queued,
                running_jobs=len(self._running_jobs),
                failed_jobs_last_hour=self._total_failed,  # Simplified
                average_job_duration=avg_duration,
                current_load=current_load
            )
    
    def _cleanup_completed_jobs(self):
        """Clean up completed and failed jobs"""
        with self._lock:
            completed_ids = []
            
            for job_id, scheduled_job in self._running_jobs.items():
                if scheduled_job.handle.is_done():
                    completed_ids.append(job_id)
                    
                    # Move to completed jobs
                    self._completed_jobs[job_id] = scheduled_job
                    
                    # Update dependency manager
                    if scheduled_job.handle.status.state == JobState.COMPLETED:
                        self.dependency_manager.mark_job_completed(job_id)
                        self._total_completed += 1
                    elif scheduled_job.handle.status.state == JobState.FAILED:
                        self.dependency_manager.mark_job_failed(job_id)
                        self._total_failed += 1
            
            # Remove from running jobs
            for job_id in completed_ids:
                del self._running_jobs[job_id]
                logger.debug(f"Cleaned up completed job {job_id}")
    
    def _record_scheduling_decision(self, scheduled_job: ScheduledJob, context: SchedulingContext):
        """Record scheduling decision for analysis"""
        decision = {
            "timestamp": datetime.now(timezone.utc),
            "job_id": scheduled_job.job.id,
            "algorithm": self.algorithm.value,
            "priority": scheduled_job.priority,
            "queue_name": scheduled_job.queue_name,
            "context": {
                "queue_depth": context.queue_depth,
                "running_jobs": context.running_jobs,
                "current_load": context.current_load
            }
        }
        
        self._scheduling_history.append(decision)
        
        # Keep history size manageable
        if len(self._scheduling_history) > 1000:
            self._scheduling_history = self._scheduling_history[-1000:]