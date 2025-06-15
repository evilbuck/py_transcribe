"""
Job definition and management classes for the parallel processing framework.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Callable, List
from datetime import datetime, timezone


class JobState(Enum):
    """Enumeration of possible job states"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass
class JobStatus:
    """Current status information for a job"""
    state: JobState
    progress: float = 0.0  # 0.0 to 1.0
    message: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    worker_id: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0
    
    @property
    def is_terminal(self) -> bool:
        """Check if job is in a terminal state"""
        return self.state in {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}
    
    @property
    def is_active(self) -> bool:
        """Check if job is actively running"""
        return self.state in {JobState.QUEUED, JobState.RUNNING, JobState.RETRYING}
    
    @property
    def duration(self) -> Optional[float]:
        """Get job duration in seconds if completed"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


@dataclass
class Job:
    """
    Represents a unit of work to be processed in parallel.
    
    This is the core abstraction that allows the framework to handle
    any type of processing task.
    """
    task_type: str  # Type of task (e.g., "audio_transcription", "video_processing")
    input_data: Any  # Input data for the task
    parameters: Dict[str, Any] = field(default_factory=dict)  # Task-specific parameters
    
    # Job metadata
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = 0  # Higher numbers = higher priority
    timeout: Optional[int] = None  # Timeout in seconds
    max_retries: int = 3  # Maximum number of retry attempts
    
    # Resource requirements (optional hints)
    memory_mb: Optional[int] = None
    cpu_cores: Optional[int] = None
    gpu_memory_mb: Optional[int] = None
    
    # Tracking fields
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Dict[str, str] = field(default_factory=dict)  # User-defined tags
    
    def __post_init__(self):
        """Validate job parameters after initialization"""
        if not self.task_type:
            raise ValueError("task_type cannot be empty")
        if self.priority < 0:
            raise ValueError("priority must be non-negative")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
    
    def with_tag(self, key: str, value: str) -> 'Job':
        """Add a tag to this job (fluent interface)"""
        self.tags[key] = value
        return self
    
    def with_priority(self, priority: int) -> 'Job':
        """Set job priority (fluent interface)"""
        self.priority = priority
        return self
    
    def with_timeout(self, timeout: int) -> 'Job':
        """Set job timeout (fluent interface)"""
        self.timeout = timeout
        return self
    
    def estimate_resources(self) -> Dict[str, Any]:
        """Get resource requirements/estimates for this job"""
        return {
            'memory_mb': self.memory_mb,
            'cpu_cores': self.cpu_cores,
            'gpu_memory_mb': self.gpu_memory_mb
        }


class JobHandle:
    """
    Handle for managing and monitoring a submitted job.
    
    Provides methods to check status, wait for completion, and cancel jobs.
    """
    
    def __init__(self, job: Job, future=None):
        self.job = job
        self._future = future
        self._status = JobStatus(state=JobState.PENDING)
        self._result = None
        self._callbacks: List[Callable[[JobStatus], None]] = []
    
    @property
    def id(self) -> str:
        """Get the job ID"""
        return self.job.id
    
    @property
    def status(self) -> JobStatus:
        """Get current job status"""
        return self._status
    
    def update_status(self, state: JobState, progress: float = None, 
                     message: str = None, error: str = None) -> None:
        """Update job status (called by execution engine)"""
        if progress is not None:
            self._status.progress = max(0.0, min(1.0, progress))
        if message is not None:
            self._status.message = message
        if error is not None:
            self._status.error = error
            
        # Handle state transitions
        old_state = self._status.state
        self._status.state = state
        
        # Update timestamps
        now = datetime.now(timezone.utc)
        if state == JobState.RUNNING and old_state != JobState.RUNNING:
            self._status.started_at = now
        elif state in {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}:
            self._status.completed_at = now
            if state == JobState.COMPLETED:
                self._status.progress = 1.0
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(self._status)
            except Exception:
                pass  # Don't let callback errors affect job execution
    
    def add_callback(self, callback: Callable[[JobStatus], None]) -> None:
        """Add a callback to be notified of status changes"""
        self._callbacks.append(callback)
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> Any:
        """
        Wait for job to complete and return result.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            Job result if successful
            
        Raises:
            TimeoutError: If timeout is exceeded
            RuntimeError: If job failed or was cancelled
        """
        if self._future is not None:
            try:
                if timeout:
                    self._result = self._future.result(timeout=timeout)
                else:
                    self._result = self._future.result()
                self.update_status(JobState.COMPLETED)
                return self._result
            except Exception as e:
                self.update_status(JobState.FAILED, error=str(e))
                raise RuntimeError(f"Job {self.id} failed: {e}") from e
        
        # Polling fallback if no future available
        start_time = time.time()
        while not self._status.is_terminal:
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Job {self.id} did not complete within {timeout} seconds")
            time.sleep(0.1)
        
        if self._status.state == JobState.COMPLETED:
            return self._result
        elif self._status.state == JobState.FAILED:
            raise RuntimeError(f"Job {self.id} failed: {self._status.error}")
        elif self._status.state == JobState.CANCELLED:
            raise RuntimeError(f"Job {self.id} was cancelled")
    
    def cancel(self) -> bool:
        """
        Attempt to cancel the job.
        
        Returns:
            True if job was cancelled, False if it couldn't be cancelled
        """
        if self._status.is_terminal:
            return False
            
        if self._future is not None:
            cancelled = self._future.cancel()
            if cancelled:
                self.update_status(JobState.CANCELLED)
            return cancelled
        
        # If no future, mark as cancelled (implementation dependent)
        self.update_status(JobState.CANCELLED)
        return True
    
    def is_done(self) -> bool:
        """Check if job is complete (successfully or not)"""
        return self._status.is_terminal
    
    def is_running(self) -> bool:
        """Check if job is currently running"""
        return self._status.state == JobState.RUNNING
    
    def get_result(self) -> Any:
        """Get job result if available (non-blocking)"""
        if self._status.state == JobState.COMPLETED:
            return self._result
        elif self._status.state == JobState.FAILED:
            raise RuntimeError(f"Job {self.id} failed: {self._status.error}")
        elif self._status.state == JobState.CANCELLED:
            raise RuntimeError(f"Job {self.id} was cancelled")
        else:
            raise RuntimeError(f"Job {self.id} is not complete (state: {self._status.state})")