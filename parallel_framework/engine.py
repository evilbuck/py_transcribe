"""
Parallel execution engine using Dask for distributed computing.
"""

import time
import warnings
from typing import Dict, List, Optional, Any, Union
from concurrent.futures import Future
from dataclasses import dataclass

try:
    import dask
    from dask.distributed import Client, as_completed
    from dask import delayed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    warnings.warn("Dask not available. Install with: pip install dask[complete]", ImportWarning)

from .job import Job, JobHandle, JobState
from .registry import TaskRegistry, get_global_registry


@dataclass
class ExecutionConfig:
    """Configuration for the execution engine"""
    scheduler: str = "threads"  # threads, processes, synchronous, distributed
    n_workers: Optional[int] = None  # None for auto-detect
    threads_per_worker: int = 1
    memory_limit: str = "auto"  # e.g., "4GB", "auto"
    dashboard_address: Optional[str] = None  # e.g., ":8787"
    silence_logs: bool = True
    timeout: Optional[float] = None  # Default timeout for jobs
    
    def __post_init__(self):
        valid_schedulers = ["threads", "processes", "synchronous", "distributed"]
        if self.scheduler not in valid_schedulers:
            raise ValueError(f"scheduler must be one of {valid_schedulers}")


class ParallelExecutionEngine:
    """
    Core execution engine using Dask for parallel processing.
    
    Provides a high-level interface for submitting and managing jobs
    across different types of computational backends.
    """
    
    def __init__(self, config: Optional[ExecutionConfig] = None, 
                 registry: Optional[TaskRegistry] = None):
        self.config = config or ExecutionConfig()
        self.registry = registry or get_global_registry()
        self._client: Optional[Any] = None
        self._active_jobs: Dict[str, JobHandle] = {}
        self._shutdown = False
        
        # Initialize Dask client
        self._initialize_dask()
    
    def _initialize_dask(self) -> None:
        """Initialize Dask client based on configuration"""
        if not DASK_AVAILABLE:
            raise RuntimeError("Dask is not available. Install with: pip install dask[complete]")
        
        try:
            if self.config.scheduler == "distributed":
                # Connect to existing cluster or start local cluster
                self._client = Client(
                    dashboard_address=self.config.dashboard_address,
                    silence_logs=self.config.silence_logs,
                    timeout=30
                )
            elif self.config.scheduler in ["threads", "processes"]:
                # Use local cluster
                from dask.distributed import LocalCluster
                
                cluster = LocalCluster(
                    n_workers=self.config.n_workers,
                    threads_per_worker=self.config.threads_per_worker,
                    memory_limit=self.config.memory_limit,
                    dashboard_address=self.config.dashboard_address,
                    silence_logs=self.config.silence_logs,
                    processes=self.config.scheduler == "processes"
                )
                self._client = Client(cluster)
            else:
                # Synchronous scheduler - no client needed
                self._client = None
                
        except Exception as e:
            warnings.warn(f"Failed to initialize Dask client: {e}. Using synchronous execution.")
            self.config.scheduler = "synchronous"
            self._client = None
    
    def submit_job(self, job: Job) -> JobHandle:
        """
        Submit a single job for execution.
        
        Args:
            job: Job to execute
            
        Returns:
            JobHandle for monitoring and managing the job
        """
        if self._shutdown:
            raise RuntimeError("Engine has been shut down")
        
        # Validate job
        errors = self.registry.validate_job(job)
        if errors:
            raise ValueError(f"Job validation failed: {'; '.join(errors)}")
        
        # Create handle
        handle = JobHandle(job)
        handle.update_status(JobState.QUEUED, message="Job queued for execution")
        
        # Submit to appropriate backend
        if self.config.scheduler == "synchronous":
            future = self._submit_synchronous(job, handle)
        else:
            future = self._submit_dask(job, handle)
        
        handle._future = future
        self._active_jobs[job.id] = handle
        
        return handle
    
    def submit_batch(self, jobs: List[Job]) -> List[JobHandle]:
        """
        Submit multiple jobs for execution.
        
        Args:
            jobs: List of jobs to execute
            
        Returns:
            List of JobHandles
        """
        handles = []
        for job in jobs:
            handle = self.submit_job(job)
            handles.append(handle)
        return handles
    
    def _submit_synchronous(self, job: Job, handle: JobHandle) -> Future:
        """Submit job for synchronous execution"""
        from concurrent.futures import Future
        
        future = Future()
        
        try:
            handle.update_status(JobState.RUNNING, message="Executing synchronously")
            
            # Execute task directly
            result = self.registry.execute_task(job)
            
            # Set result and complete
            handle._result = result
            handle.update_status(JobState.COMPLETED, message="Job completed")
            future.set_result(result)
            
        except Exception as e:
            handle.update_status(JobState.FAILED, error=str(e))
            future.set_exception(e)
        
        return future
    
    def _submit_dask(self, job: Job, handle: JobHandle) -> Any:
        """Submit job to Dask for execution"""
        if self._client is None:
            raise RuntimeError("Dask client not available")
        
        # Create delayed task
        @delayed
        def execute_job_task(job_data):
            # Reconstruct job (needed for serialization)
            reconstructed_job = Job(
                task_type=job_data['task_type'],
                input_data=job_data['input_data'],
                parameters=job_data['parameters'],
                id=job_data['id']
            )
            
            # Execute using registry
            return self.registry.execute_task(reconstructed_job)
        
        # Serialize job data
        job_data = {
            'task_type': job.task_type,
            'input_data': job.input_data,
            'parameters': job.parameters,
            'id': job.id
        }
        
        # Submit to Dask
        delayed_task = execute_job_task(job_data)
        future = self._client.compute(delayed_task)
        
        # Monitor execution in background
        self._monitor_dask_job(handle, future)
        
        return future
    
    def _monitor_dask_job(self, handle: JobHandle, future: Any) -> None:
        """Monitor Dask job execution and update handle status"""
        def update_status():
            try:
                handle.update_status(JobState.RUNNING, message="Executing on Dask")
                
                # Wait for completion
                result = future.result()
                handle._result = result
                handle.update_status(JobState.COMPLETED, message="Job completed")
                
            except Exception as e:
                handle.update_status(JobState.FAILED, error=str(e))
        
        # Submit monitoring as separate task
        if self._client:
            self._client.submit(update_status)
    
    def wait_for_jobs(self, handles: List[JobHandle], timeout: Optional[float] = None) -> List[Any]:
        """
        Wait for multiple jobs to complete.
        
        Args:
            handles: List of job handles to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            List of results in the same order as handles
        """
        results = []
        
        for handle in handles:
            try:
                result = handle.wait_for_completion(timeout)
                results.append(result)
            except Exception as e:
                results.append(e)
        
        return results
    
    def cancel_job(self, handle: JobHandle) -> bool:
        """
        Cancel a running job.
        
        Args:
            handle: Job handle to cancel
            
        Returns:
            True if job was cancelled, False otherwise
        """
        success = handle.cancel()
        
        if success and handle.id in self._active_jobs:
            del self._active_jobs[handle.id]
        
        return success
    
    def get_active_jobs(self) -> List[JobHandle]:
        """Get list of currently active jobs"""
        return list(self._active_jobs.values())
    
    def get_job_by_id(self, job_id: str) -> Optional[JobHandle]:
        """Get job handle by ID"""
        return self._active_jobs.get(job_id)
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about the compute cluster"""
        if self._client is None:
            return {
                'scheduler': self.config.scheduler,
                'workers': 0,
                'threads': 0,
                'memory': '0 GB'
            }
        
        try:
            info = self._client.scheduler_info()
            workers = info.get('workers', {})
            
            total_threads = sum(w.get('nthreads', 0) for w in workers.values())
            total_memory = sum(w.get('memory_limit', 0) for w in workers.values())
            
            return {
                'scheduler': self.config.scheduler,
                'workers': len(workers),
                'threads': total_threads,
                'memory': f"{total_memory / (1024**3):.1f} GB" if total_memory else "0 GB",
                'address': getattr(self._client.scheduler, 'address', 'local')
            }
        except Exception:
            return {
                'scheduler': self.config.scheduler,
                'workers': 'unknown',
                'threads': 'unknown',
                'memory': 'unknown'
            }
    
    def shutdown(self, timeout: float = 10.0) -> None:
        """
        Shutdown the execution engine.
        
        Args:
            timeout: Time to wait for jobs to complete before force shutdown
        """
        self._shutdown = True
        
        # Cancel all active jobs
        for handle in list(self._active_jobs.values()):
            handle.cancel()
        
        # Wait for jobs to finish or timeout
        start_time = time.time()
        while self._active_jobs and (time.time() - start_time) < timeout:
            time.sleep(0.1)
            # Remove completed jobs
            completed = [
                job_id for job_id, handle in self._active_jobs.items() 
                if handle.is_done()
            ]
            for job_id in completed:
                del self._active_jobs[job_id]
        
        # Close Dask client
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()


# Convenience functions
def create_engine(scheduler: str = "threads", n_workers: Optional[int] = None, **kwargs) -> ParallelExecutionEngine:
    """
    Create a ParallelExecutionEngine with common configurations.
    
    Args:
        scheduler: Type of scheduler ('threads', 'processes', 'synchronous', 'distributed')
        n_workers: Number of workers (None for auto-detect)
        **kwargs: Additional configuration options
        
    Returns:
        Configured ParallelExecutionEngine
    """
    config = ExecutionConfig(scheduler=scheduler, n_workers=n_workers, **kwargs)
    return ParallelExecutionEngine(config)


def execute_job_simple(job: Job, scheduler: str = "synchronous") -> Any:
    """
    Simple function to execute a single job.
    
    Args:
        job: Job to execute
        scheduler: Scheduler type to use
        
    Returns:
        Job result
    """
    with create_engine(scheduler=scheduler) as engine:
        handle = engine.submit_job(job)
        return handle.wait_for_completion()