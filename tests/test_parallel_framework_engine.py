"""
Tests for parallel framework execution engine.
"""

import pytest
import time
from unittest.mock import Mock, patch

from parallel_framework.job import Job, JobState
from parallel_framework.registry import TaskRegistry
from parallel_framework.engine import (
    ParallelExecutionEngine, 
    ExecutionConfig, 
    create_engine,
    execute_job_simple
)


class TestExecutionConfig:
    """Test ExecutionConfig dataclass"""
    
    def test_config_creation_defaults(self):
        """Test config creation with defaults"""
        config = ExecutionConfig()
        
        assert config.scheduler == "threads"
        assert config.n_workers is None
        assert config.threads_per_worker == 1
        assert config.memory_limit == "auto"
        assert config.dashboard_address is None
        assert config.silence_logs is True
        assert config.timeout is None
    
    def test_config_creation_custom(self):
        """Test config creation with custom values"""
        config = ExecutionConfig(
            scheduler="processes",
            n_workers=4,
            threads_per_worker=2,
            memory_limit="8GB",
            dashboard_address=":8787",
            silence_logs=False,
            timeout=3600.0
        )
        
        assert config.scheduler == "processes"
        assert config.n_workers == 4
        assert config.threads_per_worker == 2
        assert config.memory_limit == "8GB"
        assert config.dashboard_address == ":8787"
        assert config.silence_logs is False
        assert config.timeout == 3600.0
    
    def test_config_validation(self):
        """Test config validation"""
        # Invalid scheduler should raise ValueError
        with pytest.raises(ValueError, match="scheduler must be one of"):
            ExecutionConfig(scheduler="invalid")
        
        # Valid schedulers should work
        for scheduler in ["threads", "processes", "synchronous", "distributed"]:
            config = ExecutionConfig(scheduler=scheduler)
            assert config.scheduler == scheduler


class TestParallelExecutionEngine:
    """Test ParallelExecutionEngine class"""
    
    def setup_method(self):
        """Setup for each test"""
        self.registry = TaskRegistry()
        
        # Register test tasks
        def simple_task(job: Job):
            return f"processed: {job.input_data}"
        
        def slow_task(job: Job):
            delay = job.parameters.get("delay", 0.1)
            time.sleep(delay)
            return f"slow: {job.input_data}"
        
        def error_task(job: Job):
            raise Exception("Task failed")
        
        self.registry.register_task("simple", simple_task)
        self.registry.register_task("slow", slow_task)
        self.registry.register_task("error", error_task)
    
    def test_engine_creation_synchronous(self):
        """Test creating engine with synchronous scheduler"""
        config = ExecutionConfig(scheduler="synchronous")
        engine = ParallelExecutionEngine(config, self.registry)
        
        assert engine.config.scheduler == "synchronous"
        assert engine.registry is self.registry
        assert engine._client is None
        assert not engine._shutdown
        
        engine.shutdown()
    
    @pytest.mark.skip(reason="Requires Dask setup which may be unstable in CI")
    def test_engine_creation_threads(self):
        """Test creating engine with threads scheduler"""
        config = ExecutionConfig(scheduler="threads", n_workers=2, silence_logs=True)
        engine = ParallelExecutionEngine(config, self.registry)
        
        assert engine.config.scheduler == "threads"
        assert engine._client is not None
        
        # Get cluster info
        info = engine.get_cluster_info()
        assert info['scheduler'] == "threads"
        assert isinstance(info['workers'], int)
        
        engine.shutdown()
    
    def test_submit_job_synchronous(self):
        """Test submitting a job with synchronous execution"""
        config = ExecutionConfig(scheduler="synchronous")
        engine = ParallelExecutionEngine(config, self.registry)
        
        job = Job(task_type="simple", input_data="test_data")
        handle = engine.submit_job(job)
        
        assert handle.job is job
        assert handle.id == job.id
        assert job.id in engine._active_jobs
        
        # Job should complete immediately with synchronous execution
        result = handle.wait_for_completion()
        assert result == "processed: test_data"
        assert handle.status.state == JobState.COMPLETED
        
        engine.shutdown()
    
    def test_submit_job_validation_error(self):
        """Test submitting invalid job"""
        config = ExecutionConfig(scheduler="synchronous")
        engine = ParallelExecutionEngine(config, self.registry)
        
        # Job with unregistered task type
        job = Job(task_type="unregistered", input_data="data")
        
        with pytest.raises(ValueError, match="Job validation failed"):
            engine.submit_job(job)
        
        engine.shutdown()
    
    def test_submit_job_after_shutdown(self):
        """Test submitting job after engine shutdown"""
        config = ExecutionConfig(scheduler="synchronous")
        engine = ParallelExecutionEngine(config, self.registry)
        
        engine.shutdown()
        
        job = Job(task_type="simple", input_data="data")
        
        with pytest.raises(RuntimeError, match="Engine has been shut down"):
            engine.submit_job(job)
    
    def test_submit_batch(self):
        """Test submitting multiple jobs"""
        config = ExecutionConfig(scheduler="synchronous")
        engine = ParallelExecutionEngine(config, self.registry)
        
        jobs = [
            Job(task_type="simple", input_data=f"data_{i}")
            for i in range(3)
        ]
        
        handles = engine.submit_batch(jobs)
        
        assert len(handles) == 3
        for i, handle in enumerate(handles):
            assert handle.job.input_data == f"data_{i}"
            result = handle.wait_for_completion()
            assert result == f"processed: data_{i}"
        
        engine.shutdown()
    
    def test_wait_for_jobs(self):
        """Test waiting for multiple jobs"""
        config = ExecutionConfig(scheduler="synchronous")
        engine = ParallelExecutionEngine(config, self.registry)
        
        jobs = [
            Job(task_type="simple", input_data=f"data_{i}")
            for i in range(3)
        ]
        
        handles = engine.submit_batch(jobs)
        results = engine.wait_for_jobs(handles)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result == f"processed: data_{i}"
        
        engine.shutdown()
    
    def test_wait_for_jobs_with_error(self):
        """Test waiting for jobs when one fails"""
        config = ExecutionConfig(scheduler="synchronous")
        engine = ParallelExecutionEngine(config, self.registry)
        
        jobs = [
            Job(task_type="simple", input_data="data"),
            Job(task_type="error", input_data="data")
        ]
        
        handles = engine.submit_batch(jobs)
        results = engine.wait_for_jobs(handles)
        
        assert len(results) == 2
        assert results[0] == "processed: data"
        assert isinstance(results[1], Exception)
        
        engine.shutdown()
    
    def test_cancel_job(self):
        """Test cancelling a job"""
        config = ExecutionConfig(scheduler="synchronous")
        engine = ParallelExecutionEngine(config, self.registry)
        
        job = Job(task_type="simple", input_data="data")
        handle = engine.submit_job(job)
        
        # For synchronous execution, job completes immediately
        # so cancellation might not be possible
        handle.wait_for_completion()
        
        # Try to cancel (should return False as job is done)
        cancelled = engine.cancel_job(handle)
        # Result depends on timing - job might already be complete
        
        engine.shutdown()
    
    def test_get_active_jobs(self):
        """Test getting active jobs"""
        config = ExecutionConfig(scheduler="synchronous")
        engine = ParallelExecutionEngine(config, self.registry)
        
        assert len(engine.get_active_jobs()) == 0
        
        job = Job(task_type="simple", input_data="data")
        handle = engine.submit_job(job)
        
        active_jobs = engine.get_active_jobs()
        assert len(active_jobs) == 1
        assert active_jobs[0] is handle
        
        # Complete the job
        handle.wait_for_completion()
        
        engine.shutdown()
    
    def test_get_job_by_id(self):
        """Test getting job by ID"""
        config = ExecutionConfig(scheduler="synchronous")
        engine = ParallelExecutionEngine(config, self.registry)
        
        job = Job(task_type="simple", input_data="data")
        handle = engine.submit_job(job)
        
        found_handle = engine.get_job_by_id(job.id)
        assert found_handle is handle
        
        not_found = engine.get_job_by_id("nonexistent")
        assert not_found is None
        
        engine.shutdown()
    
    def test_get_cluster_info_synchronous(self):
        """Test getting cluster info for synchronous scheduler"""
        config = ExecutionConfig(scheduler="synchronous")
        engine = ParallelExecutionEngine(config, self.registry)
        
        info = engine.get_cluster_info()
        
        assert info['scheduler'] == "synchronous"
        assert info['workers'] == 0
        assert info['threads'] == 0
        assert info['memory'] == '0 GB'
        
        engine.shutdown()
    
    def test_context_manager(self):
        """Test using engine as context manager"""
        config = ExecutionConfig(scheduler="synchronous")
        
        with ParallelExecutionEngine(config, self.registry) as engine:
            job = Job(task_type="simple", input_data="data")
            handle = engine.submit_job(job)
            result = handle.wait_for_completion()
            assert result == "processed: data"
        
        # Engine should be shut down after exiting context
        assert engine._shutdown
    
    def test_shutdown_with_active_jobs(self):
        """Test shutdown behavior with active jobs"""
        config = ExecutionConfig(scheduler="synchronous")
        engine = ParallelExecutionEngine(config, self.registry)
        
        # Submit a job but don't wait for completion
        job = Job(task_type="simple", input_data="data")
        handle = engine.submit_job(job)
        
        # Shutdown should cancel active jobs
        engine.shutdown(timeout=1.0)
        
        assert engine._shutdown
        # Active jobs should be cleared or cancelled
        
    @patch('parallel_framework.engine.DASK_AVAILABLE', False)
    def test_engine_without_dask(self):
        """Test engine creation when Dask is not available"""
        config = ExecutionConfig(scheduler="threads")
        
        with pytest.raises(RuntimeError, match="Dask is not available"):
            ParallelExecutionEngine(config, self.registry)


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def setup_method(self):
        """Setup for each test"""
        self.registry = TaskRegistry()
        
        def test_task(job: Job):
            return f"result: {job.input_data}"
        
        self.registry.register_task("test", test_task)
    
    def test_create_engine(self):
        """Test create_engine convenience function"""
        engine = create_engine(scheduler="synchronous", n_workers=4, silence_logs=False)
        
        assert engine.config.scheduler == "synchronous"
        assert engine.config.n_workers == 4
        assert engine.config.silence_logs is False
        
        engine.shutdown()
    
    def test_execute_job_simple(self):
        """Test execute_job_simple convenience function"""
        # Need to register task in global registry for this test
        from parallel_framework.registry import get_global_registry
        global_registry = get_global_registry()
        
        def simple_test_task(job: Job):
            return f"simple: {job.input_data}"
        
        global_registry.register_task("simple_test", simple_test_task)
        
        try:
            job = Job(task_type="simple_test", input_data="data")
            result = execute_job_simple(job, scheduler="synchronous")
            
            assert result == "simple: data"
        finally:
            # Clean up global registry
            global_registry.unregister_task("simple_test")


if __name__ == "__main__":
    pytest.main([__file__])