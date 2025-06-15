"""
Tests for parallel framework job classes.
"""

import pytest
import time
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock

from parallel_framework.job import Job, JobHandle, JobState, JobStatus


class TestJobStatus:
    """Test JobStatus class"""
    
    def test_job_status_creation(self):
        """Test basic JobStatus creation"""
        status = JobStatus(state=JobState.PENDING)
        assert status.state == JobState.PENDING
        assert status.progress == 0.0
        assert status.message == ""
        assert status.started_at is None
        assert status.completed_at is None
        assert status.error is None
        assert status.retry_count == 0
    
    def test_is_terminal(self):
        """Test terminal state detection"""
        terminal_states = [JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED]
        non_terminal_states = [JobState.PENDING, JobState.QUEUED, JobState.RUNNING, JobState.RETRYING]
        
        for state in terminal_states:
            status = JobStatus(state=state)
            assert status.is_terminal
        
        for state in non_terminal_states:
            status = JobStatus(state=state)
            assert not status.is_terminal
    
    def test_is_active(self):
        """Test active state detection"""
        active_states = [JobState.QUEUED, JobState.RUNNING, JobState.RETRYING]
        inactive_states = [JobState.PENDING, JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED]
        
        for state in active_states:
            status = JobStatus(state=state)
            assert status.is_active
        
        for state in inactive_states:
            status = JobStatus(state=state)
            assert not status.is_active
    
    def test_duration_calculation(self):
        """Test duration calculation"""
        status = JobStatus(state=JobState.RUNNING)
        assert status.duration is None
        
        # Set start time
        start_time = datetime.now(timezone.utc)
        status.started_at = start_time
        assert status.duration is None  # No end time yet
        
        # Set end time
        end_time = start_time.replace(second=start_time.second + 10)  # 10 seconds later
        status.completed_at = end_time
        
        duration = status.duration
        assert duration is not None
        assert abs(duration - 10.0) < 0.1  # Should be approximately 10 seconds


class TestJob:
    """Test Job class"""
    
    def test_job_creation_minimal(self):
        """Test creating a job with minimal parameters"""
        job = Job(
            task_type="test_task",
            input_data={"test": "data"}
        )
        
        assert job.task_type == "test_task"
        assert job.input_data == {"test": "data"}
        assert job.parameters == {}
        assert job.priority == 0
        assert job.timeout is None
        assert job.max_retries == 3
        assert job.id is not None
        assert len(job.id) > 0
        assert job.created_at is not None
        assert job.tags == {}
    
    def test_job_creation_full(self):
        """Test creating a job with all parameters"""
        job = Job(
            task_type="full_task",
            input_data="test_data",
            parameters={"param1": "value1"},
            id="custom_id",
            priority=10,
            timeout=3600,
            max_retries=5,
            memory_mb=1024,
            cpu_cores=2,
            gpu_memory_mb=2048,
            tags={"env": "test"}
        )
        
        assert job.task_type == "full_task"
        assert job.input_data == "test_data"
        assert job.parameters == {"param1": "value1"}
        assert job.id == "custom_id"
        assert job.priority == 10
        assert job.timeout == 3600
        assert job.max_retries == 5
        assert job.memory_mb == 1024
        assert job.cpu_cores == 2
        assert job.gpu_memory_mb == 2048
        assert job.tags == {"env": "test"}
    
    def test_job_validation(self):
        """Test job parameter validation"""
        # Empty task_type should raise ValueError
        with pytest.raises(ValueError, match="task_type cannot be empty"):
            Job(task_type="", input_data="data")
        
        # Negative priority should raise ValueError
        with pytest.raises(ValueError, match="priority must be non-negative"):
            Job(task_type="test", input_data="data", priority=-1)
        
        # Zero or negative timeout should raise ValueError
        with pytest.raises(ValueError, match="timeout must be positive"):
            Job(task_type="test", input_data="data", timeout=0)
        
        with pytest.raises(ValueError, match="timeout must be positive"):
            Job(task_type="test", input_data="data", timeout=-10)
        
        # Negative max_retries should raise ValueError
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            Job(task_type="test", input_data="data", max_retries=-1)
    
    def test_fluent_interface(self):
        """Test fluent interface methods"""
        job = Job(task_type="test", input_data="data")
        
        # Test chaining
        result = job.with_tag("env", "test").with_priority(5).with_timeout(1800)
        
        assert result is job  # Should return self
        assert job.tags == {"env": "test"}
        assert job.priority == 5
        assert job.timeout == 1800
    
    def test_estimate_resources(self):
        """Test resource estimation"""
        job = Job(
            task_type="test",
            input_data="data",
            memory_mb=512,
            cpu_cores=4,
            gpu_memory_mb=1024
        )
        
        resources = job.estimate_resources()
        expected = {
            'memory_mb': 512,
            'cpu_cores': 4,
            'gpu_memory_mb': 1024
        }
        assert resources == expected
        
        # Test with None values
        job_minimal = Job(task_type="test", input_data="data")
        resources_minimal = job_minimal.estimate_resources()
        expected_minimal = {
            'memory_mb': None,
            'cpu_cores': None,
            'gpu_memory_mb': None
        }
        assert resources_minimal == expected_minimal


class TestJobHandle:
    """Test JobHandle class"""
    
    def test_job_handle_creation(self):
        """Test basic JobHandle creation"""
        job = Job(task_type="test", input_data="data")
        handle = JobHandle(job)
        
        assert handle.job is job
        assert handle.id == job.id
        assert handle.status.state == JobState.PENDING
        assert handle.status.progress == 0.0
    
    def test_status_updates(self):
        """Test status update functionality"""
        job = Job(task_type="test", input_data="data")
        handle = JobHandle(job)
        
        # Test basic status update
        handle.update_status(JobState.QUEUED, progress=0.1, message="Queued for processing")
        
        assert handle.status.state == JobState.QUEUED
        assert handle.status.progress == 0.1
        assert handle.status.message == "Queued for processing"
        
        # Test running state (should set started_at)
        handle.update_status(JobState.RUNNING, progress=0.5)
        assert handle.status.state == JobState.RUNNING
        assert handle.status.progress == 0.5
        assert handle.status.started_at is not None
        
        # Test completion (should set completed_at and progress to 1.0)
        handle.update_status(JobState.COMPLETED)
        assert handle.status.state == JobState.COMPLETED
        assert handle.status.progress == 1.0
        assert handle.status.completed_at is not None
    
    def test_status_update_with_error(self):
        """Test status update with error"""
        job = Job(task_type="test", input_data="data")
        handle = JobHandle(job)
        
        handle.update_status(JobState.FAILED, error="Something went wrong")
        
        assert handle.status.state == JobState.FAILED
        assert handle.status.error == "Something went wrong"
        assert handle.status.completed_at is not None
    
    def test_callbacks(self):
        """Test status change callbacks"""
        job = Job(task_type="test", input_data="data")
        handle = JobHandle(job)
        
        callback_calls = []
        
        def test_callback(status):
            callback_calls.append(status.state)
        
        handle.add_callback(test_callback)
        
        # Trigger status changes
        handle.update_status(JobState.QUEUED)
        handle.update_status(JobState.RUNNING)
        handle.update_status(JobState.COMPLETED)
        
        assert callback_calls == [JobState.QUEUED, JobState.RUNNING, JobState.COMPLETED]
    
    def test_callback_error_handling(self):
        """Test that callback errors don't affect job execution"""
        job = Job(task_type="test", input_data="data")
        handle = JobHandle(job)
        
        def failing_callback(status):
            raise Exception("Callback failed")
        
        handle.add_callback(failing_callback)
        
        # This should not raise an exception
        handle.update_status(JobState.RUNNING)
        assert handle.status.state == JobState.RUNNING
    
    def test_progress_bounds(self):
        """Test that progress is bounded between 0 and 1"""
        job = Job(task_type="test", input_data="data")
        handle = JobHandle(job)
        
        # Test negative progress
        handle.update_status(JobState.RUNNING, progress=-0.5)
        assert handle.status.progress == 0.0
        
        # Test progress > 1
        handle.update_status(JobState.RUNNING, progress=1.5)
        assert handle.status.progress == 1.0
    
    def test_is_done(self):
        """Test is_done method"""
        job = Job(task_type="test", input_data="data")
        handle = JobHandle(job)
        
        assert not handle.is_done()
        
        handle.update_status(JobState.RUNNING)
        assert not handle.is_done()
        
        handle.update_status(JobState.COMPLETED)
        assert handle.is_done()
        
        # Test other terminal states
        handle.update_status(JobState.FAILED)
        assert handle.is_done()
        
        handle.update_status(JobState.CANCELLED)
        assert handle.is_done()
    
    def test_is_running(self):
        """Test is_running method"""
        job = Job(task_type="test", input_data="data")
        handle = JobHandle(job)
        
        assert not handle.is_running()
        
        handle.update_status(JobState.QUEUED)
        assert not handle.is_running()
        
        handle.update_status(JobState.RUNNING)
        assert handle.is_running()
        
        handle.update_status(JobState.COMPLETED)
        assert not handle.is_running()
    
    def test_cancel_without_future(self):
        """Test cancelling a job without a future"""
        job = Job(task_type="test", input_data="data")
        handle = JobHandle(job)
        
        result = handle.cancel()
        assert result is True
        assert handle.status.state == JobState.CANCELLED
        
        # Can't cancel a cancelled job
        result2 = handle.cancel()
        assert result2 is False
    
    def test_cancel_with_future(self):
        """Test cancelling a job with a future"""
        job = Job(task_type="test", input_data="data")
        mock_future = Mock()
        mock_future.cancel.return_value = True
        
        handle = JobHandle(job, future=mock_future)
        
        result = handle.cancel()
        assert result is True
        assert handle.status.state == JobState.CANCELLED
        mock_future.cancel.assert_called_once()
    
    def test_get_result_not_complete(self):
        """Test getting result when job is not complete"""
        job = Job(task_type="test", input_data="data")
        handle = JobHandle(job)
        
        with pytest.raises(RuntimeError, match="is not complete"):
            handle.get_result()
    
    def test_get_result_completed(self):
        """Test getting result when job is completed"""
        job = Job(task_type="test", input_data="data")
        handle = JobHandle(job)
        
        # Simulate completion
        handle._result = "test_result"
        handle.update_status(JobState.COMPLETED)
        
        result = handle.get_result()
        assert result == "test_result"
    
    def test_get_result_failed(self):
        """Test getting result when job failed"""
        job = Job(task_type="test", input_data="data")
        handle = JobHandle(job)
        
        handle.update_status(JobState.FAILED, error="Test error")
        
        with pytest.raises(RuntimeError, match="failed: Test error"):
            handle.get_result()
    
    def test_get_result_cancelled(self):
        """Test getting result when job was cancelled"""
        job = Job(task_type="test", input_data="data")
        handle = JobHandle(job)
        
        handle.update_status(JobState.CANCELLED)
        
        with pytest.raises(RuntimeError, match="was cancelled"):
            handle.get_result()


if __name__ == "__main__":
    pytest.main([__file__])