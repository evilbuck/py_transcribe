"""
Tests for parallel framework state management.
"""

import pytest
import time
from datetime import datetime, timezone
from unittest.mock import Mock

from parallel_framework.job import Job, JobHandle, JobState
from parallel_framework.state_manager import (
    StateValidator, 
    JobStateManager, 
    JobStateMonitor, 
    StateTransitionError,
    StateTransition
)


class TestStateTransition:
    """Test StateTransition dataclass"""
    
    def test_state_transition_creation(self):
        """Test creating state transition"""
        now = datetime.now(timezone.utc)
        transition = StateTransition(
            from_state=JobState.PENDING,
            to_state=JobState.QUEUED,
            timestamp=now,
            message="Transition message"
        )
        
        assert transition.from_state == JobState.PENDING
        assert transition.to_state == JobState.QUEUED
        assert transition.timestamp == now
        assert transition.message == "Transition message"
        assert transition.error is None
    
    def test_state_transition_auto_timestamp(self):
        """Test automatic timestamp generation"""
        before = datetime.now(timezone.utc)
        transition = StateTransition(
            from_state=JobState.PENDING,
            to_state=JobState.QUEUED,
            timestamp=None
        )
        after = datetime.now(timezone.utc)
        
        assert before <= transition.timestamp <= after


class TestStateValidator:
    """Test StateValidator class"""
    
    def test_valid_transitions(self):
        """Test valid state transitions"""
        valid_cases = [
            (JobState.PENDING, JobState.QUEUED),
            (JobState.PENDING, JobState.CANCELLED),
            (JobState.QUEUED, JobState.RUNNING),
            (JobState.QUEUED, JobState.CANCELLED),
            (JobState.RUNNING, JobState.COMPLETED),
            (JobState.RUNNING, JobState.FAILED),
            (JobState.RUNNING, JobState.CANCELLED),
            (JobState.RUNNING, JobState.RETRYING),
            (JobState.RETRYING, JobState.RUNNING),
            (JobState.RETRYING, JobState.FAILED),
            (JobState.RETRYING, JobState.CANCELLED),
            (JobState.FAILED, JobState.RETRYING),
        ]
        
        for from_state, to_state in valid_cases:
            assert StateValidator.is_valid_transition(from_state, to_state)
    
    def test_invalid_transitions(self):
        """Test invalid state transitions"""
        invalid_cases = [
            (JobState.PENDING, JobState.RUNNING),  # Must go through QUEUED
            (JobState.PENDING, JobState.COMPLETED),  # Must go through QUEUED and RUNNING
            (JobState.COMPLETED, JobState.RUNNING),  # Terminal state
            (JobState.COMPLETED, JobState.FAILED),  # Terminal state
            (JobState.CANCELLED, JobState.RUNNING),  # Terminal state
            (JobState.QUEUED, JobState.COMPLETED),  # Must go through RUNNING
        ]
        
        for from_state, to_state in invalid_cases:
            assert not StateValidator.is_valid_transition(from_state, to_state)
    
    def test_same_state_transition(self):
        """Test that same state transitions are always valid"""
        for state in JobState:
            assert StateValidator.is_valid_transition(state, state)
    
    def test_get_valid_next_states(self):
        """Test getting valid next states"""
        # Test PENDING state
        pending_next = StateValidator.get_valid_next_states(JobState.PENDING)
        assert pending_next == {JobState.QUEUED, JobState.CANCELLED}
        
        # Test RUNNING state
        running_next = StateValidator.get_valid_next_states(JobState.RUNNING)
        assert running_next == {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED, JobState.RETRYING}
        
        # Test terminal state
        completed_next = StateValidator.get_valid_next_states(JobState.COMPLETED)
        assert completed_next == set()
    
    def test_validate_transition_success(self):
        """Test successful transition validation"""
        # Should not raise exception
        StateValidator.validate_transition(JobState.PENDING, JobState.QUEUED)
        StateValidator.validate_transition(JobState.RUNNING, JobState.COMPLETED)
    
    def test_validate_transition_failure(self):
        """Test failed transition validation"""
        with pytest.raises(StateTransitionError, match="Invalid state transition"):
            StateValidator.validate_transition(JobState.PENDING, JobState.RUNNING)
        
        with pytest.raises(StateTransitionError, match="Invalid state transition"):
            StateValidator.validate_transition(JobState.COMPLETED, JobState.RUNNING)
    
    def test_get_transition_path_direct(self):
        """Test finding direct transition path"""
        path = StateValidator.get_transition_path(JobState.PENDING, JobState.QUEUED)
        assert path == [JobState.PENDING, JobState.QUEUED]
    
    def test_get_transition_path_indirect(self):
        """Test finding indirect transition path"""
        path = StateValidator.get_transition_path(JobState.PENDING, JobState.RUNNING)
        assert path == [JobState.PENDING, JobState.QUEUED, JobState.RUNNING]
    
    def test_get_transition_path_same_state(self):
        """Test transition path for same state"""
        path = StateValidator.get_transition_path(JobState.RUNNING, JobState.RUNNING)
        assert path == [JobState.RUNNING]
    
    def test_get_transition_path_impossible(self):
        """Test transition path for impossible transitions"""
        path = StateValidator.get_transition_path(JobState.COMPLETED, JobState.RUNNING)
        assert path is None


class TestJobStateManager:
    """Test JobStateManager class"""
    
    def setup_method(self):
        """Setup for each test"""
        self.job = Job(task_type="test", input_data="data")
        self.handle = JobHandle(self.job)
        self.manager = JobStateManager(self.handle)
    
    def test_manager_creation(self):
        """Test creating job state manager"""
        assert self.manager.handle is self.handle
        assert isinstance(self.manager.validator, StateValidator)
        assert len(self.manager.transition_history) == 1  # Initial state recorded
        assert self.manager.transition_history[0].to_state == JobState.PENDING
    
    def test_valid_transition(self):
        """Test successful state transition"""
        success = self.manager.transition_to(JobState.QUEUED, "Job queued")
        assert success
        assert self.handle.status.state == JobState.QUEUED
        assert len(self.manager.transition_history) == 2
        
        last_transition = self.manager.transition_history[-1]
        assert last_transition.from_state == JobState.PENDING
        assert last_transition.to_state == JobState.QUEUED
        assert last_transition.message == "Job queued"
    
    def test_invalid_transition(self):
        """Test invalid state transition"""
        success = self.manager.transition_to(JobState.RUNNING, "Invalid transition")
        assert not success
        assert self.handle.status.state == JobState.PENDING  # Should remain unchanged
        assert len(self.manager.transition_history) == 1  # No new transition recorded
    
    def test_forced_transition(self):
        """Test forced state transition bypassing validation"""
        success = self.manager.transition_to(JobState.RUNNING, "Forced transition", force=True)
        assert success
        assert self.handle.status.state == JobState.RUNNING
        assert len(self.manager.transition_history) == 2
    
    def test_same_state_transition(self):
        """Test transitioning to same state"""
        success = self.manager.transition_to(JobState.PENDING, "Same state")
        assert success
        assert self.handle.status.state == JobState.PENDING
        # Should not record transition for same state
        assert len(self.manager.transition_history) == 1
    
    def test_can_transition_to(self):
        """Test checking if transition is possible"""
        assert self.manager.can_transition_to(JobState.QUEUED)
        assert self.manager.can_transition_to(JobState.CANCELLED)
        assert not self.manager.can_transition_to(JobState.RUNNING)
        assert not self.manager.can_transition_to(JobState.COMPLETED)
    
    def test_get_valid_next_states(self):
        """Test getting valid next states"""
        next_states = self.manager.get_valid_next_states()
        assert next_states == {JobState.QUEUED, JobState.CANCELLED}
    
    def test_transition_with_error(self):
        """Test transition with error message"""
        self.manager.transition_to(JobState.QUEUED)
        self.manager.transition_to(JobState.RUNNING)
        success = self.manager.transition_to(JobState.FAILED, "Task failed", error="Connection timeout")
        
        assert success
        assert self.handle.status.state == JobState.FAILED
        assert self.handle.status.error == "Connection timeout"
        
        last_transition = self.manager.transition_history[-1]
        assert last_transition.error == "Connection timeout"
    
    def test_get_state_duration(self):
        """Test calculating state duration"""
        # Initial state duration (still in PENDING)
        duration = self.manager.get_state_duration(JobState.PENDING)
        assert duration is not None
        assert duration > 0
        
        # State never entered
        duration = self.manager.get_state_duration(JobState.RUNNING)
        assert duration is None
        
        # Transition to new state and check duration
        time.sleep(0.01)  # Small delay
        self.manager.transition_to(JobState.QUEUED)
        
        pending_duration = self.manager.get_state_duration(JobState.PENDING)
        queued_duration = self.manager.get_state_duration(JobState.QUEUED)
        
        assert pending_duration is not None
        assert pending_duration > 0
        assert queued_duration is not None
        assert queued_duration >= 0
    
    def test_state_listeners(self):
        """Test state listeners"""
        queued_calls = []
        running_calls = []
        
        def on_queued(handle):
            queued_calls.append(handle)
        
        def on_running(handle):
            running_calls.append(handle)
        
        self.manager.add_state_listener(JobState.QUEUED, on_queued)
        self.manager.add_state_listener(JobState.RUNNING, on_running)
        
        # Transition to QUEUED
        self.manager.transition_to(JobState.QUEUED)
        assert len(queued_calls) == 1
        assert queued_calls[0] is self.handle
        assert len(running_calls) == 0
        
        # Transition to RUNNING
        self.manager.transition_to(JobState.RUNNING)
        assert len(queued_calls) == 1
        assert len(running_calls) == 1
        assert running_calls[0] is self.handle
    
    def test_transition_listeners(self):
        """Test transition listeners"""
        transitions = []
        
        def on_transition(from_state, to_state, handle):
            transitions.append((from_state, to_state, handle))
        
        self.manager.add_transition_listener(on_transition)
        
        self.manager.transition_to(JobState.QUEUED)
        self.manager.transition_to(JobState.RUNNING)
        
        assert len(transitions) == 2
        assert transitions[0] == (JobState.PENDING, JobState.QUEUED, self.handle)
        assert transitions[1] == (JobState.QUEUED, JobState.RUNNING, self.handle)
    
    def test_remove_listeners(self):
        """Test removing listeners"""
        listener = Mock()
        
        self.manager.add_state_listener(JobState.QUEUED, listener)
        assert self.manager.remove_state_listener(JobState.QUEUED, listener)
        assert not self.manager.remove_state_listener(JobState.QUEUED, listener)  # Already removed
        
        self.manager.add_transition_listener(listener)
        assert self.manager.remove_transition_listener(listener)
        assert not self.manager.remove_transition_listener(listener)  # Already removed
    
    def test_get_state_metrics(self):
        """Test getting state metrics"""
        # Initial metrics
        metrics = self.manager.get_state_metrics()
        assert metrics['current_state'] == JobState.PENDING
        assert metrics['total_transitions'] == 0  # Excludes initial state
        assert JobState.PENDING in metrics['states_visited']
        
        # Perform some transitions
        self.manager.transition_to(JobState.QUEUED)
        self.manager.transition_to(JobState.RUNNING)
        self.manager.transition_to(JobState.FAILED, error="Test error")
        self.manager.transition_to(JobState.RETRYING)
        
        metrics = self.manager.get_state_metrics()
        assert metrics['current_state'] == JobState.RETRYING
        assert metrics['total_transitions'] == 4
        assert metrics['error_count'] == 1
        assert metrics['retry_count'] == 1
        assert len(metrics['states_visited']) == 5  # PENDING, QUEUED, RUNNING, FAILED, RETRYING


class TestJobStateMonitor:
    """Test JobStateMonitor class"""
    
    def test_monitor_creation(self):
        """Test creating job state monitor"""
        monitor = JobStateMonitor()
        assert len(monitor.job_managers) == 0
        assert len(monitor.global_listeners) == 0
    
    def test_add_remove_job(self):
        """Test adding and removing jobs"""
        monitor = JobStateMonitor()
        
        job = Job(task_type="test", input_data="data")
        handle = JobHandle(job)
        
        manager = monitor.add_job(handle)
        assert isinstance(manager, JobStateManager)
        assert handle.id in monitor.job_managers
        assert monitor.job_managers[handle.id] is manager
        
        success = monitor.remove_job(handle.id)
        assert success
        assert handle.id not in monitor.job_managers
        
        # Remove non-existent job
        success = monitor.remove_job("nonexistent")
        assert not success
    
    def test_get_job_manager(self):
        """Test getting job manager"""
        monitor = JobStateMonitor()
        
        job = Job(task_type="test", input_data="data")
        handle = JobHandle(job)
        
        manager = monitor.add_job(handle)
        
        retrieved_manager = monitor.get_job_manager(handle.id)
        assert retrieved_manager is manager
        
        not_found = monitor.get_job_manager("nonexistent")
        assert not_found is None
    
    def test_get_jobs_by_state(self):
        """Test getting jobs by state"""
        monitor = JobStateMonitor()
        
        # Create multiple jobs in different states
        jobs = []
        handles = []
        managers = []
        
        for i in range(5):
            job = Job(task_type="test", input_data=f"data_{i}")
            handle = JobHandle(job)
            manager = monitor.add_job(handle)
            
            jobs.append(job)
            handles.append(handle)
            managers.append(manager)
        
        # Transition jobs to different states
        managers[0].transition_to(JobState.QUEUED)
        managers[1].transition_to(JobState.QUEUED)
        managers[1].transition_to(JobState.RUNNING)
        managers[2].transition_to(JobState.QUEUED)
        managers[2].transition_to(JobState.RUNNING)
        managers[2].transition_to(JobState.COMPLETED)
        managers[3].transition_to(JobState.QUEUED)
        managers[3].transition_to(JobState.RUNNING)
        managers[3].transition_to(JobState.FAILED)
        # managers[4] stays in PENDING
        
        # Test getting jobs by state
        pending_jobs = monitor.get_jobs_by_state(JobState.PENDING)
        assert len(pending_jobs) == 1
        assert handles[4] in pending_jobs
        
        queued_jobs = monitor.get_jobs_by_state(JobState.QUEUED)
        assert len(queued_jobs) == 1
        assert handles[0] in queued_jobs
        
        running_jobs = monitor.get_jobs_by_state(JobState.RUNNING)
        assert len(running_jobs) == 1
        assert handles[1] in running_jobs
        
        completed_jobs = monitor.get_jobs_by_state(JobState.COMPLETED)
        assert len(completed_jobs) == 1
        assert handles[2] in completed_jobs
        
        failed_jobs = monitor.get_jobs_by_state(JobState.FAILED)
        assert len(failed_jobs) == 1
        assert handles[3] in failed_jobs
    
    def test_aggregate_metrics(self):
        """Test getting aggregate metrics"""
        monitor = JobStateMonitor()
        
        # Empty monitor
        metrics = monitor.get_aggregate_metrics()
        assert metrics == {}
        
        # Add jobs and test metrics
        handles = []
        managers = []
        
        for i in range(3):
            job = Job(task_type="test", input_data=f"data_{i}")
            handle = JobHandle(job)
            manager = monitor.add_job(handle)
            handles.append(handle)
            managers.append(manager)
        
        # Transition jobs
        managers[0].transition_to(JobState.QUEUED)
        managers[0].transition_to(JobState.RUNNING)
        managers[0].transition_to(JobState.COMPLETED)
        
        managers[1].transition_to(JobState.QUEUED)
        managers[1].transition_to(JobState.RUNNING)
        managers[1].transition_to(JobState.FAILED, error="Test error")
        
        managers[2].transition_to(JobState.QUEUED)
        managers[2].transition_to(JobState.CANCELLED)
        
        metrics = monitor.get_aggregate_metrics()
        
        assert metrics['total_jobs'] == 3
        assert metrics['jobs_by_state']['completed'] == 1
        assert metrics['jobs_by_state']['failed'] == 1
        assert metrics['jobs_by_state']['cancelled'] == 1
        assert metrics['total_transitions'] == 8  # 3 + 3 + 2 transitions (including initial states)
        assert metrics['total_errors'] == 1
        assert metrics['completion_rate'] == 1/3  # 1 completed out of 3 terminal jobs
    
    def test_global_listeners(self):
        """Test global listeners"""
        monitor = JobStateMonitor()
        transitions = []
        
        def global_listener(job_id, from_state, to_state):
            transitions.append((job_id, from_state, to_state))
        
        monitor.add_global_listener(global_listener)
        
        # Add job and perform transitions
        job = Job(task_type="test", input_data="data")
        handle = JobHandle(job)
        manager = monitor.add_job(handle)
        
        manager.transition_to(JobState.QUEUED)
        manager.transition_to(JobState.RUNNING)
        
        # Check transitions were recorded
        assert len(transitions) == 2
        assert transitions[0] == (handle.id, JobState.PENDING, JobState.QUEUED)
        assert transitions[1] == (handle.id, JobState.QUEUED, JobState.RUNNING)
        
        # Remove listener
        success = monitor.remove_global_listener(global_listener)
        assert success
        
        # Further transitions should not be recorded
        manager.transition_to(JobState.COMPLETED)
        assert len(transitions) == 2


if __name__ == "__main__":
    pytest.main([__file__])