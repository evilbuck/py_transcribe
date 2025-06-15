"""
Advanced job state management and transitions for the parallel processing framework.
"""

import time
from typing import Dict, List, Optional, Set, Callable, Any
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .job import JobState, JobHandle


class StateTransitionError(Exception):
    """Raised when an invalid state transition is attempted"""
    pass


@dataclass
class StateTransition:
    """Represents a state transition with metadata"""
    from_state: JobState
    to_state: JobState
    timestamp: datetime
    message: Optional[str] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class StateValidator:
    """
    Validates job state transitions according to business rules.
    
    Ensures that jobs can only transition between valid states and
    provides detailed error messages for invalid transitions.
    """
    
    # Valid state transitions (from_state -> set of valid to_states)
    VALID_TRANSITIONS: Dict[JobState, Set[JobState]] = {
        JobState.PENDING: {JobState.QUEUED, JobState.CANCELLED},
        JobState.QUEUED: {JobState.RUNNING, JobState.CANCELLED},
        JobState.RUNNING: {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED, JobState.RETRYING},
        JobState.RETRYING: {JobState.RUNNING, JobState.FAILED, JobState.CANCELLED},
        JobState.COMPLETED: set(),  # Terminal state
        JobState.FAILED: {JobState.RETRYING},  # Can retry failed jobs
        JobState.CANCELLED: set(),  # Terminal state
    }
    
    @classmethod
    def is_valid_transition(cls, from_state: JobState, to_state: JobState) -> bool:
        """Check if a state transition is valid"""
        if from_state == to_state:
            return True  # Same state is always valid
        
        return to_state in cls.VALID_TRANSITIONS.get(from_state, set())
    
    @classmethod
    def get_valid_next_states(cls, current_state: JobState) -> Set[JobState]:
        """Get all valid next states for a given current state"""
        return cls.VALID_TRANSITIONS.get(current_state, set()).copy()
    
    @classmethod
    def validate_transition(cls, from_state: JobState, to_state: JobState) -> None:
        """
        Validate a state transition, raising an exception if invalid.
        
        Args:
            from_state: Current state
            to_state: Desired state
            
        Raises:
            StateTransitionError: If transition is invalid
        """
        if not cls.is_valid_transition(from_state, to_state):
            valid_states = cls.get_valid_next_states(from_state)
            raise StateTransitionError(
                f"Invalid state transition from {from_state} to {to_state}. "
                f"Valid transitions: {valid_states}"
            )
    
    @classmethod
    def get_transition_path(cls, from_state: JobState, to_state: JobState) -> Optional[List[JobState]]:
        """
        Find a valid path between two states, if one exists.
        
        Args:
            from_state: Starting state
            to_state: Target state
            
        Returns:
            List of states representing the path, or None if no path exists
        """
        if from_state == to_state:
            return [from_state]
        
        # Simple BFS to find shortest path
        from collections import deque
        
        queue = deque([(from_state, [from_state])])
        visited = {from_state}
        
        while queue:
            current_state, path = queue.popleft()
            
            for next_state in cls.VALID_TRANSITIONS.get(current_state, set()):
                if next_state == to_state:
                    return path + [next_state]
                
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, path + [next_state]))
        
        return None  # No valid path found


class JobStateManager:
    """
    Advanced state management for job handles with history tracking,
    validation, and event notifications.
    """
    
    def __init__(self, handle: JobHandle, validator: Optional[StateValidator] = None):
        self.handle = handle
        self.validator = validator or StateValidator()
        self.transition_history: List[StateTransition] = []
        self.state_listeners: Dict[JobState, List[Callable]] = {}
        self.transition_listeners: List[Callable] = []
        
        # Record initial state
        self._record_transition(None, handle.status.state, "Initial state")
    
    def transition_to(self, new_state: JobState, message: Optional[str] = None, 
                     error: Optional[str] = None, force: bool = False) -> bool:
        """
        Transition job to a new state with validation and history tracking.
        
        Args:
            new_state: Target state
            message: Optional message describing the transition
            error: Optional error message if transitioning due to error
            force: If True, skip validation (use with caution)
            
        Returns:
            True if transition was successful, False otherwise
        """
        current_state = self.handle.status.state
        
        if current_state == new_state:
            return True  # No transition needed
        
        # Validate transition unless forced
        if not force:
            try:
                self.validator.validate_transition(current_state, new_state)
            except StateTransitionError as e:
                return False
        
        # Perform transition
        self.handle.update_status(new_state, message=message, error=error)
        
        # Record transition
        self._record_transition(current_state, new_state, message, error)
        
        # Notify listeners
        self._notify_state_listeners(new_state)
        self._notify_transition_listeners(current_state, new_state)
        
        return True
    
    def can_transition_to(self, target_state: JobState) -> bool:
        """Check if job can transition to target state"""
        current_state = self.handle.status.state
        return self.validator.is_valid_transition(current_state, target_state)
    
    def get_valid_next_states(self) -> Set[JobState]:
        """Get all valid next states for current job state"""
        return self.validator.get_valid_next_states(self.handle.status.state)
    
    def get_transition_history(self) -> List[StateTransition]:
        """Get complete transition history"""
        return self.transition_history.copy()
    
    def get_state_duration(self, state: JobState) -> Optional[float]:
        """
        Get total time spent in a specific state.
        
        Args:
            state: State to calculate duration for
            
        Returns:
            Duration in seconds, or None if state was never entered
        """
        transitions = [t for t in self.transition_history if t.to_state == state]
        if not transitions:
            return None
        
        total_duration = 0.0
        
        for i, transition in enumerate(transitions):
            start_time = transition.timestamp
            
            # Find when we left this state
            end_time = None
            for j in range(len(self.transition_history) - 1, i, -1):
                if self.transition_history[j].from_state == state:
                    end_time = self.transition_history[j].timestamp
                    break
            
            if end_time is None and self.handle.status.state == state:
                # Still in this state
                end_time = datetime.now(timezone.utc)
            
            if end_time:
                duration = (end_time - start_time).total_seconds()
                total_duration += duration
        
        return total_duration
    
    def add_state_listener(self, state: JobState, listener: Callable[[JobHandle], None]) -> None:
        """
        Add a listener that will be called when job enters a specific state.
        
        Args:
            state: State to listen for
            listener: Function to call when state is entered
        """
        if state not in self.state_listeners:
            self.state_listeners[state] = []
        self.state_listeners[state].append(listener)
    
    def add_transition_listener(self, listener: Callable[[JobState, JobState, JobHandle], None]) -> None:
        """
        Add a listener that will be called on any state transition.
        
        Args:
            listener: Function to call on transitions (from_state, to_state, handle)
        """
        self.transition_listeners.append(listener)
    
    def remove_state_listener(self, state: JobState, listener: Callable) -> bool:
        """Remove a state listener"""
        if state in self.state_listeners and listener in self.state_listeners[state]:
            self.state_listeners[state].remove(listener)
            return True
        return False
    
    def remove_transition_listener(self, listener: Callable) -> bool:
        """Remove a transition listener"""
        if listener in self.transition_listeners:
            self.transition_listeners.remove(listener)
            return True
        return False
    
    def get_state_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics about job state transitions.
        
        Returns:
            Dictionary with state metrics and timing information
        """
        metrics = {
            'current_state': self.handle.status.state,
            'total_transitions': len(self.transition_history) - 1,  # Exclude initial state
            'states_visited': set(t.to_state for t in self.transition_history),
            'state_durations': {},
            'transition_count': {},
            'error_count': 0,
            'retry_count': 0
        }
        
        # Calculate state durations
        for state in JobState:
            duration = self.get_state_duration(state)
            if duration is not None:
                metrics['state_durations'][state.value] = duration
        
        # Count transitions
        for transition in self.transition_history[1:]:  # Skip initial state
            from_state = transition.from_state
            to_state = transition.to_state
            
            if from_state:
                transition_key = f"{from_state.value} -> {to_state.value}"
                metrics['transition_count'][transition_key] = metrics['transition_count'].get(transition_key, 0) + 1
            
            if transition.error:
                metrics['error_count'] += 1
            
            if to_state == JobState.RETRYING:
                metrics['retry_count'] += 1
        
        return metrics
    
    def _record_transition(self, from_state: Optional[JobState], to_state: JobState, 
                          message: Optional[str] = None, error: Optional[str] = None) -> None:
        """Record a state transition in history"""
        transition = StateTransition(
            from_state=from_state,
            to_state=to_state,
            timestamp=datetime.now(timezone.utc),
            message=message,
            error=error
        )
        self.transition_history.append(transition)
    
    def _notify_state_listeners(self, new_state: JobState) -> None:
        """Notify listeners when entering a new state"""
        if new_state in self.state_listeners:
            for listener in self.state_listeners[new_state]:
                try:
                    listener(self.handle)
                except Exception:
                    pass  # Don't let listener errors affect state transitions
    
    def _notify_transition_listeners(self, from_state: Optional[JobState], to_state: JobState) -> None:
        """Notify listeners of state transitions"""
        for listener in self.transition_listeners:
            try:
                listener(from_state, to_state, self.handle)
            except Exception:
                pass  # Don't let listener errors affect state transitions


class JobStateMonitor:
    """
    Monitor multiple jobs and provide aggregate state information.
    """
    
    def __init__(self):
        self.job_managers: Dict[str, JobStateManager] = {}
        self.global_listeners: List[Callable] = []
    
    def add_job(self, handle: JobHandle) -> JobStateManager:
        """
        Add a job to monitoring.
        
        Args:
            handle: Job handle to monitor
            
        Returns:
            JobStateManager for the job
        """
        manager = JobStateManager(handle)
        self.job_managers[handle.id] = manager
        
        # Add global transition listener
        manager.add_transition_listener(self._on_job_transition)
        
        return manager
    
    def remove_job(self, job_id: str) -> bool:
        """Remove job from monitoring"""
        if job_id in self.job_managers:
            del self.job_managers[job_id]
            return True
        return False
    
    def get_job_manager(self, job_id: str) -> Optional[JobStateManager]:
        """Get state manager for a specific job"""
        return self.job_managers.get(job_id)
    
    def get_jobs_by_state(self, state: JobState) -> List[JobHandle]:
        """Get all jobs currently in a specific state"""
        return [
            manager.handle for manager in self.job_managers.values()
            if manager.handle.status.state == state
        ]
    
    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """
        Get aggregate metrics across all monitored jobs.
        
        Returns:
            Dictionary with aggregate statistics
        """
        if not self.job_managers:
            return {}
        
        metrics = {
            'total_jobs': len(self.job_managers),
            'jobs_by_state': {},
            'average_state_durations': {},
            'total_transitions': 0,
            'total_errors': 0,
            'total_retries': 0,
            'completion_rate': 0.0
        }
        
        # Count jobs by state
        for state in JobState:
            jobs_in_state = len(self.get_jobs_by_state(state))
            if jobs_in_state > 0:
                metrics['jobs_by_state'][state.value] = jobs_in_state
        
        # Aggregate individual job metrics
        all_job_metrics = [manager.get_state_metrics() for manager in self.job_managers.values()]
        
        if all_job_metrics:
            metrics['total_transitions'] = sum(m['total_transitions'] for m in all_job_metrics)
            metrics['total_errors'] = sum(m['error_count'] for m in all_job_metrics)
            metrics['total_retries'] = sum(m['retry_count'] for m in all_job_metrics)
            
            # Calculate completion rate
            completed_jobs = metrics['jobs_by_state'].get('completed', 0)
            failed_jobs = metrics['jobs_by_state'].get('failed', 0)
            cancelled_jobs = metrics['jobs_by_state'].get('cancelled', 0)
            terminal_jobs = completed_jobs + failed_jobs + cancelled_jobs
            
            if terminal_jobs > 0:
                metrics['completion_rate'] = completed_jobs / terminal_jobs
        
        return metrics
    
    def add_global_listener(self, listener: Callable[[str, JobState, JobState], None]) -> None:
        """
        Add a listener for all job state transitions.
        
        Args:
            listener: Function called with (job_id, from_state, to_state)
        """
        self.global_listeners.append(listener)
    
    def remove_global_listener(self, listener: Callable) -> bool:
        """Remove a global listener"""
        if listener in self.global_listeners:
            self.global_listeners.remove(listener)
            return True
        return False
    
    def _on_job_transition(self, from_state: Optional[JobState], to_state: JobState, handle: JobHandle) -> None:
        """Handle job state transitions for global listeners"""
        for listener in self.global_listeners:
            try:
                listener(handle.id, from_state, to_state)
            except Exception:
                pass  # Don't let listener errors affect monitoring