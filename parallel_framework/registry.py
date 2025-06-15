"""
Task registry system for managing different types of processing tasks.
"""

import inspect
from typing import Dict, Callable, Any, List, Optional, Type
from dataclasses import dataclass
from .job import Job


@dataclass
class TaskInfo:
    """Information about a registered task"""
    task_type: str
    handler: Callable
    description: str = ""
    required_parameters: List[str] = None
    optional_parameters: List[str] = None
    resource_estimator: Optional[Callable[[Job], Dict[str, Any]]] = None
    
    def __post_init__(self):
        if self.required_parameters is None:
            self.required_parameters = []
        if self.optional_parameters is None:
            self.optional_parameters = []


class TaskRegistry:
    """
    Registry for different types of processing tasks.
    
    Allows registration of task handlers and provides a plugin-like
    system for extending the framework with new task types.
    """
    
    def __init__(self):
        self._tasks: Dict[str, TaskInfo] = {}
        self._middleware: List[Callable] = []
    
    def register_task(self, task_type: str, handler: Callable, 
                     description: str = "", 
                     required_parameters: List[str] = None,
                     optional_parameters: List[str] = None,
                     resource_estimator: Callable[[Job], Dict[str, Any]] = None) -> None:
        """
        Register a task handler.
        
        Args:
            task_type: Unique identifier for the task type
            handler: Function that processes the task
            description: Human-readable description of the task
            required_parameters: List of required parameter names
            optional_parameters: List of optional parameter names
            resource_estimator: Function to estimate resource requirements
        """
        if not task_type:
            raise ValueError("task_type cannot be empty")
        
        if not callable(handler):
            raise ValueError("handler must be callable")
        
        if task_type in self._tasks:
            raise ValueError(f"Task type '{task_type}' is already registered")
        
        # Validate handler signature
        self._validate_handler_signature(handler)
        
        task_info = TaskInfo(
            task_type=task_type,
            handler=handler,
            description=description,
            required_parameters=required_parameters or [],
            optional_parameters=optional_parameters or [],
            resource_estimator=resource_estimator
        )
        
        self._tasks[task_type] = task_info
    
    def register_task_decorator(self, task_type: str, 
                               description: str = "",
                               required_parameters: List[str] = None,
                               optional_parameters: List[str] = None,
                               resource_estimator: Callable[[Job], Dict[str, Any]] = None):
        """
        Decorator for registering task handlers.
        
        Usage:
            @registry.register_task_decorator("my_task")
            def my_handler(job: Job) -> Any:
                return process_data(job.input_data)
        """
        def decorator(handler: Callable) -> Callable:
            self.register_task(
                task_type, handler, description, 
                required_parameters, optional_parameters, resource_estimator
            )
            return handler
        return decorator
    
    def unregister_task(self, task_type: str) -> bool:
        """
        Unregister a task handler.
        
        Args:
            task_type: Task type to unregister
            
        Returns:
            True if task was unregistered, False if it wasn't registered
        """
        if task_type in self._tasks:
            del self._tasks[task_type]
            return True
        return False
    
    def get_handler(self, task_type: str) -> Callable:
        """
        Get the handler for a specific task type.
        
        Args:
            task_type: Task type to get handler for
            
        Returns:
            Handler function
            
        Raises:
            KeyError: If task type is not registered
        """
        if task_type not in self._tasks:
            raise KeyError(f"Task type '{task_type}' is not registered")
        
        return self._tasks[task_type].handler
    
    def get_task_info(self, task_type: str) -> TaskInfo:
        """
        Get complete information about a task type.
        
        Args:
            task_type: Task type to get info for
            
        Returns:
            TaskInfo object with handler and metadata
            
        Raises:
            KeyError: If task type is not registered
        """
        if task_type not in self._tasks:
            raise KeyError(f"Task type '{task_type}' is not registered")
        
        return self._tasks[task_type]
    
    def list_task_types(self) -> List[str]:
        """
        Get list of all registered task types.
        
        Returns:
            List of registered task type names
        """
        return list(self._tasks.keys())
    
    def is_registered(self, task_type: str) -> bool:
        """
        Check if a task type is registered.
        
        Args:
            task_type: Task type to check
            
        Returns:
            True if registered, False otherwise
        """
        return task_type in self._tasks
    
    def validate_job(self, job: Job) -> List[str]:
        """
        Validate that a job has all required parameters for its task type.
        
        Args:
            job: Job to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if not self.is_registered(job.task_type):
            errors.append(f"Task type '{job.task_type}' is not registered")
            return errors
        
        task_info = self._tasks[job.task_type]
        
        # Check required parameters
        for param in task_info.required_parameters:
            if param not in job.parameters:
                errors.append(f"Required parameter '{param}' is missing")
        
        return errors
    
    def estimate_resources(self, job: Job) -> Dict[str, Any]:
        """
        Estimate resource requirements for a job.
        
        Args:
            job: Job to estimate resources for
            
        Returns:
            Dictionary with resource estimates
        """
        if not self.is_registered(job.task_type):
            return {}
        
        task_info = self._tasks[job.task_type]
        
        # Use job's built-in estimates first
        resources = job.estimate_resources()
        
        # Override with task-specific estimator if available
        if task_info.resource_estimator:
            try:
                task_resources = task_info.resource_estimator(job)
                resources.update(task_resources)
            except Exception:
                pass  # Fall back to job's estimates
        
        return resources
    
    def add_middleware(self, middleware: Callable) -> None:
        """
        Add middleware that wraps all task handlers.
        
        Middleware function should accept (job, handler) and return result.
        """
        if not callable(middleware):
            raise ValueError("middleware must be callable")
        
        self._middleware.append(middleware)
    
    def execute_task(self, job: Job) -> Any:
        """
        Execute a task with middleware support.
        
        Args:
            job: Job to execute
            
        Returns:
            Task result
            
        Raises:
            KeyError: If task type is not registered
            ValueError: If job validation fails
        """
        # Validate job
        errors = self.validate_job(job)
        if errors:
            raise ValueError(f"Job validation failed: {'; '.join(errors)}")
        
        handler = self.get_handler(job.task_type)
        
        # Apply middleware
        def execute_with_middleware(current_job: Job, current_handler: Callable) -> Any:
            if not self._middleware:
                return current_handler(current_job)
            
            # Chain middleware in correct order
            def wrapped_handler(j: Job) -> Any:
                return current_handler(j)
            
            # Apply middleware in the order they were added
            for middleware in self._middleware:
                prev_handler = wrapped_handler
                wrapped_handler = lambda j, mw=middleware, ph=prev_handler: mw(j, ph)
            
            return wrapped_handler(current_job)
        
        return execute_with_middleware(job, handler)
    
    def get_registry_info(self) -> Dict[str, Any]:
        """
        Get information about the registry state.
        
        Returns:
            Dictionary with registry statistics and task information
        """
        return {
            'task_count': len(self._tasks),
            'task_types': list(self._tasks.keys()),
            'middleware_count': len(self._middleware),
            'tasks': {
                task_type: {
                    'description': info.description,
                    'required_parameters': info.required_parameters,
                    'optional_parameters': info.optional_parameters,
                    'has_resource_estimator': info.resource_estimator is not None
                }
                for task_type, info in self._tasks.items()
            }
        }
    
    def _validate_handler_signature(self, handler: Callable) -> None:
        """
        Validate that handler has correct signature.
        
        Handler should accept a Job parameter and return Any.
        """
        try:
            sig = inspect.signature(handler)
            params = list(sig.parameters.values())
            
            if len(params) != 1:
                raise ValueError(f"Handler must accept exactly one parameter (Job), got {len(params)}")
            
            # Check parameter annotation if present
            param = params[0]
            if param.annotation != inspect.Parameter.empty:
                if not (param.annotation == Job or 
                       (hasattr(param.annotation, '__origin__') and 
                        getattr(param.annotation, '__args__', [None])[0] == Job)):
                    # Allow for cases where annotation might be a Union or other type
                    pass  # We'll be lenient about type annotations
                    
        except Exception as e:
            raise ValueError(f"Failed to validate handler signature: {e}")


# Global registry instance
_global_registry = TaskRegistry()

def get_global_registry() -> TaskRegistry:
    """Get the global task registry instance"""
    return _global_registry

def register_task(task_type: str, handler: Callable, **kwargs) -> None:
    """Convenience function to register a task with the global registry"""
    _global_registry.register_task(task_type, handler, **kwargs)