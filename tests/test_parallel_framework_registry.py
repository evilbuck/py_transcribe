"""
Tests for parallel framework task registry.
"""

import pytest
from unittest.mock import Mock

from parallel_framework.job import Job
from parallel_framework.registry import TaskRegistry, TaskInfo, get_global_registry, register_task


class TestTaskInfo:
    """Test TaskInfo dataclass"""
    
    def test_task_info_creation_minimal(self):
        """Test TaskInfo creation with minimal parameters"""
        handler = lambda job: "result"
        info = TaskInfo(task_type="test", handler=handler)
        
        assert info.task_type == "test"
        assert info.handler == handler
        assert info.description == ""
        assert info.required_parameters == []
        assert info.optional_parameters == []
        assert info.resource_estimator is None
    
    def test_task_info_creation_full(self):
        """Test TaskInfo creation with all parameters"""
        handler = lambda job: "result"
        estimator = lambda job: {"memory_mb": 1024}
        
        info = TaskInfo(
            task_type="full_test",
            handler=handler,
            description="Test task",
            required_parameters=["param1", "param2"],
            optional_parameters=["opt1"],
            resource_estimator=estimator
        )
        
        assert info.task_type == "full_test"
        assert info.handler == handler
        assert info.description == "Test task"
        assert info.required_parameters == ["param1", "param2"]
        assert info.optional_parameters == ["opt1"]
        assert info.resource_estimator == estimator
    
    def test_task_info_post_init(self):
        """Test TaskInfo post_init default handling"""
        handler = lambda job: "result"
        info = TaskInfo(
            task_type="test",
            handler=handler,
            required_parameters=None,
            optional_parameters=None
        )
        
        assert info.required_parameters == []
        assert info.optional_parameters == []


class TestTaskRegistry:
    """Test TaskRegistry class"""
    
    def test_registry_creation(self):
        """Test creating a new registry"""
        registry = TaskRegistry()
        assert len(registry._tasks) == 0
        assert len(registry._middleware) == 0
        assert registry.list_task_types() == []
    
    def test_register_task_basic(self):
        """Test basic task registration"""
        registry = TaskRegistry()
        
        def test_handler(job: Job):
            return f"processed: {job.input_data}"
        
        registry.register_task("test_task", test_handler, description="Test task")
        
        assert registry.is_registered("test_task")
        assert "test_task" in registry.list_task_types()
        
        handler = registry.get_handler("test_task")
        assert handler == test_handler
        
        info = registry.get_task_info("test_task")
        assert info.task_type == "test_task"
        assert info.description == "Test task"
    
    def test_register_task_with_parameters(self):
        """Test task registration with parameter specifications"""
        registry = TaskRegistry()
        
        def handler_with_params(job: Job):
            return job.parameters
        
        registry.register_task(
            "param_task",
            handler_with_params,
            required_parameters=["input_file", "model"],
            optional_parameters=["timeout", "quality"]
        )
        
        info = registry.get_task_info("param_task")
        assert info.required_parameters == ["input_file", "model"]
        assert info.optional_parameters == ["timeout", "quality"]
    
    def test_register_task_with_resource_estimator(self):
        """Test task registration with resource estimator"""
        registry = TaskRegistry()
        
        def test_handler(job: Job):
            return "result"
        
        def resource_estimator(job: Job):
            return {"memory_mb": 2048, "cpu_cores": 4}
        
        registry.register_task(
            "resource_task",
            test_handler,
            resource_estimator=resource_estimator
        )
        
        info = registry.get_task_info("resource_task")
        assert info.resource_estimator == resource_estimator
    
    def test_register_task_validation(self):
        """Test task registration validation"""
        registry = TaskRegistry()
        
        # Empty task_type should raise ValueError
        with pytest.raises(ValueError, match="task_type cannot be empty"):
            registry.register_task("", lambda job: None)
        
        # Non-callable handler should raise ValueError
        with pytest.raises(ValueError, match="handler must be callable"):
            registry.register_task("test", "not_callable")
        
        # Duplicate registration should raise ValueError
        registry.register_task("duplicate", lambda job: None)
        with pytest.raises(ValueError, match="already registered"):
            registry.register_task("duplicate", lambda job: None)
    
    def test_handler_signature_validation(self):
        """Test handler signature validation"""
        registry = TaskRegistry()
        
        # Handler with no parameters should raise ValueError
        with pytest.raises(ValueError, match="exactly one parameter"):
            registry.register_task("no_params", lambda: None)
        
        # Handler with multiple parameters should raise ValueError
        with pytest.raises(ValueError, match="exactly one parameter"):
            registry.register_task("multi_params", lambda job, extra: None)
        
        # Valid handler should work
        registry.register_task("valid", lambda job: None)
        assert registry.is_registered("valid")
    
    def test_register_task_decorator(self):
        """Test task registration using decorator"""
        registry = TaskRegistry()
        
        @registry.register_task_decorator(
            "decorated_task",
            description="Decorated task",
            required_parameters=["input"]
        )
        def decorated_handler(job: Job):
            return f"decorated: {job.input_data}"
        
        assert registry.is_registered("decorated_task")
        
        info = registry.get_task_info("decorated_task")
        assert info.description == "Decorated task"
        assert info.required_parameters == ["input"]
        
        # Original function should still work
        test_job = Job(task_type="test", input_data="test_data")
        result = decorated_handler(test_job)
        assert result == "decorated: test_data"
    
    def test_unregister_task(self):
        """Test task unregistration"""
        registry = TaskRegistry()
        
        registry.register_task("removable", lambda job: None)
        assert registry.is_registered("removable")
        
        # Unregister existing task
        result = registry.unregister_task("removable")
        assert result is True
        assert not registry.is_registered("removable")
        
        # Unregister non-existing task
        result = registry.unregister_task("nonexistent")
        assert result is False
    
    def test_get_handler_not_registered(self):
        """Test getting handler for unregistered task"""
        registry = TaskRegistry()
        
        with pytest.raises(KeyError, match="not registered"):
            registry.get_handler("nonexistent")
    
    def test_get_task_info_not_registered(self):
        """Test getting task info for unregistered task"""
        registry = TaskRegistry()
        
        with pytest.raises(KeyError, match="not registered"):
            registry.get_task_info("nonexistent")
    
    def test_job_validation_valid(self):
        """Test job validation for valid job"""
        registry = TaskRegistry()
        
        registry.register_task(
            "validation_test",
            lambda job: None,
            required_parameters=["model", "input_file"]
        )
        
        job = Job(
            task_type="validation_test",
            input_data="data",
            parameters={"model": "base", "input_file": "test.mp3", "extra": "value"}
        )
        
        errors = registry.validate_job(job)
        assert errors == []
    
    def test_job_validation_missing_parameters(self):
        """Test job validation with missing required parameters"""
        registry = TaskRegistry()
        
        registry.register_task(
            "validation_test",
            lambda job: None,
            required_parameters=["model", "input_file", "quality"]
        )
        
        job = Job(
            task_type="validation_test",
            input_data="data",
            parameters={"model": "base"}  # Missing input_file and quality
        )
        
        errors = registry.validate_job(job)
        assert len(errors) == 2
        assert "Required parameter 'input_file' is missing" in errors
        assert "Required parameter 'quality' is missing" in errors
    
    def test_job_validation_unregistered_task(self):
        """Test job validation for unregistered task type"""
        registry = TaskRegistry()
        
        job = Job(task_type="unregistered", input_data="data")
        
        errors = registry.validate_job(job)
        assert len(errors) == 1
        assert "Task type 'unregistered' is not registered" in errors[0]
    
    def test_resource_estimation_no_estimator(self):
        """Test resource estimation without task-specific estimator"""
        registry = TaskRegistry()
        
        registry.register_task("no_estimator", lambda job: None)
        
        job = Job(
            task_type="no_estimator",
            input_data="data",
            memory_mb=1024,
            cpu_cores=2
        )
        
        resources = registry.estimate_resources(job)
        expected = {"memory_mb": 1024, "cpu_cores": 2, "gpu_memory_mb": None}
        assert resources == expected
    
    def test_resource_estimation_with_estimator(self):
        """Test resource estimation with task-specific estimator"""
        registry = TaskRegistry()
        
        def estimator(job: Job):
            return {"memory_mb": 2048, "cpu_cores": 4, "gpu_memory_mb": 1024}
        
        registry.register_task("with_estimator", lambda job: None, resource_estimator=estimator)
        
        job = Job(
            task_type="with_estimator",
            input_data="data",
            memory_mb=512  # This should be overridden
        )
        
        resources = registry.estimate_resources(job)
        expected = {"memory_mb": 2048, "cpu_cores": 4, "gpu_memory_mb": 1024}
        assert resources == expected
    
    def test_resource_estimation_unregistered(self):
        """Test resource estimation for unregistered task"""
        registry = TaskRegistry()
        
        job = Job(task_type="unregistered", input_data="data")
        
        resources = registry.estimate_resources(job)
        assert resources == {}
    
    def test_resource_estimation_estimator_error(self):
        """Test resource estimation when estimator raises error"""
        registry = TaskRegistry()
        
        def failing_estimator(job: Job):
            raise Exception("Estimator failed")
        
        registry.register_task("failing", lambda job: None, resource_estimator=failing_estimator)
        
        job = Job(
            task_type="failing",
            input_data="data",
            memory_mb=512
        )
        
        # Should fall back to job's estimates
        resources = registry.estimate_resources(job)
        expected = {"memory_mb": 512, "cpu_cores": None, "gpu_memory_mb": None}
        assert resources == expected
    
    def test_middleware_addition(self):
        """Test adding middleware"""
        registry = TaskRegistry()
        
        def test_middleware(job: Job, handler):
            result = handler(job)
            return f"middleware: {result}"
        
        registry.add_middleware(test_middleware)
        assert len(registry._middleware) == 1
        
        # Non-callable middleware should raise error
        with pytest.raises(ValueError, match="middleware must be callable"):
            registry.add_middleware("not_callable")
    
    def test_execute_task_basic(self):
        """Test basic task execution"""
        registry = TaskRegistry()
        
        def test_handler(job: Job):
            return f"processed: {job.input_data}"
        
        registry.register_task("execute_test", test_handler)
        
        job = Job(task_type="execute_test", input_data="test_data")
        
        result = registry.execute_task(job)
        assert result == "processed: test_data"
    
    def test_execute_task_with_middleware(self):
        """Test task execution with middleware"""
        registry = TaskRegistry()
        
        def test_handler(job: Job):
            return f"processed: {job.input_data}"
        
        def middleware1(job: Job, handler):
            result = handler(job)
            return f"mw1({result})"
        
        def middleware2(job: Job, handler):
            result = handler(job)
            return f"mw2({result})"
        
        registry.register_task("middleware_test", test_handler)
        registry.add_middleware(middleware1)
        registry.add_middleware(middleware2)
        
        job = Job(task_type="middleware_test", input_data="test")
        
        result = registry.execute_task(job)
        # Middleware wraps like an onion - last added becomes outermost
        assert result == "mw2(mw1(processed: test))"
    
    def test_execute_task_validation_error(self):
        """Test task execution with validation error"""
        registry = TaskRegistry()
        
        registry.register_task(
            "validation_error",
            lambda job: None,
            required_parameters=["required_param"]
        )
        
        job = Job(task_type="validation_error", input_data="data")  # Missing required_param
        
        with pytest.raises(ValueError, match="Job validation failed"):
            registry.execute_task(job)
    
    def test_execute_task_unregistered(self):
        """Test executing unregistered task"""
        registry = TaskRegistry()
        
        job = Job(task_type="unregistered", input_data="data")
        
        with pytest.raises(ValueError, match="Job validation failed"):
            registry.execute_task(job)
    
    def test_get_registry_info(self):
        """Test getting registry information"""
        registry = TaskRegistry()
        
        def handler1(job: Job):
            return "result1"
        
        def handler2(job: Job):
            return "result2"
        
        def estimator(job: Job):
            return {"memory_mb": 1024}
        
        registry.register_task(
            "task1",
            handler1,
            description="First task",
            required_parameters=["param1"],
            optional_parameters=["opt1"]
        )
        
        registry.register_task(
            "task2",
            handler2,
            description="Second task",
            resource_estimator=estimator
        )
        
        registry.add_middleware(lambda job, handler: handler(job))
        
        info = registry.get_registry_info()
        
        assert info['task_count'] == 2
        assert set(info['task_types']) == {"task1", "task2"}
        assert info['middleware_count'] == 1
        
        task1_info = info['tasks']['task1']
        assert task1_info['description'] == "First task"
        assert task1_info['required_parameters'] == ["param1"]
        assert task1_info['optional_parameters'] == ["opt1"]
        assert task1_info['has_resource_estimator'] is False
        
        task2_info = info['tasks']['task2']
        assert task2_info['has_resource_estimator'] is True


class TestGlobalRegistry:
    """Test global registry functions"""
    
    def test_get_global_registry(self):
        """Test getting global registry"""
        registry = get_global_registry()
        assert isinstance(registry, TaskRegistry)
        
        # Should return same instance
        registry2 = get_global_registry()
        assert registry is registry2
    
    def test_register_task_global(self):
        """Test registering task with global function"""
        def global_handler(job: Job):
            return "global_result"
        
        # Note: This modifies the global registry, which could affect other tests
        # In a real test suite, we'd want to reset the global state or use a fixture
        register_task("global_test", global_handler, description="Global test")
        
        global_registry = get_global_registry()
        assert global_registry.is_registered("global_test")
        
        # Clean up
        global_registry.unregister_task("global_test")


if __name__ == "__main__":
    pytest.main([__file__])