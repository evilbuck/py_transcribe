"""
Tests for parallel framework resource pool management.
"""

import pytest
import time
import threading
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock

from parallel_framework.resource_pool import (
    ResourceType, ResourceSpec, ResourceAllocation, ResourcePool,
    SystemResourceMonitor, ResourceError, InsufficientResourcesError,
    ResourceAllocationError
)


class TestResourceType:
    """Test ResourceType enum"""
    
    def test_resource_types(self):
        """Test all resource types are available"""
        assert ResourceType.CPU.value == "cpu"
        assert ResourceType.MEMORY.value == "memory"
        assert ResourceType.GPU.value == "gpu"
        assert ResourceType.DISK.value == "disk"
        assert ResourceType.NETWORK.value == "network"
        assert ResourceType.CUSTOM.value == "custom"


class TestResourceSpec:
    """Test ResourceSpec dataclass"""
    
    def test_resource_spec_creation(self):
        """Test creating resource specification"""
        spec = ResourceSpec(
            resource_type=ResourceType.CPU,
            amount=4,
            unit="cores",
            metadata={"priority": "high"}
        )
        
        assert spec.resource_type == ResourceType.CPU
        assert spec.amount == 4
        assert spec.unit == "cores"
        assert spec.metadata == {"priority": "high"}
    
    def test_resource_spec_defaults(self):
        """Test resource spec with default metadata"""
        spec = ResourceSpec(ResourceType.MEMORY, 1024, "MB")
        
        assert spec.metadata == {}
    
    def test_resource_spec_validation(self):
        """Test resource spec validation"""
        # Negative amount
        with pytest.raises(ValueError, match="Resource amount must be non-negative"):
            ResourceSpec(ResourceType.CPU, -1, "cores")
        
        # Invalid unit for CPU
        with pytest.raises(ValueError, match="Invalid unit 'invalid' for cpu"):
            ResourceSpec(ResourceType.CPU, 4, "invalid")
        
        # Invalid unit for memory
        with pytest.raises(ValueError, match="Invalid unit 'cores' for memory"):
            ResourceSpec(ResourceType.MEMORY, 1024, "cores")
    
    def test_valid_units(self):
        """Test valid units for different resource types"""
        # CPU
        ResourceSpec(ResourceType.CPU, 4, "cores")
        ResourceSpec(ResourceType.CPU, 50, "percent")
        
        # Memory
        ResourceSpec(ResourceType.MEMORY, 1024, "bytes")
        ResourceSpec(ResourceType.MEMORY, 1024, "MB")
        ResourceSpec(ResourceType.MEMORY, 1, "GB")
        
        # GPU
        ResourceSpec(ResourceType.GPU, 1, "devices")
        ResourceSpec(ResourceType.GPU, 1024, "memory_mb")
        
        # Custom resource (no validation)
        ResourceSpec(ResourceType.CUSTOM, 100, "custom_unit")


class TestResourceAllocation:
    """Test ResourceAllocation dataclass"""
    
    def test_allocation_creation(self):
        """Test creating resource allocation"""
        spec = ResourceSpec(ResourceType.CPU, 2, "cores")
        allocation = ResourceAllocation(
            resource_spec=spec,
            job_id="test_job",
            metadata={"test": "value"}
        )
        
        assert allocation.resource_spec == spec
        assert allocation.job_id == "test_job"
        assert allocation.metadata == {"test": "value"}
        assert allocation.id is not None
        assert allocation.allocated_at is not None
        assert allocation.expires_at is None
    
    def test_allocation_expiration(self):
        """Test allocation expiration"""
        spec = ResourceSpec(ResourceType.CPU, 2, "cores")
        
        # Non-expiring allocation
        allocation = ResourceAllocation(resource_spec=spec)
        assert not allocation.is_expired
        
        # Expired allocation
        past_time = datetime.now(timezone.utc) - timedelta(seconds=1)
        allocation.expires_at = past_time
        assert allocation.is_expired
        
        # Future expiration
        future_time = datetime.now(timezone.utc) + timedelta(seconds=60)
        allocation.expires_at = future_time
        assert not allocation.is_expired


class TestSystemResourceMonitor:
    """Test SystemResourceMonitor class"""
    
    def test_monitor_creation(self):
        """Test creating system resource monitor"""
        monitor = SystemResourceMonitor(update_interval=0.5)
        assert monitor.update_interval == 0.5
        assert not monitor._running
    
    def test_monitor_lifecycle(self):
        """Test monitor start/stop lifecycle"""
        monitor = SystemResourceMonitor(update_interval=0.1)
        
        # Start monitoring
        monitor.start()
        assert monitor._running
        assert monitor._thread is not None
        
        time.sleep(0.2)  # Let it run briefly
        
        # Stop monitoring
        monitor.stop()
        assert not monitor._running
    
    def test_get_system_info(self):
        """Test getting system information"""
        monitor = SystemResourceMonitor()
        info = monitor.get_system_info()
        
        assert isinstance(info, dict)
        assert 'cpu_count' in info
        assert 'memory' in info
        assert 'disk' in info
        assert info['cpu_count'] > 0
    
    def test_get_available_resources(self):
        """Test getting available resources"""
        monitor = SystemResourceMonitor()
        available = monitor.get_available_resources()
        
        assert ResourceType.CPU in available
        assert ResourceType.MEMORY in available
        assert ResourceType.DISK in available
        
        cpu_spec = available[ResourceType.CPU]
        assert cpu_spec.resource_type == ResourceType.CPU
        assert cpu_spec.unit == "percent"
        assert 0 <= cpu_spec.amount <= 100


class TestResourcePool:
    """Test ResourcePool class"""
    
    def setup_method(self):
        """Setup for each test"""
        self.pool = ResourcePool(name="test_pool", auto_detect=False)
        
        # Set up some basic capacity
        self.pool.set_capacity(ResourceType.CPU, 8, "cores")
        self.pool.set_capacity(ResourceType.MEMORY, 16, "GB")
        self.pool.set_capacity(ResourceType.GPU, 2, "devices")
    
    def test_pool_creation(self):
        """Test creating resource pool"""
        pool = ResourcePool(name="test", auto_detect=False)
        assert pool.name == "test"
        assert len(pool._capacity) == 0
        assert len(pool._allocations) == 0
        assert len(pool._reservations) == 0
    
    def test_auto_detect_creation(self):
        """Test creating pool with auto-detection"""
        with patch('parallel_framework.resource_pool.SystemResourceMonitor'):
            pool = ResourcePool(auto_detect=True)
            assert pool._system_monitor is not None
    
    def test_set_get_capacity(self):
        """Test setting and getting resource capacity"""
        self.pool.set_capacity(ResourceType.DISK, 1000, "GB", {"type": "SSD"})
        
        capacity = self.pool.get_capacity(ResourceType.DISK)
        assert capacity.resource_type == ResourceType.DISK
        assert capacity.amount == 1000
        assert capacity.unit == "GB"
        assert capacity.metadata == {"type": "SSD"}
        
        # Non-existent resource
        assert self.pool.get_capacity(ResourceType.NETWORK) is None
    
    def test_get_available_amount(self):
        """Test getting available resource amount"""
        # No allocations - full capacity available
        available_cpu = self.pool.get_available_amount(ResourceType.CPU, "cores")
        assert available_cpu == 8
        
        available_memory = self.pool.get_available_amount(ResourceType.MEMORY, "GB")
        assert available_memory == 16
        
        # Non-existent resource
        available_network = self.pool.get_available_amount(ResourceType.NETWORK, "mbps")
        assert available_network == 0
    
    def test_can_allocate(self):
        """Test checking if resources can be allocated"""
        # Can allocate within capacity
        specs = [
            ResourceSpec(ResourceType.CPU, 4, "cores"),
            ResourceSpec(ResourceType.MEMORY, 8, "GB")
        ]
        assert self.pool.can_allocate(specs)
        
        # Cannot allocate beyond capacity
        specs = [
            ResourceSpec(ResourceType.CPU, 10, "cores"),  # Exceeds capacity
            ResourceSpec(ResourceType.MEMORY, 8, "GB")
        ]
        assert not self.pool.can_allocate(specs)
    
    def test_allocate_resources_success(self):
        """Test successful resource allocation"""
        specs = [
            ResourceSpec(ResourceType.CPU, 4, "cores"),
            ResourceSpec(ResourceType.MEMORY, 8, "GB")
        ]
        
        allocations = self.pool.allocate_resources(specs, job_id="test_job")
        
        assert len(allocations) == 2
        assert all(isinstance(alloc, ResourceAllocation) for alloc in allocations)
        assert all(alloc.job_id == "test_job" for alloc in allocations)
        
        # Check available amounts reduced
        assert self.pool.get_available_amount(ResourceType.CPU, "cores") == 4
        assert self.pool.get_available_amount(ResourceType.MEMORY, "GB") == 8
    
    def test_allocate_resources_insufficient(self):
        """Test allocation failure due to insufficient resources"""
        specs = [
            ResourceSpec(ResourceType.CPU, 10, "cores"),  # Exceeds capacity
        ]
        
        with pytest.raises(InsufficientResourcesError, match="Insufficient resources"):
            self.pool.allocate_resources(specs)
    
    def test_allocate_resources_with_expiration(self):
        """Test resource allocation with expiration"""
        specs = [ResourceSpec(ResourceType.CPU, 2, "cores")]
        
        allocations = self.pool.allocate_resources(specs, expires_in_seconds=60)
        
        assert len(allocations) == 1
        assert allocations[0].expires_at is not None
        assert not allocations[0].is_expired
    
    def test_deallocate_resources(self):
        """Test resource deallocation"""
        # Allocate resources first
        specs = [ResourceSpec(ResourceType.CPU, 4, "cores")]
        allocations = self.pool.allocate_resources(specs)
        
        allocation_ids = [alloc.id for alloc in allocations]
        
        # Deallocate
        deallocated = self.pool.deallocate_resources(allocation_ids)
        assert deallocated == 1
        assert self.pool.get_available_amount(ResourceType.CPU, "cores") == 8
    
    def test_deallocate_job_resources(self):
        """Test deallocating all resources for a job"""
        specs = [
            ResourceSpec(ResourceType.CPU, 2, "cores"),
            ResourceSpec(ResourceType.MEMORY, 4, "GB")
        ]
        
        self.pool.allocate_resources(specs, job_id="job1")
        self.pool.allocate_resources([ResourceSpec(ResourceType.CPU, 1, "cores")], job_id="job2")
        
        # Deallocate job1 resources
        deallocated = self.pool.deallocate_job_resources("job1")
        assert deallocated == 2
        assert self.pool.get_available_amount(ResourceType.CPU, "cores") == 7  # 8 - 1 (job2)
        assert self.pool.get_available_amount(ResourceType.MEMORY, "GB") == 16  # All available
    
    def test_cleanup_expired_allocations(self):
        """Test cleaning up expired allocations"""
        # Create expired allocation
        specs = [ResourceSpec(ResourceType.CPU, 2, "cores")]
        allocations = self.pool.allocate_resources(specs, expires_in_seconds=1)
        
        # Wait for expiration
        time.sleep(1.1)
        
        cleaned = self.pool.cleanup_expired_allocations()
        assert cleaned == 1
        assert self.pool.get_available_amount(ResourceType.CPU, "cores") == 8
    
    def test_reserve_resources(self):
        """Test resource reservation"""
        specs = [ResourceSpec(ResourceType.CPU, 4, "cores")]
        
        reservations = self.pool.reserve_resources(specs, job_id="test_job")
        
        assert len(reservations) == 1
        assert reservations[0].job_id == "test_job"
        assert reservations[0].metadata["type"] == "reservation"
        
        # Available amount should be reduced by reservation
        available = self.pool._get_available_amount_with_reservations(ResourceType.CPU, "cores")
        assert available == 4
    
    def test_reserve_resources_insufficient(self):
        """Test reservation failure due to insufficient resources"""
        specs = [ResourceSpec(ResourceType.CPU, 10, "cores")]
        
        with pytest.raises(InsufficientResourcesError, match="Insufficient resources for reservation"):
            self.pool.reserve_resources(specs)
    
    def test_convert_reservation_to_allocation(self):
        """Test converting reservations to allocations"""
        specs = [ResourceSpec(ResourceType.CPU, 2, "cores")]
        reservations = self.pool.reserve_resources(specs, job_id="test_job")
        
        reservation_ids = [res.id for res in reservations]
        allocations = self.pool.convert_reservation_to_allocation(reservation_ids)
        
        assert len(allocations) == 1
        assert allocations[0].job_id == "test_job"
        assert "converted_from_reservation" in allocations[0].metadata
        
        # Reservation should be removed
        assert len(self.pool._reservations) == 0
        assert len(self.pool._allocations) == 1
    
    def test_cancel_reservations(self):
        """Test cancelling reservations"""
        specs = [ResourceSpec(ResourceType.CPU, 2, "cores")]
        reservations = self.pool.reserve_resources(specs)
        
        reservation_ids = [res.id for res in reservations]
        cancelled = self.pool.cancel_reservations(reservation_ids)
        
        assert cancelled == 1
        assert len(self.pool._reservations) == 0
    
    def test_cleanup_expired_reservations(self):
        """Test cleaning up expired reservations"""
        specs = [ResourceSpec(ResourceType.CPU, 2, "cores")]
        reservations = self.pool.reserve_resources(specs, expires_in_seconds=1)
        
        # Wait for expiration
        time.sleep(1.1)
        
        cleaned = self.pool.cleanup_expired_reservations()
        assert cleaned == 1
        assert len(self.pool._reservations) == 0
    
    def test_get_allocation(self):
        """Test getting allocation by ID"""
        specs = [ResourceSpec(ResourceType.CPU, 2, "cores")]
        allocations = self.pool.allocate_resources(specs)
        
        allocation = self.pool.get_allocation(allocations[0].id)
        assert allocation == allocations[0]
        
        # Non-existent allocation
        assert self.pool.get_allocation("nonexistent") is None
    
    def test_get_job_allocations(self):
        """Test getting allocations for a job"""
        specs1 = [ResourceSpec(ResourceType.CPU, 2, "cores")]
        specs2 = [ResourceSpec(ResourceType.MEMORY, 4, "GB")]
        
        self.pool.allocate_resources(specs1, job_id="job1")
        self.pool.allocate_resources(specs2, job_id="job1")
        self.pool.allocate_resources(specs1, job_id="job2")
        
        job1_allocations = self.pool.get_job_allocations("job1")
        assert len(job1_allocations) == 2
        
        job2_allocations = self.pool.get_job_allocations("job2")
        assert len(job2_allocations) == 1
    
    def test_get_reservation(self):
        """Test getting reservation by ID"""
        specs = [ResourceSpec(ResourceType.CPU, 2, "cores")]
        reservations = self.pool.reserve_resources(specs)
        
        reservation = self.pool.get_reservation(reservations[0].id)
        assert reservation == reservations[0]
        
        # Non-existent reservation
        assert self.pool.get_reservation("nonexistent") is None
    
    def test_get_job_reservations(self):
        """Test getting reservations for a job"""
        specs1 = [ResourceSpec(ResourceType.CPU, 2, "cores")]
        specs2 = [ResourceSpec(ResourceType.MEMORY, 4, "GB")]
        
        self.pool.reserve_resources(specs1, job_id="job1")
        self.pool.reserve_resources(specs2, job_id="job1")
        self.pool.reserve_resources(specs1, job_id="job2")
        
        job1_reservations = self.pool.get_job_reservations("job1")
        assert len(job1_reservations) == 2
        
        job2_reservations = self.pool.get_job_reservations("job2")
        assert len(job2_reservations) == 1
    
    def test_resource_utilization(self):
        """Test getting resource utilization"""
        # Allocate some resources
        specs = [
            ResourceSpec(ResourceType.CPU, 4, "cores"),
            ResourceSpec(ResourceType.MEMORY, 8, "GB")
        ]
        self.pool.allocate_resources(specs)
        
        utilization = self.pool.get_resource_utilization()
        
        assert ResourceType.CPU in utilization
        assert ResourceType.MEMORY in utilization
        
        cpu_util = utilization[ResourceType.CPU]
        assert cpu_util["total_capacity"] == 8
        assert cpu_util["allocated_amount"] == 4
        assert cpu_util["available_amount"] == 4
        assert cpu_util["utilization_percent"] == 50.0
        assert cpu_util["allocation_count"] == 1
    
    def test_statistics(self):
        """Test getting pool statistics"""
        # Perform some operations
        specs = [ResourceSpec(ResourceType.CPU, 2, "cores")]
        allocations = self.pool.allocate_resources(specs)
        reservations = self.pool.reserve_resources(specs)
        
        stats = self.pool.get_statistics()
        
        assert stats["name"] == "test_pool"
        assert stats["total_allocations"] == 1
        assert stats["current_allocations"] == 1
        assert stats["current_reservations"] == 1
        assert "resource_utilization" in stats
        assert "capacity" in stats
    
    def test_convert_resource_amount(self):
        """Test resource amount conversion"""
        # Memory conversions
        bytes_to_mb = self.pool._convert_resource_amount(1048576, "bytes", "MB", ResourceType.MEMORY)
        assert bytes_to_mb == 1.0
        
        gb_to_bytes = self.pool._convert_resource_amount(1, "GB", "bytes", ResourceType.MEMORY)
        assert gb_to_bytes == 1024 ** 3
        
        # Same unit conversion
        same_unit = self.pool._convert_resource_amount(100, "cores", "cores", ResourceType.CPU)
        assert same_unit == 100.0
    
    def test_thread_safety(self):
        """Test thread safety of resource pool operations"""
        results = []
        errors = []
        
        def allocate_worker():
            try:
                specs = [ResourceSpec(ResourceType.CPU, 1, "cores")]
                allocation = self.pool.allocate_resources(specs)
                results.append(allocation)
                time.sleep(0.1)
                self.pool.deallocate_resources([allocation[0].id])
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=allocate_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0  # No errors should occur
        assert len(results) == 5  # All allocations should succeed
        assert self.pool.get_available_amount(ResourceType.CPU, "cores") == 8  # All deallocated


class TestResourcePoolIntegration:
    """Integration tests for resource pool with monitoring"""
    
    @patch('parallel_framework.resource_pool.psutil')
    def test_pool_with_monitoring(self, mock_psutil):
        """Test resource pool with system monitoring"""
        # Mock system info
        mock_psutil.cpu_count.return_value = 4
        mock_psutil.cpu_percent.return_value = 25.0
        
        mock_memory = MagicMock()
        mock_memory.total = 8 * 1024**3  # 8GB
        mock_memory.available = 6 * 1024**3  # 6GB available
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_disk = MagicMock()
        mock_disk.total = 500 * 1024**3  # 500GB
        mock_disk.free = 200 * 1024**3  # 200GB free
        mock_psutil.disk_usage.return_value = mock_disk
        
        # Create pool with monitoring
        pool = ResourcePool(auto_detect=True)
        
        # Should have initialized capacity from system
        cpu_capacity = pool.get_capacity(ResourceType.CPU)
        assert cpu_capacity is not None
        assert cpu_capacity.amount == 4
        assert cpu_capacity.unit == "cores"
        
        memory_capacity = pool.get_capacity(ResourceType.MEMORY)
        assert memory_capacity is not None
        assert memory_capacity.unit == "bytes"


if __name__ == "__main__":
    pytest.main([__file__])