"""
Resource pool management for the parallel processing framework.

This module provides classes for managing computational resources including
CPU cores, memory, GPU devices, and custom resources.
"""

import os
import psutil
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Set, Union, Tuple
from enum import Enum
from datetime import datetime, timezone
import uuid
import logging

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of computational resources"""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"
    CUSTOM = "custom"


@dataclass
class ResourceSpec:
    """Specification for a resource requirement or allocation"""
    resource_type: ResourceType
    amount: Union[int, float]
    unit: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.amount < 0:
            raise ValueError("Resource amount must be non-negative")
        
        # Validate units for known resource types
        valid_units = {
            ResourceType.CPU: ["cores", "percent"],
            ResourceType.MEMORY: ["bytes", "KB", "MB", "GB", "TB"],
            ResourceType.GPU: ["devices", "memory_mb", "memory_gb"],
            ResourceType.DISK: ["bytes", "KB", "MB", "GB", "TB"],
            ResourceType.NETWORK: ["mbps", "gbps"]
        }
        
        if self.resource_type in valid_units:
            if self.unit not in valid_units[self.resource_type]:
                raise ValueError(f"Invalid unit '{self.unit}' for {self.resource_type.value}")


@dataclass
class ResourceAllocation:
    """Represents an allocated resource with tracking information"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    resource_spec: ResourceSpec = None
    job_id: Optional[str] = None
    allocated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if allocation has expired"""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at


class ResourceError(Exception):
    """Base exception for resource management errors"""
    pass


class InsufficientResourcesError(ResourceError):
    """Raised when requested resources are not available"""
    pass


class ResourceAllocationError(ResourceError):
    """Raised when resource allocation fails"""
    pass


class SystemResourceMonitor:
    """Monitors system resource usage and availability"""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self._running = False
        self._thread = None
        self._lock = threading.RLock()
        self._system_info = {}
        
        # Initialize system info
        self._update_system_info()
    
    def start(self):
        """Start monitoring system resources"""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()
            logger.info("System resource monitor started")
    
    def stop(self):
        """Stop monitoring system resources"""
        with self._lock:
            if not self._running:
                return
            
            self._running = False
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=2.0)
            logger.info("System resource monitor stopped")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get current system resource information"""
        with self._lock:
            return self._system_info.copy()
    
    def get_available_resources(self) -> Dict[ResourceType, ResourceSpec]:
        """Get currently available system resources"""
        info = self.get_system_info()
        
        available = {}
        
        # CPU availability
        cpu_percent = info.get('cpu_percent', 0)
        available_cpu_percent = max(0, 100 - cpu_percent)
        available[ResourceType.CPU] = ResourceSpec(
            resource_type=ResourceType.CPU,
            amount=available_cpu_percent,
            unit="percent",
            metadata={"cores": info.get('cpu_count', 1)}
        )
        
        # Memory availability
        memory_info = info.get('memory', {})
        available_memory = memory_info.get('available', 0)
        available[ResourceType.MEMORY] = ResourceSpec(
            resource_type=ResourceType.MEMORY,
            amount=available_memory,
            unit="bytes",
            metadata={"total": memory_info.get('total', 0)}
        )
        
        # Disk availability
        disk_info = info.get('disk', {})
        available_disk = disk_info.get('free', 0)
        available[ResourceType.DISK] = ResourceSpec(
            resource_type=ResourceType.DISK,
            amount=available_disk,
            unit="bytes",
            metadata={"total": disk_info.get('total', 0)}
        )
        
        return available
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                self._update_system_info()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error updating system info: {e}")
                time.sleep(self.update_interval)
    
    def _update_system_info(self):
        """Update system resource information"""
        try:
            info = {}
            
            # CPU information
            info['cpu_count'] = psutil.cpu_count()
            info['cpu_count_logical'] = psutil.cpu_count(logical=True)
            info['cpu_percent'] = psutil.cpu_percent(interval=0.1)
            info['cpu_freq'] = psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {}
            
            # Memory information
            memory = psutil.virtual_memory()
            info['memory'] = {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used,
                'free': memory.free
            }
            
            # Disk information (root partition)
            disk = psutil.disk_usage('/')
            info['disk'] = {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': (disk.used / disk.total) * 100
            }
            
            # Network information
            net_io = psutil.net_io_counters()
            if net_io:
                info['network'] = {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv
                }
            
            # GPU information (if available)
            info['gpu'] = self._get_gpu_info()
            
            with self._lock:
                self._system_info = info
                
        except Exception as e:
            logger.error(f"Failed to update system info: {e}")
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information if available"""
        try:
            # Try to detect GPU using various methods
            gpu_info = {"available": False, "devices": []}
            
            # Check for NVIDIA GPUs
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_info["available"] = True
                    gpu_info["type"] = "nvidia"
                    gpu_info["devices"] = [
                        {
                            "id": gpu.id,
                            "name": gpu.name,
                            "memory_total": gpu.memoryTotal,
                            "memory_used": gpu.memoryUsed,
                            "memory_free": gpu.memoryFree,
                            "load": gpu.load,
                            "temperature": gpu.temperature
                        }
                        for gpu in gpus
                    ]
            except ImportError:
                pass
            
            # Check for Metal Performance Shaders (macOS)
            if not gpu_info["available"]:
                try:
                    import platform
                    if platform.system() == "Darwin":
                        # Basic Metal detection (simplified)
                        gpu_info["available"] = True
                        gpu_info["type"] = "metal"
                        gpu_info["devices"] = [{"id": 0, "name": "Metal GPU", "type": "integrated"}]
                except Exception:
                    pass
            
            return gpu_info
            
        except Exception as e:
            logger.debug(f"Failed to get GPU info: {e}")
            return {"available": False, "devices": []}


class ResourcePool:
    """
    Manages a pool of computational resources with allocation and tracking.
    
    Provides resource allocation, deallocation, reservation, and monitoring
    capabilities for different types of computational resources.
    """
    
    def __init__(self, name: str = "default", auto_detect: bool = True):
        self.name = name
        self._lock = threading.RLock()
        
        # Resource capacity and current allocations
        self._capacity: Dict[ResourceType, ResourceSpec] = {}
        self._allocations: Dict[str, ResourceAllocation] = {}
        self._reservations: Dict[str, ResourceAllocation] = {}
        
        # System monitoring
        self._system_monitor = SystemResourceMonitor() if auto_detect else None
        
        # Statistics
        self._allocation_history: List[ResourceAllocation] = []
        self._total_allocations = 0
        self._total_deallocations = 0
        
        if auto_detect:
            self._initialize_from_system()
    
    def set_capacity(self, resource_type: ResourceType, amount: Union[int, float], 
                    unit: str, metadata: Optional[Dict[str, Any]] = None):
        """Set the total capacity for a resource type"""
        with self._lock:
            self._capacity[resource_type] = ResourceSpec(
                resource_type=resource_type,
                amount=amount,
                unit=unit,
                metadata=metadata or {}
            )
            logger.info(f"Set {resource_type.value} capacity to {amount} {unit}")
    
    def get_capacity(self, resource_type: ResourceType) -> Optional[ResourceSpec]:
        """Get the total capacity for a resource type"""
        with self._lock:
            return self._capacity.get(resource_type)
    
    def get_available_amount(self, resource_type: ResourceType, unit: str) -> float:
        """Get the available amount of a specific resource type"""
        with self._lock:
            capacity = self._capacity.get(resource_type)
            if not capacity:
                return 0.0
            
            # Convert capacity to requested unit
            total_amount = self._convert_resource_amount(
                capacity.amount, capacity.unit, unit, resource_type
            )
            
            # Calculate allocated amount
            allocated_amount = 0.0
            for allocation in self._allocations.values():
                if allocation.resource_spec.resource_type == resource_type:
                    allocated_amount += self._convert_resource_amount(
                        allocation.resource_spec.amount,
                        allocation.resource_spec.unit,
                        unit,
                        resource_type
                    )
            
            return max(0.0, total_amount - allocated_amount)
    
    def can_allocate(self, resource_specs: List[ResourceSpec]) -> bool:
        """Check if resources can be allocated"""
        with self._lock:
            for spec in resource_specs:
                available = self.get_available_amount(spec.resource_type, spec.unit)
                if available < spec.amount:
                    return False
            return True
    
    def reserve_resources(self, resource_specs: List[ResourceSpec],
                         job_id: Optional[str] = None,
                         expires_in_seconds: int = 300) -> List[ResourceAllocation]:
        """
        Reserve resources for future allocation.
        
        Args:
            resource_specs: List of resource specifications to reserve
            job_id: Optional job ID for tracking
            expires_in_seconds: Reservation expiration time (default 5 minutes)
            
        Returns:
            List of resource reservations
            
        Raises:
            InsufficientResourcesError: If resources are not available
        """
        with self._lock:
            # Check if resources are available (including reservations)
            if not self._can_allocate_with_reservations(resource_specs):
                unavailable = []
                for spec in resource_specs:
                    available = self._get_available_amount_with_reservations(spec.resource_type, spec.unit)
                    if available < spec.amount:
                        unavailable.append(f"{spec.resource_type.value}: need {spec.amount} {spec.unit}, have {available}")
                
                raise InsufficientResourcesError(
                    f"Insufficient resources for reservation: {', '.join(unavailable)}"
                )
            
            # Create reservations
            reservations = []
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in_seconds)
            
            for spec in resource_specs:
                reservation = ResourceAllocation(
                    resource_spec=spec,
                    job_id=job_id,
                    expires_at=expires_at,
                    metadata={"type": "reservation"}
                )
                reservations.append(reservation)
                self._reservations[reservation.id] = reservation
            
            logger.info(f"Reserved {len(reservations)} resources for job {job_id}")
            return reservations
    
    def convert_reservation_to_allocation(self, reservation_ids: List[str]) -> List[ResourceAllocation]:
        """
        Convert reservations to actual allocations.
        
        Args:
            reservation_ids: List of reservation IDs to convert
            
        Returns:
            List of new resource allocations
        """
        with self._lock:
            allocations = []
            
            for reservation_id in reservation_ids:
                if reservation_id not in self._reservations:
                    logger.warning(f"Reservation {reservation_id} not found")
                    continue
                
                reservation = self._reservations.pop(reservation_id)
                
                # Check if reservation has expired
                if reservation.is_expired:
                    logger.warning(f"Reservation {reservation_id} has expired")
                    continue
                
                # Create new allocation with same spec but no expiration
                allocation = ResourceAllocation(
                    resource_spec=reservation.resource_spec,
                    job_id=reservation.job_id,
                    metadata=reservation.metadata.copy()
                )
                allocation.metadata["converted_from_reservation"] = reservation_id
                
                allocations.append(allocation)
                self._allocations[allocation.id] = allocation
            
            if allocations:
                self._total_allocations += len(allocations)
                self._allocation_history.extend(allocations)
                logger.info(f"Converted {len(allocations)} reservations to allocations")
            
            return allocations
    
    def cancel_reservations(self, reservation_ids: List[str]) -> int:
        """
        Cancel resource reservations.
        
        Args:
            reservation_ids: List of reservation IDs to cancel
            
        Returns:
            Number of reservations cancelled
        """
        with self._lock:
            cancelled = 0
            for reservation_id in reservation_ids:
                if reservation_id in self._reservations:
                    self._reservations.pop(reservation_id)
                    cancelled += 1
            
            if cancelled > 0:
                logger.info(f"Cancelled {cancelled} reservations")
            
            return cancelled
    
    def cleanup_expired_reservations(self) -> int:
        """Clean up expired resource reservations"""
        with self._lock:
            expired_ids = [
                res_id for res_id, res in self._reservations.items()
                if res.is_expired
            ]
            
            if expired_ids:
                logger.info(f"Cleaning up {len(expired_ids)} expired reservations")
                return self.cancel_reservations(expired_ids)
            
            return 0
    
    def get_reservation(self, reservation_id: str) -> Optional[ResourceAllocation]:
        """Get reservation by ID"""
        with self._lock:
            return self._reservations.get(reservation_id)
    
    def get_job_reservations(self, job_id: str) -> List[ResourceAllocation]:
        """Get all reservations for a specific job"""
        with self._lock:
            return [
                res for res in self._reservations.values()
                if res.job_id == job_id
            ]
    
    def _can_allocate_with_reservations(self, resource_specs: List[ResourceSpec]) -> bool:
        """Check if resources can be allocated considering existing reservations"""
        for spec in resource_specs:
            available = self._get_available_amount_with_reservations(spec.resource_type, spec.unit)
            if available < spec.amount:
                return False
        return True
    
    def _get_available_amount_with_reservations(self, resource_type: ResourceType, unit: str) -> float:
        """Get available amount considering both allocations and reservations"""
        capacity = self._capacity.get(resource_type)
        if not capacity:
            return 0.0
        
        # Convert capacity to requested unit
        total_amount = self._convert_resource_amount(
            capacity.amount, capacity.unit, unit, resource_type
        )
        
        # Calculate allocated amount (including reservations)
        used_amount = 0.0
        
        # Add current allocations
        for allocation in self._allocations.values():
            if allocation.resource_spec.resource_type == resource_type:
                used_amount += self._convert_resource_amount(
                    allocation.resource_spec.amount,
                    allocation.resource_spec.unit,
                    unit,
                    resource_type
                )
        
        # Add current reservations
        for reservation in self._reservations.values():
            if reservation.resource_spec.resource_type == resource_type and not reservation.is_expired:
                used_amount += self._convert_resource_amount(
                    reservation.resource_spec.amount,
                    reservation.resource_spec.unit,
                    unit,
                    resource_type
                )
        
        return max(0.0, total_amount - used_amount)
    
    def allocate_resources(self, resource_specs: List[ResourceSpec], 
                          job_id: Optional[str] = None,
                          expires_in_seconds: Optional[int] = None) -> List[ResourceAllocation]:
        """
        Allocate resources and return allocation objects.
        
        Args:
            resource_specs: List of resource specifications to allocate
            job_id: Optional job ID for tracking
            expires_in_seconds: Optional expiration time in seconds
            
        Returns:
            List of resource allocations
            
        Raises:
            InsufficientResourcesError: If resources are not available
            ResourceAllocationError: If allocation fails
        """
        with self._lock:
            # Check if resources are available
            if not self.can_allocate(resource_specs):
                unavailable = []
                for spec in resource_specs:
                    available = self.get_available_amount(spec.resource_type, spec.unit)
                    if available < spec.amount:
                        unavailable.append(f"{spec.resource_type.value}: need {spec.amount} {spec.unit}, have {available}")
                
                raise InsufficientResourcesError(
                    f"Insufficient resources: {', '.join(unavailable)}"
                )
            
            # Create allocations
            allocations = []
            expires_at = None
            if expires_in_seconds:
                expires_at = datetime.now(timezone.utc).replace(
                    microsecond=0
                ) + timedelta(seconds=expires_in_seconds)
            
            try:
                for spec in resource_specs:
                    allocation = ResourceAllocation(
                        resource_spec=spec,
                        job_id=job_id,
                        expires_at=expires_at
                    )
                    allocations.append(allocation)
                    self._allocations[allocation.id] = allocation
                
                # Update statistics
                self._total_allocations += len(allocations)
                self._allocation_history.extend(allocations)
                
                # Limit history size
                if len(self._allocation_history) > 1000:
                    self._allocation_history = self._allocation_history[-1000:]
                
                logger.info(f"Allocated {len(allocations)} resources for job {job_id}")
                return allocations
                
            except Exception as e:
                # Rollback allocations on error
                for allocation in allocations:
                    self._allocations.pop(allocation.id, None)
                raise ResourceAllocationError(f"Failed to allocate resources: {e}")
    
    def deallocate_resources(self, allocation_ids: List[str]) -> int:
        """
        Deallocate resources by allocation IDs.
        
        Args:
            allocation_ids: List of allocation IDs to deallocate
            
        Returns:
            Number of resources deallocated
        """
        with self._lock:
            deallocated = 0
            for allocation_id in allocation_ids:
                if allocation_id in self._allocations:
                    allocation = self._allocations.pop(allocation_id)
                    deallocated += 1
                    logger.debug(f"Deallocated resource {allocation_id}")
            
            self._total_deallocations += deallocated
            if deallocated > 0:
                logger.info(f"Deallocated {deallocated} resources")
            
            return deallocated
    
    def deallocate_job_resources(self, job_id: str) -> int:
        """Deallocate all resources for a specific job"""
        with self._lock:
            job_allocations = [
                alloc_id for alloc_id, alloc in self._allocations.items()
                if alloc.job_id == job_id
            ]
            return self.deallocate_resources(job_allocations)
    
    def cleanup_expired_allocations(self) -> int:
        """Clean up expired resource allocations"""
        with self._lock:
            expired_ids = [
                alloc_id for alloc_id, alloc in self._allocations.items()
                if alloc.is_expired
            ]
            
            if expired_ids:
                logger.info(f"Cleaning up {len(expired_ids)} expired allocations")
                return self.deallocate_resources(expired_ids)
            
            return 0
    
    def get_allocation(self, allocation_id: str) -> Optional[ResourceAllocation]:
        """Get allocation by ID"""
        with self._lock:
            return self._allocations.get(allocation_id)
    
    def get_job_allocations(self, job_id: str) -> List[ResourceAllocation]:
        """Get all allocations for a specific job"""
        with self._lock:
            return [
                alloc for alloc in self._allocations.values()
                if alloc.job_id == job_id
            ]
    
    def get_resource_utilization(self) -> Dict[ResourceType, Dict[str, float]]:
        """Get current resource utilization statistics"""
        with self._lock:
            utilization = {}
            
            for resource_type, capacity in self._capacity.items():
                allocated_amount = 0.0
                allocation_count = 0
                
                for allocation in self._allocations.values():
                    if allocation.resource_spec.resource_type == resource_type:
                        allocated_amount += self._convert_resource_amount(
                            allocation.resource_spec.amount,
                            allocation.resource_spec.unit,
                            capacity.unit,
                            resource_type
                        )
                        allocation_count += 1
                
                utilization_percent = (allocated_amount / capacity.amount) * 100 if capacity.amount > 0 else 0
                
                utilization[resource_type] = {
                    "total_capacity": capacity.amount,
                    "allocated_amount": allocated_amount,
                    "available_amount": capacity.amount - allocated_amount,
                    "utilization_percent": utilization_percent,
                    "allocation_count": allocation_count,
                    "unit": capacity.unit
                }
            
            return utilization
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self._lock:
            return {
                "name": self.name,
                "total_allocations": self._total_allocations,
                "total_deallocations": self._total_deallocations,
                "current_allocations": len(self._allocations),
                "current_reservations": len(self._reservations),
                "resource_utilization": self.get_resource_utilization(),
                "capacity": {rt.value: spec.__dict__ for rt, spec in self._capacity.items()}
            }
    
    def start_monitoring(self):
        """Start system resource monitoring"""
        if self._system_monitor:
            self._system_monitor.start()
    
    def stop_monitoring(self):
        """Stop system resource monitoring"""
        if self._system_monitor:
            self._system_monitor.stop()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get current system resource information"""
        if self._system_monitor:
            return self._system_monitor.get_system_info()
        return {}
    
    def _initialize_from_system(self):
        """Initialize resource capacity from system detection"""
        if not self._system_monitor:
            return
        
        try:
            self._system_monitor._update_system_info()
            system_info = self._system_monitor.get_system_info()
            
            # Set CPU capacity
            cpu_count = system_info.get('cpu_count', 1)
            self.set_capacity(ResourceType.CPU, cpu_count, "cores", 
                            {"logical_cores": system_info.get('cpu_count_logical', cpu_count)})
            
            # Set memory capacity (use 80% of total to be safe)
            memory_total = system_info.get('memory', {}).get('total', 0)
            if memory_total > 0:
                usable_memory = int(memory_total * 0.8)
                self.set_capacity(ResourceType.MEMORY, usable_memory, "bytes")
            
            # Set disk capacity (use 50% of available space to be conservative)
            disk_free = system_info.get('disk', {}).get('free', 0)
            if disk_free > 0:
                usable_disk = int(disk_free * 0.5)
                self.set_capacity(ResourceType.DISK, usable_disk, "bytes")
            
            # Set GPU capacity if available
            gpu_info = system_info.get('gpu', {})
            if gpu_info.get('available', False):
                gpu_devices = len(gpu_info.get('devices', []))
                if gpu_devices > 0:
                    self.set_capacity(ResourceType.GPU, gpu_devices, "devices", 
                                    {"type": gpu_info.get('type', 'unknown')})
            
            logger.info(f"Initialized resource pool '{self.name}' from system")
            
        except Exception as e:
            logger.error(f"Failed to initialize from system: {e}")
    
    def _convert_resource_amount(self, amount: Union[int, float], from_unit: str, 
                                to_unit: str, resource_type: ResourceType) -> float:
        """Convert resource amount between units"""
        if from_unit == to_unit:
            return float(amount)
        
        # Memory and disk conversions (bytes to larger units)
        size_conversions = {
            "bytes": 1,
            "KB": 1024,
            "MB": 1024 ** 2,
            "GB": 1024 ** 3,
            "TB": 1024 ** 4
        }
        
        if (resource_type in [ResourceType.MEMORY, ResourceType.DISK] and
            from_unit in size_conversions and to_unit in size_conversions):
            
            # Convert to bytes first, then to target unit
            bytes_amount = amount * size_conversions[from_unit]
            return bytes_amount / size_conversions[to_unit]
        
        # Network conversions
        network_conversions = {
            "mbps": 1,
            "gbps": 1000
        }
        
        if (resource_type == ResourceType.NETWORK and
            from_unit in network_conversions and to_unit in network_conversions):
            
            mbps_amount = amount * network_conversions[from_unit]
            return mbps_amount / network_conversions[to_unit]
        
        # If we can't convert, return the original amount
        logger.warning(f"Cannot convert {resource_type.value} from {from_unit} to {to_unit}")
        return float(amount)


# Import datetime for expiration handling
from datetime import timedelta