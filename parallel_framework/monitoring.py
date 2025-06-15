"""
Resource monitoring and metrics collection system for the parallel processing framework.

This module provides comprehensive monitoring capabilities for resource usage,
system health, performance metrics, and alerting based on configurable thresholds.
"""

import statistics
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any, Tuple, Union
import logging

from .resource_pool import ResourcePool, ResourceType, ResourceSpec

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """A single metric data point"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """A time series of metric points"""
    name: str
    metric_type: MetricType
    unit: str
    description: str
    points: deque = field(default_factory=lambda: deque(maxlen=1000))
    labels: Dict[str, str] = field(default_factory=dict)
    
    def add_point(self, value: float, labels: Optional[Dict[str, str]] = None, 
                  metadata: Optional[Dict[str, Any]] = None):
        """Add a metric point to the series"""
        point = MetricPoint(
            timestamp=datetime.now(timezone.utc),
            value=value,
            labels=labels or {},
            metadata=metadata or {}
        )
        self.points.append(point)
    
    def get_latest_value(self) -> Optional[float]:
        """Get the most recent metric value"""
        return self.points[-1].value if self.points else None
    
    def get_average(self, duration_seconds: Optional[int] = None) -> Optional[float]:
        """Get average value over specified duration or all points"""
        if not self.points:
            return None
        
        if duration_seconds is None:
            values = [point.value for point in self.points]
        else:
            cutoff = datetime.now(timezone.utc) - timedelta(seconds=duration_seconds)
            values = [point.value for point in self.points if point.timestamp >= cutoff]
        
        return statistics.mean(values) if values else None
    
    def get_percentile(self, percentile: float, duration_seconds: Optional[int] = None) -> Optional[float]:
        """Get percentile value over specified duration"""
        if not self.points:
            return None
        
        if duration_seconds is None:
            values = [point.value for point in self.points]
        else:
            cutoff = datetime.now(timezone.utc) - timedelta(seconds=duration_seconds)
            values = [point.value for point in self.points if point.timestamp >= cutoff]
        
        if not values:
            return None
        
        return statistics.quantiles(values, n=100)[int(percentile) - 1] if len(values) > 1 else values[0]


@dataclass
class Alert:
    """System alert"""
    id: str
    name: str
    severity: AlertSeverity
    message: str
    metric_name: str
    threshold_value: float
    current_value: float
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_active(self) -> bool:
        """Check if alert is currently active"""
        return self.resolved_at is None
    
    @property
    def duration(self) -> timedelta:
        """Get alert duration"""
        end_time = self.resolved_at or datetime.now(timezone.utc)
        return end_time - self.triggered_at


@dataclass
class ThresholdRule:
    """Threshold-based alerting rule"""
    name: str
    metric_name: str
    operator: str  # gt, lt, gte, lte, eq, ne
    threshold: float
    severity: AlertSeverity
    duration_seconds: int = 60  # How long condition must persist
    description: str = ""
    enabled: bool = True
    
    def evaluate(self, current_value: float) -> bool:
        """Evaluate if threshold condition is met"""
        if not self.enabled:
            return False
        
        operators = {
            "gt": lambda x, y: x > y,
            "lt": lambda x, y: x < y,
            "gte": lambda x, y: x >= y,
            "lte": lambda x, y: x <= y,
            "eq": lambda x, y: x == y,
            "ne": lambda x, y: x != y
        }
        
        return operators.get(self.operator, lambda x, y: False)(current_value, self.threshold)


class ResourceMetrics:
    """
    Tracks resource usage patterns and performance metrics.
    
    Collects detailed metrics about resource utilization, allocation patterns,
    job performance, and system efficiency.
    """
    
    def __init__(self, resource_pool: ResourcePool):
        self.resource_pool = resource_pool
        self._metrics: Dict[str, MetricSeries] = {}
        self._lock = threading.RLock()
        
        # Initialize core resource metrics
        self._initialize_core_metrics()
    
    def _initialize_core_metrics(self):
        """Initialize core resource metrics"""
        resource_types = [ResourceType.CPU, ResourceType.MEMORY, ResourceType.GPU, ResourceType.DISK]
        
        for resource_type in resource_types:
            rt_name = resource_type.value
            
            # Utilization metrics
            self._create_metric(
                f"{rt_name}_utilization_percent",
                MetricType.GAUGE,
                "percent",
                f"{rt_name.upper()} utilization percentage"
            )
            
            # Allocation metrics
            self._create_metric(
                f"{rt_name}_allocated_amount",
                MetricType.GAUGE,
                "units",
                f"{rt_name.upper()} allocated amount"
            )
            
            # Available metrics
            self._create_metric(
                f"{rt_name}_available_amount",
                MetricType.GAUGE,
                "units",
                f"{rt_name.upper()} available amount"
            )
            
            # Allocation count metrics
            self._create_metric(
                f"{rt_name}_allocation_count",
                MetricType.GAUGE,
                "count",
                f"Number of active {rt_name.upper()} allocations"
            )
        
        # System-wide metrics
        self._create_metric("total_jobs_scheduled", MetricType.COUNTER, "count", "Total jobs scheduled")
        self._create_metric("total_jobs_completed", MetricType.COUNTER, "count", "Total jobs completed")
        self._create_metric("total_jobs_failed", MetricType.COUNTER, "count", "Total jobs failed")
        self._create_metric("active_jobs", MetricType.GAUGE, "count", "Currently active jobs")
        self._create_metric("queue_depth", MetricType.GAUGE, "count", "Total jobs in all queues")
        self._create_metric("average_job_duration", MetricType.GAUGE, "seconds", "Average job completion time")
        self._create_metric("system_load", MetricType.GAUGE, "percent", "Overall system load percentage")
    
    def _create_metric(self, name: str, metric_type: MetricType, unit: str, description: str) -> MetricSeries:
        """Create a new metric series"""
        with self._lock:
            metric = MetricSeries(
                name=name,
                metric_type=metric_type,
                unit=unit,
                description=description
            )
            self._metrics[name] = metric
            return metric
    
    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None,
                     metadata: Optional[Dict[str, Any]] = None):
        """Record a metric value"""
        with self._lock:
            if name in self._metrics:
                self._metrics[name].add_point(value, labels, metadata)
            else:
                logger.warning(f"Attempted to record unknown metric: {name}")
    
    def get_metric(self, name: str) -> Optional[MetricSeries]:
        """Get a metric series by name"""
        with self._lock:
            return self._metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, MetricSeries]:
        """Get all metric series"""
        with self._lock:
            return self._metrics.copy()
    
    def update_resource_metrics(self):
        """Update all resource-related metrics"""
        try:
            utilization = self.resource_pool.get_resource_utilization()
            
            for resource_type, util_info in utilization.items():
                rt_name = resource_type.value
                
                # Record utilization metrics
                self.record_metric(f"{rt_name}_utilization_percent", util_info["utilization_percent"])
                self.record_metric(f"{rt_name}_allocated_amount", util_info["allocated_amount"])
                self.record_metric(f"{rt_name}_available_amount", util_info["available_amount"])
                self.record_metric(f"{rt_name}_allocation_count", util_info["allocation_count"])
                
        except Exception as e:
            logger.error(f"Error updating resource metrics: {e}")
    
    def record_job_scheduled(self, job_id: str, metadata: Optional[Dict[str, Any]] = None):
        """Record that a job was scheduled"""
        current_count = self.get_metric("total_jobs_scheduled")
        current_value = current_count.get_latest_value() or 0
        self.record_metric("total_jobs_scheduled", current_value + 1, metadata=metadata)
    
    def record_job_completed(self, job_id: str, duration_seconds: float, 
                           metadata: Optional[Dict[str, Any]] = None):
        """Record job completion"""
        current_count = self.get_metric("total_jobs_completed")
        current_value = current_count.get_latest_value() or 0
        self.record_metric("total_jobs_completed", current_value + 1, metadata=metadata)
        
        # Update average duration
        avg_metric = self.get_metric("average_job_duration")
        if avg_metric and avg_metric.points:
            current_avg = avg_metric.get_average()
            total_completed = current_value + 1
            new_avg = ((current_avg * (total_completed - 1)) + duration_seconds) / total_completed
            self.record_metric("average_job_duration", new_avg)
        else:
            self.record_metric("average_job_duration", duration_seconds)
    
    def record_job_failed(self, job_id: str, error: str, metadata: Optional[Dict[str, Any]] = None):
        """Record job failure"""
        current_count = self.get_metric("total_jobs_failed")
        current_value = current_count.get_latest_value() or 0
        failure_metadata = {"error": error, **(metadata or {})}
        self.record_metric("total_jobs_failed", current_value + 1, metadata=failure_metadata)
    
    def update_queue_metrics(self, queue_depth: int, active_jobs: int):
        """Update queue-related metrics"""
        self.record_metric("queue_depth", queue_depth)
        self.record_metric("active_jobs", active_jobs)
    
    def calculate_system_load(self) -> float:
        """Calculate overall system load percentage"""
        try:
            utilization = self.resource_pool.get_resource_utilization()
            
            # Weight different resources (CPU and Memory are most important)
            weights = {
                ResourceType.CPU: 0.4,
                ResourceType.MEMORY: 0.4,
                ResourceType.GPU: 0.15,
                ResourceType.DISK: 0.05
            }
            
            weighted_load = 0.0
            total_weight = 0.0
            
            for resource_type, util_info in utilization.items():
                weight = weights.get(resource_type, 0.0)
                if weight > 0:
                    weighted_load += util_info["utilization_percent"] * weight
                    total_weight += weight
            
            return weighted_load / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating system load: {e}")
            return 0.0
    
    def get_performance_summary(self, duration_seconds: int = 3600) -> Dict[str, Any]:
        """Get performance summary over specified duration"""
        with self._lock:
            summary = {
                "duration_seconds": duration_seconds,
                "timestamp": datetime.now(timezone.utc),
                "resource_utilization": {},
                "job_statistics": {},
                "system_health": {}
            }
            
            # Resource utilization summary
            for resource_type in [ResourceType.CPU, ResourceType.MEMORY, ResourceType.GPU, ResourceType.DISK]:
                rt_name = resource_type.value
                util_metric = self.get_metric(f"{rt_name}_utilization_percent")
                
                if util_metric:
                    summary["resource_utilization"][rt_name] = {
                        "current": util_metric.get_latest_value(),
                        "average": util_metric.get_average(duration_seconds),
                        "peak": max([p.value for p in util_metric.points] or [0])
                    }
            
            # Job statistics
            jobs_scheduled = self.get_metric("total_jobs_scheduled")
            jobs_completed = self.get_metric("total_jobs_completed")
            jobs_failed = self.get_metric("total_jobs_failed")
            avg_duration = self.get_metric("average_job_duration")
            
            summary["job_statistics"] = {
                "scheduled": jobs_scheduled.get_latest_value() if jobs_scheduled else 0,
                "completed": jobs_completed.get_latest_value() if jobs_completed else 0,
                "failed": jobs_failed.get_latest_value() if jobs_failed else 0,
                "average_duration": avg_duration.get_latest_value() if avg_duration else 0,
                "success_rate": self._calculate_success_rate()
            }
            
            # System health
            system_load = self.calculate_system_load()
            self.record_metric("system_load", system_load)
            
            summary["system_health"] = {
                "current_load": system_load,
                "status": self._determine_health_status(system_load)
            }
            
            return summary
    
    def _calculate_success_rate(self) -> float:
        """Calculate job success rate"""
        completed = self.get_metric("total_jobs_completed")
        failed = self.get_metric("total_jobs_failed")
        
        completed_count = completed.get_latest_value() if completed else 0
        failed_count = failed.get_latest_value() if failed else 0
        
        # Handle None values
        completed_count = completed_count or 0
        failed_count = failed_count or 0
        
        total = completed_count + failed_count
        return (completed_count / total * 100) if total > 0 else 100.0
    
    def _determine_health_status(self, system_load: float) -> str:
        """Determine system health status based on load"""
        if system_load < 50:
            return HealthStatus.HEALTHY.value
        elif system_load < 75:
            return HealthStatus.DEGRADED.value
        elif system_load < 90:
            return HealthStatus.UNHEALTHY.value
        else:
            return HealthStatus.CRITICAL.value


class AlertManager:
    """
    Manages threshold-based alerting system.
    
    Monitors metrics and triggers alerts when thresholds are exceeded,
    with support for alert suppression, escalation, and notification.
    """
    
    def __init__(self, metrics: ResourceMetrics):
        self.metrics = metrics
        self._rules: Dict[str, ThresholdRule] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._lock = threading.RLock()
        
        # Alert state tracking
        self._rule_violations: Dict[str, datetime] = {}
        
        # Notification callbacks
        self._notification_handlers: List[Callable[[Alert], None]] = []
        
        # Initialize default rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default alerting rules"""
        default_rules = [
            ThresholdRule("high_cpu_usage", "cpu_utilization_percent", "gt", 90.0, 
                         AlertSeverity.WARNING, 300, "High CPU utilization"),
            ThresholdRule("critical_cpu_usage", "cpu_utilization_percent", "gt", 95.0,
                         AlertSeverity.CRITICAL, 60, "Critical CPU utilization"),
            ThresholdRule("high_memory_usage", "memory_utilization_percent", "gt", 85.0,
                         AlertSeverity.WARNING, 300, "High memory utilization"),
            ThresholdRule("critical_memory_usage", "memory_utilization_percent", "gt", 95.0,
                         AlertSeverity.CRITICAL, 60, "Critical memory utilization"),
            ThresholdRule("high_job_failure_rate", "total_jobs_failed", "gt", 10.0,
                         AlertSeverity.ERROR, 600, "High job failure rate"),
            ThresholdRule("system_overload", "system_load", "gt", 90.0,
                         AlertSeverity.CRITICAL, 120, "System overload detected")
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
    
    def add_rule(self, rule: ThresholdRule):
        """Add an alerting rule"""
        with self._lock:
            self._rules[rule.name] = rule
            logger.info(f"Added alerting rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove an alerting rule"""
        with self._lock:
            if rule_name in self._rules:
                del self._rules[rule_name]
                # Clean up any violations for this rule
                if rule_name in self._rule_violations:
                    del self._rule_violations[rule_name]
                logger.info(f"Removed alerting rule: {rule_name}")
                return True
            return False
    
    def get_rule(self, rule_name: str) -> Optional[ThresholdRule]:
        """Get a specific rule"""
        with self._lock:
            return self._rules.get(rule_name)
    
    def get_all_rules(self) -> Dict[str, ThresholdRule]:
        """Get all alerting rules"""
        with self._lock:
            return self._rules.copy()
    
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add notification handler for alerts"""
        with self._lock:
            self._notification_handlers.append(handler)
    
    def check_thresholds(self):
        """Check all threshold rules against current metrics"""
        with self._lock:
            current_time = datetime.now(timezone.utc)
            
            for rule_name, rule in self._rules.items():
                if not rule.enabled:
                    continue
                
                try:
                    metric = self.metrics.get_metric(rule.metric_name)
                    if not metric:
                        continue
                    
                    current_value = metric.get_latest_value()
                    if current_value is None:
                        continue
                    
                    # Check if threshold is violated
                    if rule.evaluate(current_value):
                        # Track violation start time
                        if rule_name not in self._rule_violations:
                            self._rule_violations[rule_name] = current_time
                        
                        # Check if violation has persisted long enough
                        violation_duration = current_time - self._rule_violations[rule_name]
                        if violation_duration.total_seconds() >= rule.duration_seconds:
                            self._trigger_alert(rule, current_value, current_time)
                    else:
                        # Clear violation tracking
                        if rule_name in self._rule_violations:
                            del self._rule_violations[rule_name]
                        
                        # Resolve active alert if exists
                        self._resolve_alert(rule_name, current_time)
                        
                except Exception as e:
                    logger.error(f"Error checking threshold rule {rule_name}: {e}")
    
    def _trigger_alert(self, rule: ThresholdRule, current_value: float, timestamp: datetime):
        """Trigger an alert"""
        alert_id = f"{rule.name}_{int(timestamp.timestamp())}"
        
        # Don't create duplicate alerts
        if rule.name in self._active_alerts:
            return
        
        alert = Alert(
            id=alert_id,
            name=rule.name,
            severity=rule.severity,
            message=f"{rule.description}: {current_value} {rule.operator} {rule.threshold}",
            metric_name=rule.metric_name,
            threshold_value=rule.threshold,
            current_value=current_value,
            triggered_at=timestamp,
            metadata={"rule": rule.name}
        )
        
        self._active_alerts[rule.name] = alert
        self._alert_history.append(alert)
        
        logger.warning(f"Alert triggered: {alert.message}")
        
        # Send notifications
        for handler in self._notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in notification handler: {e}")
    
    def _resolve_alert(self, rule_name: str, timestamp: datetime):
        """Resolve an active alert"""
        if rule_name in self._active_alerts:
            alert = self._active_alerts[rule_name]
            alert.resolved_at = timestamp
            del self._active_alerts[rule_name]
            
            logger.info(f"Alert resolved: {rule_name}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        with self._lock:
            return list(self._active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified hours"""
        with self._lock:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            return [alert for alert in self._alert_history if alert.triggered_at >= cutoff]
    
    def acknowledge_alert(self, rule_name: str, acknowledged_by: str) -> bool:
        """Acknowledge an active alert"""
        with self._lock:
            if rule_name in self._active_alerts:
                alert = self._active_alerts[rule_name]
                alert.metadata["acknowledged"] = True
                alert.metadata["acknowledged_by"] = acknowledged_by
                alert.metadata["acknowledged_at"] = datetime.now(timezone.utc)
                return True
            return False
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert status"""
        with self._lock:
            active_by_severity = defaultdict(int)
            for alert in self._active_alerts.values():
                active_by_severity[alert.severity.value] += 1
            
            return {
                "active_alerts": len(self._active_alerts),
                "active_by_severity": dict(active_by_severity),
                "total_rules": len(self._rules),
                "enabled_rules": sum(1 for rule in self._rules.values() if rule.enabled),
                "recent_alerts": len(self.get_alert_history(24))
            }


class SystemHealthMonitor:
    """
    Monitors overall system health and performance.
    
    Provides comprehensive monitoring of system components, resource health,
    job processing health, and overall system stability.
    """
    
    def __init__(self, resource_pool: ResourcePool, metrics: ResourceMetrics, alert_manager: AlertManager):
        self.resource_pool = resource_pool
        self.metrics = metrics
        self.alert_manager = alert_manager
        
        self._lock = threading.RLock()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_interval = 30.0  # seconds
        
        # Health check functions
        self._health_checks: Dict[str, Callable[[], Tuple[HealthStatus, str]]] = {}
        self._health_status_history: deque = deque(maxlen=100)
        
        # Initialize default health checks
        self._initialize_health_checks()
    
    def _initialize_health_checks(self):
        """Initialize default health check functions"""
        self._health_checks = {
            "resource_availability": self._check_resource_availability,
            "job_processing": self._check_job_processing_health,
            "alert_status": self._check_alert_status,
            "system_stability": self._check_system_stability,
            "memory_leaks": self._check_memory_leaks
        }
    
    def add_health_check(self, name: str, check_func: Callable[[], Tuple[HealthStatus, str]]):
        """Add a custom health check function"""
        with self._lock:
            self._health_checks[name] = check_func
            logger.info(f"Added health check: {name}")
    
    def remove_health_check(self, name: str) -> bool:
        """Remove a health check function"""
        with self._lock:
            if name in self._health_checks:
                del self._health_checks[name]
                logger.info(f"Removed health check: {name}")
                return True
            return False
    
    def start_monitoring(self):
        """Start the health monitoring loop"""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                name="SystemHealthMonitor",
                daemon=True
            )
            self._monitor_thread.start()
            logger.info("System health monitoring started")
    
    def stop_monitoring(self):
        """Stop the health monitoring loop"""
        with self._lock:
            if not self._running:
                return
            
            self._running = False
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=5.0)
            
            logger.info("System health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                self._perform_health_checks()
                self.alert_manager.check_thresholds()
                self.metrics.update_resource_metrics()
                time.sleep(self._monitor_interval)
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(self._monitor_interval)
    
    def _perform_health_checks(self):
        """Perform all health checks"""
        with self._lock:
            check_results = {}
            overall_status = HealthStatus.HEALTHY
            
            for check_name, check_func in self._health_checks.items():
                try:
                    status, message = check_func()
                    check_results[check_name] = {
                        "status": status.value,
                        "message": message,
                        "timestamp": datetime.now(timezone.utc)
                    }
                    
                    # Determine overall status (worst case)
                    if status.value == "critical":
                        overall_status = HealthStatus.CRITICAL
                    elif status.value == "unhealthy" and overall_status != HealthStatus.CRITICAL:
                        overall_status = HealthStatus.UNHEALTHY
                    elif status.value == "degraded" and overall_status == HealthStatus.HEALTHY:
                        overall_status = HealthStatus.DEGRADED
                        
                except Exception as e:
                    logger.error(f"Error in health check {check_name}: {e}")
                    check_results[check_name] = {
                        "status": HealthStatus.CRITICAL.value,
                        "message": f"Health check failed: {str(e)}",
                        "timestamp": datetime.now(timezone.utc)
                    }
                    overall_status = HealthStatus.CRITICAL
            
            # Record health status
            health_record = {
                "timestamp": datetime.now(timezone.utc),
                "overall_status": overall_status.value,
                "checks": check_results
            }
            self._health_status_history.append(health_record)
    
    def _check_resource_availability(self) -> Tuple[HealthStatus, str]:
        """Check resource availability health"""
        try:
            utilization = self.resource_pool.get_resource_utilization()
            
            critical_resources = []
            degraded_resources = []
            
            for resource_type, util_info in utilization.items():
                util_percent = util_info["utilization_percent"]
                
                if util_percent > 95:
                    critical_resources.append(f"{resource_type.value}({util_percent:.1f}%)")
                elif util_percent > 85:
                    degraded_resources.append(f"{resource_type.value}({util_percent:.1f}%)")
            
            if critical_resources:
                return HealthStatus.CRITICAL, f"Critical resource usage: {', '.join(critical_resources)}"
            elif degraded_resources:
                return HealthStatus.DEGRADED, f"High resource usage: {', '.join(degraded_resources)}"
            else:
                return HealthStatus.HEALTHY, "Resource availability is healthy"
                
        except Exception as e:
            return HealthStatus.CRITICAL, f"Failed to check resource availability: {str(e)}"
    
    def _check_job_processing_health(self) -> Tuple[HealthStatus, str]:
        """Check job processing health"""
        try:
            completed_metric = self.metrics.get_metric("total_jobs_completed")
            failed_metric = self.metrics.get_metric("total_jobs_failed")
            
            if not completed_metric or not failed_metric:
                return HealthStatus.HEALTHY, "No job processing data available"
            
            completed = completed_metric.get_latest_value() or 0
            failed = failed_metric.get_latest_value() or 0
            
            if completed + failed == 0:
                return HealthStatus.HEALTHY, "No jobs processed yet"
            
            failure_rate = (failed / (completed + failed)) * 100
            
            if failure_rate > 50:
                return HealthStatus.CRITICAL, f"High job failure rate: {failure_rate:.1f}%"
            elif failure_rate > 20:
                return HealthStatus.UNHEALTHY, f"Elevated job failure rate: {failure_rate:.1f}%"
            elif failure_rate > 10:
                return HealthStatus.DEGRADED, f"Some job failures: {failure_rate:.1f}%"
            else:
                return HealthStatus.HEALTHY, f"Job processing healthy: {failure_rate:.1f}% failure rate"
                
        except Exception as e:
            return HealthStatus.CRITICAL, f"Failed to check job processing health: {str(e)}"
    
    def _check_alert_status(self) -> Tuple[HealthStatus, str]:
        """Check alert system health"""
        try:
            active_alerts = self.alert_manager.get_active_alerts()
            
            critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
            error_alerts = [a for a in active_alerts if a.severity == AlertSeverity.ERROR]
            warning_alerts = [a for a in active_alerts if a.severity == AlertSeverity.WARNING]
            
            if critical_alerts:
                return HealthStatus.CRITICAL, f"{len(critical_alerts)} critical alerts active"
            elif error_alerts:
                return HealthStatus.UNHEALTHY, f"{len(error_alerts)} error alerts active"
            elif warning_alerts:
                return HealthStatus.DEGRADED, f"{len(warning_alerts)} warning alerts active"
            else:
                return HealthStatus.HEALTHY, "No active alerts"
                
        except Exception as e:
            return HealthStatus.CRITICAL, f"Failed to check alert status: {str(e)}"
    
    def _check_system_stability(self) -> Tuple[HealthStatus, str]:
        """Check system stability"""
        try:
            # Check if there have been recent system restarts or errors
            if len(self._health_status_history) < 5:
                return HealthStatus.HEALTHY, "Insufficient history for stability check"
            
            recent_records = list(self._health_status_history)[-10:]
            critical_count = sum(1 for r in recent_records if r["overall_status"] == "critical")
            
            if critical_count > 5:
                return HealthStatus.CRITICAL, "System frequently critical"
            elif critical_count > 2:
                return HealthStatus.UNHEALTHY, "System occasionally critical"
            else:
                return HealthStatus.HEALTHY, "System stable"
                
        except Exception as e:
            return HealthStatus.CRITICAL, f"Failed to check system stability: {str(e)}"
    
    def _check_memory_leaks(self) -> Tuple[HealthStatus, str]:
        """Check for potential memory leaks"""
        try:
            memory_metric = self.metrics.get_metric("memory_utilization_percent")
            
            if not memory_metric or len(memory_metric.points) < 10:
                return HealthStatus.HEALTHY, "Insufficient memory data for leak detection"
            
            # Check if memory usage is consistently increasing
            recent_points = list(memory_metric.points)[-10:]
            values = [p.value for p in recent_points]
            
            # Simple trend analysis
            if len(values) >= 5:
                first_half_avg = statistics.mean(values[:len(values)//2])
                second_half_avg = statistics.mean(values[len(values)//2:])
                
                increase = second_half_avg - first_half_avg
                
                if increase > 20:
                    return HealthStatus.CRITICAL, f"Potential memory leak detected: {increase:.1f}% increase"
                elif increase > 10:
                    return HealthStatus.DEGRADED, f"Memory usage trending up: {increase:.1f}% increase"
            
            return HealthStatus.HEALTHY, "Memory usage stable"
            
        except Exception as e:
            return HealthStatus.CRITICAL, f"Failed to check for memory leaks: {str(e)}"
    
    def get_current_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        with self._lock:
            if not self._health_status_history:
                return {
                    "overall_status": HealthStatus.HEALTHY.value,
                    "message": "No health data available",
                    "timestamp": datetime.now(timezone.utc),
                    "checks": {}
                }
            
            return self._health_status_history[-1]
    
    def get_health_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get health status history"""
        with self._lock:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            return [
                record for record in self._health_status_history
                if record["timestamp"] >= cutoff
            ]
    
    def set_monitor_interval(self, interval_seconds: float):
        """Set monitoring interval"""
        with self._lock:
            self._monitor_interval = max(1.0, interval_seconds)
            logger.info(f"Set monitoring interval to {self._monitor_interval} seconds")


class MetricsCollector:
    """
    Aggregates and manages metrics collection from multiple sources.
    
    Provides centralized metrics collection, storage, and querying capabilities
    for the parallel processing framework.
    """
    
    def __init__(self):
        self._metrics_sources: Dict[str, ResourceMetrics] = {}
        self._aggregated_metrics: Dict[str, MetricSeries] = {}
        self._collection_history: deque = deque(maxlen=1000)
        self._lock = threading.RLock()
        
        # Collection configuration
        self._collection_interval = 60.0  # seconds
        self._running = False
        self._collector_thread: Optional[threading.Thread] = None
        
        # Export handlers
        self._export_handlers: List[Callable[[Dict[str, Any]], None]] = []
    
    def add_metrics_source(self, name: str, metrics: ResourceMetrics):
        """Add a metrics source"""
        with self._lock:
            self._metrics_sources[name] = metrics
            logger.info(f"Added metrics source: {name}")
    
    def remove_metrics_source(self, name: str) -> bool:
        """Remove a metrics source"""
        with self._lock:
            if name in self._metrics_sources:
                del self._metrics_sources[name]
                logger.info(f"Removed metrics source: {name}")
                return True
            return False
    
    def add_export_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Add export handler for metrics"""
        with self._lock:
            self._export_handlers.append(handler)
    
    def start_collection(self):
        """Start metrics collection"""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._collector_thread = threading.Thread(
                target=self._collection_loop,
                name="MetricsCollector",
                daemon=True
            )
            self._collector_thread.start()
            logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection"""
        with self._lock:
            if not self._running:
                return
            
            self._running = False
            if self._collector_thread and self._collector_thread.is_alive():
                self._collector_thread.join(timeout=5.0)
            
            logger.info("Metrics collection stopped")
    
    def _collection_loop(self):
        """Main collection loop"""
        while self._running:
            try:
                self._collect_and_aggregate()
                time.sleep(self._collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                time.sleep(self._collection_interval)
    
    def _collect_and_aggregate(self):
        """Collect and aggregate metrics from all sources"""
        with self._lock:
            collection_timestamp = datetime.now(timezone.utc)
            aggregated_data = {
                "timestamp": collection_timestamp,
                "sources": {},
                "aggregated": {}
            }
            
            # Collect from all sources
            for source_name, metrics in self._metrics_sources.items():
                try:
                    source_metrics = {}
                    for metric_name, metric_series in metrics.get_all_metrics().items():
                        latest_value = metric_series.get_latest_value()
                        if latest_value is not None:
                            source_metrics[metric_name] = {
                                "value": latest_value,
                                "unit": metric_series.unit,
                                "type": metric_series.metric_type.value
                            }
                    
                    aggregated_data["sources"][source_name] = source_metrics
                    
                except Exception as e:
                    logger.error(f"Error collecting from source {source_name}: {e}")
            
            # Aggregate common metrics across sources
            self._aggregate_common_metrics(aggregated_data)
            
            # Store collection record
            self._collection_history.append(aggregated_data)
            
            # Export to handlers
            for handler in self._export_handlers:
                try:
                    handler(aggregated_data)
                except Exception as e:
                    logger.error(f"Error in export handler: {e}")
    
    def _aggregate_common_metrics(self, data: Dict[str, Any]):
        """Aggregate common metrics across sources"""
        common_metrics = [
            "cpu_utilization_percent", "memory_utilization_percent",
            "total_jobs_scheduled", "total_jobs_completed", "total_jobs_failed"
        ]
        
        aggregated = {}
        
        for metric_name in common_metrics:
            values = []
            units = set()
            types = set()
            
            for source_data in data["sources"].values():
                if metric_name in source_data:
                    values.append(source_data[metric_name]["value"])
                    units.add(source_data[metric_name]["unit"])
                    types.add(source_data[metric_name]["type"])
            
            if values:
                if len(units) == 1 and len(types) == 1:
                    metric_type = list(types)[0]
                    
                    if metric_type in ["counter", "gauge"]:
                        # Sum counters and gauges
                        aggregated[metric_name] = {
                            "value": sum(values),
                            "unit": list(units)[0],
                            "type": metric_type,
                            "source_count": len(values)
                        }
                    elif metric_type == "histogram":
                        # Average histograms
                        aggregated[metric_name] = {
                            "value": statistics.mean(values),
                            "unit": list(units)[0],
                            "type": metric_type,
                            "source_count": len(values)
                        }
        
        data["aggregated"] = aggregated
    
    def get_latest_collection(self) -> Optional[Dict[str, Any]]:
        """Get the latest metrics collection"""
        with self._lock:
            return self._collection_history[-1] if self._collection_history else None
    
    def get_collection_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get collection history for specified hours"""
        with self._lock:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            return [
                record for record in self._collection_history
                if record["timestamp"] >= cutoff
            ]
    
    def query_metric(self, metric_name: str, source_name: Optional[str] = None,
                    hours: int = 24) -> List[Dict[str, Any]]:
        """Query specific metric over time"""
        with self._lock:
            history = self.get_collection_history(hours)
            results = []
            
            for record in history:
                if source_name:
                    # Query specific source
                    if source_name in record["sources"] and metric_name in record["sources"][source_name]:
                        results.append({
                            "timestamp": record["timestamp"],
                            "value": record["sources"][source_name][metric_name]["value"],
                            "source": source_name
                        })
                else:
                    # Query aggregated metrics
                    if metric_name in record["aggregated"]:
                        results.append({
                            "timestamp": record["timestamp"],
                            "value": record["aggregated"][metric_name]["value"],
                            "source": "aggregated"
                        })
            
            return results
    
    def export_metrics(self, format_type: str = "json") -> Dict[str, Any]:
        """Export current metrics in specified format"""
        with self._lock:
            latest = self.get_latest_collection()
            
            if format_type == "json":
                return latest or {}
            elif format_type == "prometheus":
                return self._format_prometheus_metrics(latest)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
    
    def _format_prometheus_metrics(self, data: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """Format metrics in Prometheus format"""
        if not data:
            return {}
        
        prometheus_lines = []
        
        # Export aggregated metrics
        for metric_name, metric_info in data.get("aggregated", {}).items():
            line = f'{metric_name.replace("_percent", "_ratio")} {metric_info["value"]}'
            prometheus_lines.append(line)
        
        return {"prometheus_format": "\n".join(prometheus_lines)}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        with self._lock:
            return {
                "sources_count": len(self._metrics_sources),
                "collection_history_size": len(self._collection_history),
                "collection_interval": self._collection_interval,
                "running": self._running,
                "export_handlers_count": len(self._export_handlers),
                "latest_collection": self.get_latest_collection()["timestamp"] if self._collection_history else None
            }