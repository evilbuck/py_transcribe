"""
Tests for parallel framework monitoring and metrics system.
"""

import pytest
import time
import threading
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock

from parallel_framework.resource_pool import ResourcePool, ResourceType, ResourceSpec
from parallel_framework.monitoring import (
    MetricType, AlertSeverity, HealthStatus, MetricPoint, MetricSeries,
    Alert, ThresholdRule, ResourceMetrics, AlertManager, SystemHealthMonitor,
    MetricsCollector
)


class TestMetricType:
    """Test MetricType enum"""
    
    def test_metric_types(self):
        """Test all metric types are available"""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.TIMER.value == "timer"
        assert MetricType.RATE.value == "rate"


class TestAlertSeverity:
    """Test AlertSeverity enum"""
    
    def test_alert_severities(self):
        """Test all alert severities are available"""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"


class TestHealthStatus:
    """Test HealthStatus enum"""
    
    def test_health_statuses(self):
        """Test all health statuses are available"""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.CRITICAL.value == "critical"


class TestMetricPoint:
    """Test MetricPoint dataclass"""
    
    def test_metric_point_creation(self):
        """Test creating metric point"""
        timestamp = datetime.now(timezone.utc)
        point = MetricPoint(
            timestamp=timestamp,
            value=42.5,
            labels={"host": "server1"},
            metadata={"source": "test"}
        )
        
        assert point.timestamp == timestamp
        assert point.value == 42.5
        assert point.labels == {"host": "server1"}
        assert point.metadata == {"source": "test"}
    
    def test_metric_point_defaults(self):
        """Test metric point with default values"""
        timestamp = datetime.now(timezone.utc)
        point = MetricPoint(timestamp=timestamp, value=10.0)
        
        assert point.labels == {}
        assert point.metadata == {}


class TestMetricSeries:
    """Test MetricSeries class"""
    
    def test_metric_series_creation(self):
        """Test creating metric series"""
        series = MetricSeries(
            name="cpu_usage",
            metric_type=MetricType.GAUGE,
            unit="percent",
            description="CPU usage percentage",
            labels={"instance": "web1"}
        )
        
        assert series.name == "cpu_usage"
        assert series.metric_type == MetricType.GAUGE
        assert series.unit == "percent"
        assert series.description == "CPU usage percentage"
        assert series.labels == {"instance": "web1"}
        assert len(series.points) == 0
    
    def test_add_point(self):
        """Test adding metric points"""
        series = MetricSeries("test_metric", MetricType.COUNTER, "count", "Test metric")
        
        series.add_point(10.0, {"tag": "value"}, {"info": "test"})
        series.add_point(20.0)
        
        assert len(series.points) == 2
        assert series.points[0].value == 10.0
        assert series.points[0].labels == {"tag": "value"}
        assert series.points[0].metadata == {"info": "test"}
        assert series.points[1].value == 20.0
        assert series.points[1].labels == {}
    
    def test_get_latest_value(self):
        """Test getting latest metric value"""
        series = MetricSeries("test_metric", MetricType.GAUGE, "count", "Test metric")
        
        # No points initially
        assert series.get_latest_value() is None
        
        # Add points
        series.add_point(10.0)
        assert series.get_latest_value() == 10.0
        
        series.add_point(20.0)
        assert series.get_latest_value() == 20.0
    
    def test_get_average(self):
        """Test getting average metric value"""
        series = MetricSeries("test_metric", MetricType.GAUGE, "count", "Test metric")
        
        # No points
        assert series.get_average() is None
        
        # Add points
        series.add_point(10.0)
        series.add_point(20.0)
        series.add_point(30.0)
        
        assert series.get_average() == 20.0
    
    def test_get_average_with_duration(self):
        """Test getting average with duration filter"""
        series = MetricSeries("test_metric", MetricType.GAUGE, "count", "Test metric")
        
        # Add old points (should be filtered out)
        old_point = MetricPoint(
            timestamp=datetime.now(timezone.utc) - timedelta(hours=2),
            value=5.0
        )
        series.points.append(old_point)
        
        # Add recent points
        series.add_point(10.0)
        series.add_point(20.0)
        
        # Average of recent points only (last hour)
        avg = series.get_average(duration_seconds=3600)
        assert avg == 15.0  # (10 + 20) / 2
    
    def test_get_percentile(self):
        """Test getting percentile values"""
        series = MetricSeries("test_metric", MetricType.GAUGE, "count", "Test metric")
        
        # No points
        assert series.get_percentile(50) is None
        
        # Single point
        series.add_point(10.0)
        assert series.get_percentile(50) == 10.0
        
        # Multiple points
        for value in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            series.add_point(float(value))
        
        # Test different percentiles
        p50 = series.get_percentile(50)
        p90 = series.get_percentile(90)
        
        assert p50 is not None
        assert p90 is not None
        assert p90 >= p50  # 90th percentile should be >= 50th percentile


class TestAlert:
    """Test Alert dataclass"""
    
    def test_alert_creation(self):
        """Test creating alert"""
        triggered_at = datetime.now(timezone.utc)
        alert = Alert(
            id="alert1",
            name="high_cpu",
            severity=AlertSeverity.WARNING,
            message="CPU usage is high",
            metric_name="cpu_usage",
            threshold_value=80.0,
            current_value=85.0,
            triggered_at=triggered_at,
            metadata={"instance": "web1"}
        )
        
        assert alert.id == "alert1"
        assert alert.name == "high_cpu"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.message == "CPU usage is high"
        assert alert.metric_name == "cpu_usage"
        assert alert.threshold_value == 80.0
        assert alert.current_value == 85.0
        assert alert.triggered_at == triggered_at
        assert alert.resolved_at is None
        assert alert.metadata == {"instance": "web1"}
    
    def test_alert_is_active(self):
        """Test alert active status"""
        alert = Alert(
            id="alert1", name="test", severity=AlertSeverity.INFO,
            message="test", metric_name="test", threshold_value=1.0,
            current_value=2.0, triggered_at=datetime.now(timezone.utc)
        )
        
        # Active initially
        assert alert.is_active
        
        # Not active after resolution
        alert.resolved_at = datetime.now(timezone.utc)
        assert not alert.is_active
    
    def test_alert_duration(self):
        """Test alert duration calculation"""
        triggered_at = datetime.now(timezone.utc)
        alert = Alert(
            id="alert1", name="test", severity=AlertSeverity.INFO,
            message="test", metric_name="test", threshold_value=1.0,
            current_value=2.0, triggered_at=triggered_at
        )
        
        # Duration for active alert
        duration = alert.duration
        assert duration.total_seconds() >= 0
        
        # Duration for resolved alert
        resolved_at = triggered_at + timedelta(minutes=5)
        alert.resolved_at = resolved_at
        duration = alert.duration
        assert duration == timedelta(minutes=5)


class TestThresholdRule:
    """Test ThresholdRule dataclass"""
    
    def test_threshold_rule_creation(self):
        """Test creating threshold rule"""
        rule = ThresholdRule(
            name="high_cpu",
            metric_name="cpu_usage",
            operator="gt",
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            duration_seconds=300,
            description="High CPU usage detected"
        )
        
        assert rule.name == "high_cpu"
        assert rule.metric_name == "cpu_usage"
        assert rule.operator == "gt"
        assert rule.threshold == 80.0
        assert rule.severity == AlertSeverity.WARNING
        assert rule.duration_seconds == 300
        assert rule.description == "High CPU usage detected"
        assert rule.enabled
    
    def test_threshold_rule_evaluate(self):
        """Test threshold rule evaluation"""
        rule = ThresholdRule("test", "metric", "gt", 50.0, AlertSeverity.WARNING)
        
        # Test greater than
        assert rule.evaluate(60.0)
        assert not rule.evaluate(40.0)
        assert not rule.evaluate(50.0)
        
        # Test less than
        rule.operator = "lt"
        assert rule.evaluate(40.0)
        assert not rule.evaluate(60.0)
        
        # Test disabled rule
        rule.enabled = False
        assert not rule.evaluate(40.0)
    
    def test_threshold_rule_operators(self):
        """Test all threshold operators"""
        rule = ThresholdRule("test", "metric", "eq", 50.0, AlertSeverity.INFO)
        
        # Test all operators
        operators_tests = [
            ("gt", 60.0, True),
            ("gt", 40.0, False),
            ("lt", 40.0, True),
            ("lt", 60.0, False),
            ("gte", 50.0, True),
            ("gte", 40.0, False),
            ("lte", 50.0, True),
            ("lte", 60.0, False),
            ("eq", 50.0, True),
            ("eq", 40.0, False),
            ("ne", 40.0, True),
            ("ne", 50.0, False)
        ]
        
        for operator, value, expected in operators_tests:
            rule.operator = operator
            assert rule.evaluate(value) == expected


class TestResourceMetrics:
    """Test ResourceMetrics class"""
    
    def setup_method(self):
        """Setup for each test"""
        self.resource_pool = ResourcePool(auto_detect=False)
        self.resource_pool.set_capacity(ResourceType.CPU, 8, "cores")
        self.resource_pool.set_capacity(ResourceType.MEMORY, 16, "GB")
        
        self.metrics = ResourceMetrics(self.resource_pool)
    
    def test_metrics_creation(self):
        """Test creating resource metrics"""
        assert self.metrics.resource_pool is self.resource_pool
        assert len(self.metrics._metrics) > 0
        
        # Check core metrics exist
        assert "cpu_utilization_percent" in self.metrics._metrics
        assert "memory_utilization_percent" in self.metrics._metrics
        assert "total_jobs_scheduled" in self.metrics._metrics
    
    def test_create_metric(self):
        """Test creating custom metric"""
        metric = self.metrics._create_metric(
            "custom_metric", MetricType.COUNTER, "count", "Custom test metric"
        )
        
        assert metric.name == "custom_metric"
        assert metric.metric_type == MetricType.COUNTER
        assert metric.unit == "count"
        assert metric.description == "Custom test metric"
        assert "custom_metric" in self.metrics._metrics
    
    def test_record_metric(self):
        """Test recording metric values"""
        self.metrics.record_metric("cpu_utilization_percent", 75.0, {"host": "server1"})
        
        metric = self.metrics.get_metric("cpu_utilization_percent")
        assert metric.get_latest_value() == 75.0
        assert len(metric.points) == 1
        assert metric.points[0].labels == {"host": "server1"}
    
    def test_record_unknown_metric(self):
        """Test recording unknown metric"""
        # Should not crash but should log warning
        self.metrics.record_metric("unknown_metric", 42.0)
        
        # Metric should not exist
        assert self.metrics.get_metric("unknown_metric") is None
    
    def test_get_metric(self):
        """Test getting metric by name"""
        # Existing metric
        metric = self.metrics.get_metric("cpu_utilization_percent")
        assert metric is not None
        assert metric.name == "cpu_utilization_percent"
        
        # Non-existing metric
        assert self.metrics.get_metric("nonexistent") is None
    
    def test_get_all_metrics(self):
        """Test getting all metrics"""
        all_metrics = self.metrics.get_all_metrics()
        assert isinstance(all_metrics, dict)
        assert len(all_metrics) > 0
        assert "cpu_utilization_percent" in all_metrics
    
    def test_update_resource_metrics(self):
        """Test updating resource metrics"""
        # Mock resource utilization
        with patch.object(self.resource_pool, 'get_resource_utilization') as mock_util:
            mock_util.return_value = {
                ResourceType.CPU: {
                    "utilization_percent": 50.0,
                    "allocated_amount": 4.0,
                    "available_amount": 4.0,
                    "allocation_count": 2
                },
                ResourceType.MEMORY: {
                    "utilization_percent": 25.0,
                    "allocated_amount": 4.0,
                    "available_amount": 12.0,
                    "allocation_count": 1
                }
            }
            
            self.metrics.update_resource_metrics()
            
            # Check metrics were updated
            cpu_util = self.metrics.get_metric("cpu_utilization_percent")
            assert cpu_util.get_latest_value() == 50.0
            
            memory_util = self.metrics.get_metric("memory_utilization_percent")
            assert memory_util.get_latest_value() == 25.0
    
    def test_record_job_events(self):
        """Test recording job events"""
        # Record job scheduled
        self.metrics.record_job_scheduled("job1", {"priority": "high"})
        scheduled_metric = self.metrics.get_metric("total_jobs_scheduled")
        assert scheduled_metric.get_latest_value() == 1
        
        # Record job completed
        self.metrics.record_job_completed("job1", 120.0, {"status": "success"})
        completed_metric = self.metrics.get_metric("total_jobs_completed")
        assert completed_metric.get_latest_value() == 1
        
        duration_metric = self.metrics.get_metric("average_job_duration")
        assert duration_metric.get_latest_value() == 120.0
        
        # Record job failed
        self.metrics.record_job_failed("job2", "timeout", {"retry": False})
        failed_metric = self.metrics.get_metric("total_jobs_failed")
        assert failed_metric.get_latest_value() == 1
    
    def test_update_queue_metrics(self):
        """Test updating queue metrics"""
        self.metrics.update_queue_metrics(queue_depth=5, active_jobs=3)
        
        queue_metric = self.metrics.get_metric("queue_depth")
        assert queue_metric.get_latest_value() == 5
        
        active_metric = self.metrics.get_metric("active_jobs")
        assert active_metric.get_latest_value() == 3
    
    def test_calculate_system_load(self):
        """Test system load calculation"""
        with patch.object(self.resource_pool, 'get_resource_utilization') as mock_util:
            mock_util.return_value = {
                ResourceType.CPU: {"utilization_percent": 80.0},
                ResourceType.MEMORY: {"utilization_percent": 60.0},
                ResourceType.GPU: {"utilization_percent": 40.0},
                ResourceType.DISK: {"utilization_percent": 20.0}
            }
            
            load = self.metrics.calculate_system_load()
            
            # Weighted average: 0.4*80 + 0.4*60 + 0.15*40 + 0.05*20 = 63
            expected = (0.4 * 80 + 0.4 * 60 + 0.15 * 40 + 0.05 * 20)
            assert abs(load - expected) < 0.1
    
    def test_get_performance_summary(self):
        """Test getting performance summary"""
        # Add some metrics data
        self.metrics.record_metric("cpu_utilization_percent", 75.0)
        self.metrics.record_job_scheduled("job1")
        self.metrics.record_job_completed("job1", 60.0)
        
        summary = self.metrics.get_performance_summary(3600)
        
        assert "duration_seconds" in summary
        assert "timestamp" in summary
        assert "resource_utilization" in summary
        assert "job_statistics" in summary
        assert "system_health" in summary
        
        # Check resource utilization data
        assert "cpu" in summary["resource_utilization"]
        
        # Check job statistics
        job_stats = summary["job_statistics"]
        assert job_stats["scheduled"] == 1
        assert job_stats["completed"] == 1
        assert job_stats["average_duration"] == 60.0


class TestAlertManager:
    """Test AlertManager class"""
    
    def setup_method(self):
        """Setup for each test"""
        self.resource_pool = ResourcePool(auto_detect=False)
        self.metrics = ResourceMetrics(self.resource_pool)
        self.alert_manager = AlertManager(self.metrics)
    
    def test_alert_manager_creation(self):
        """Test creating alert manager"""
        assert self.alert_manager.metrics is self.metrics
        assert len(self.alert_manager._rules) > 0  # Default rules
        assert len(self.alert_manager._active_alerts) == 0
    
    def test_add_remove_rule(self):
        """Test adding and removing alerting rules"""
        rule = ThresholdRule(
            "test_rule", "test_metric", "gt", 100.0, AlertSeverity.WARNING
        )
        
        self.alert_manager.add_rule(rule)
        assert "test_rule" in self.alert_manager._rules
        assert self.alert_manager.get_rule("test_rule") == rule
        
        # Remove rule
        success = self.alert_manager.remove_rule("test_rule")
        assert success
        assert "test_rule" not in self.alert_manager._rules
        assert self.alert_manager.get_rule("test_rule") is None
        
        # Remove non-existent rule
        success = self.alert_manager.remove_rule("nonexistent")
        assert not success
    
    def test_get_all_rules(self):
        """Test getting all rules"""
        rules = self.alert_manager.get_all_rules()
        assert isinstance(rules, dict)
        assert len(rules) > 0
    
    def test_notification_handlers(self):
        """Test notification handlers"""
        notifications = []
        
        def handler(alert):
            notifications.append(alert)
        
        self.alert_manager.add_notification_handler(handler)
        
        # Create metric and trigger alert
        self.metrics._create_metric("test_metric", MetricType.GAUGE, "percent", "Test")
        self.metrics.record_metric("test_metric", 95.0)
        
        rule = ThresholdRule(
            "test_alert", "test_metric", "gt", 90.0, AlertSeverity.WARNING, 0
        )
        self.alert_manager.add_rule(rule)
        
        # Check thresholds
        self.alert_manager.check_thresholds()
        
        # Should have triggered notification
        assert len(notifications) == 1
        assert notifications[0].name == "test_alert"
    
    def test_check_thresholds_violation(self):
        """Test threshold checking with violations"""
        # Create test metric
        self.metrics._create_metric("test_metric", MetricType.GAUGE, "percent", "Test")
        self.metrics.record_metric("test_metric", 95.0)
        
        # Add rule with no duration requirement
        rule = ThresholdRule(
            "test_alert", "test_metric", "gt", 90.0, AlertSeverity.WARNING, 0
        )
        self.alert_manager.add_rule(rule)
        
        # Check thresholds - should trigger alert
        self.alert_manager.check_thresholds()
        
        active_alerts = self.alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].name == "test_alert"
        assert active_alerts[0].current_value == 95.0
    
    def test_check_thresholds_resolution(self):
        """Test threshold checking with alert resolution"""
        # Create test metric and trigger alert
        self.metrics._create_metric("test_metric", MetricType.GAUGE, "percent", "Test")
        self.metrics.record_metric("test_metric", 95.0)
        
        rule = ThresholdRule(
            "test_alert", "test_metric", "gt", 90.0, AlertSeverity.WARNING, 0
        )
        self.alert_manager.add_rule(rule)
        
        # Trigger alert
        self.alert_manager.check_thresholds()
        assert len(self.alert_manager.get_active_alerts()) == 1
        
        # Update metric to resolve condition
        self.metrics.record_metric("test_metric", 85.0)
        
        # Check thresholds - should resolve alert
        self.alert_manager.check_thresholds()
        assert len(self.alert_manager.get_active_alerts()) == 0
    
    def test_check_thresholds_duration(self):
        """Test threshold checking with duration requirement"""
        self.metrics._create_metric("test_metric", MetricType.GAUGE, "percent", "Test")
        self.metrics.record_metric("test_metric", 95.0)
        
        # Rule requires 60 seconds violation
        rule = ThresholdRule(
            "test_alert", "test_metric", "gt", 90.0, AlertSeverity.WARNING, 60
        )
        self.alert_manager.add_rule(rule)
        
        # First check - should not trigger (duration not met)
        self.alert_manager.check_thresholds()
        assert len(self.alert_manager.get_active_alerts()) == 0
        
        # Simulate time passing by updating violation time
        rule_name = "test_alert"
        past_time = datetime.now(timezone.utc) - timedelta(seconds=61)
        self.alert_manager._rule_violations[rule_name] = past_time
        
        # Check again - should trigger now
        self.alert_manager.check_thresholds()
        assert len(self.alert_manager.get_active_alerts()) == 1
    
    def test_acknowledge_alert(self):
        """Test acknowledging alerts"""
        # Trigger alert
        self.metrics._create_metric("test_metric", MetricType.GAUGE, "percent", "Test")
        self.metrics.record_metric("test_metric", 95.0)
        
        rule = ThresholdRule(
            "test_alert", "test_metric", "gt", 90.0, AlertSeverity.WARNING, 0
        )
        self.alert_manager.add_rule(rule)
        self.alert_manager.check_thresholds()
        
        # Acknowledge alert
        success = self.alert_manager.acknowledge_alert("test_alert", "admin")
        assert success
        
        active_alerts = self.alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].metadata["acknowledged"]
        assert active_alerts[0].metadata["acknowledged_by"] == "admin"
        
        # Acknowledge non-existent alert
        success = self.alert_manager.acknowledge_alert("nonexistent", "admin")
        assert not success
    
    def test_get_alert_history(self):
        """Test getting alert history"""
        # Trigger and resolve alert
        self.metrics._create_metric("test_metric", MetricType.GAUGE, "percent", "Test")
        self.metrics.record_metric("test_metric", 95.0)
        
        rule = ThresholdRule(
            "test_alert", "test_metric", "gt", 90.0, AlertSeverity.WARNING, 0
        )
        self.alert_manager.add_rule(rule)
        self.alert_manager.check_thresholds()
        
        # Update metric to resolve
        self.metrics.record_metric("test_metric", 85.0)
        self.alert_manager.check_thresholds()
        
        history = self.alert_manager.get_alert_history(24)
        assert len(history) == 1
        assert not history[0].is_active
    
    def test_get_alert_summary(self):
        """Test getting alert summary"""
        summary = self.alert_manager.get_alert_summary()
        
        assert "active_alerts" in summary
        assert "active_by_severity" in summary
        assert "total_rules" in summary
        assert "enabled_rules" in summary
        assert "recent_alerts" in summary
        
        assert summary["active_alerts"] == 0
        assert summary["total_rules"] > 0


class TestSystemHealthMonitor:
    """Test SystemHealthMonitor class"""
    
    def setup_method(self):
        """Setup for each test"""
        self.resource_pool = ResourcePool(auto_detect=False)
        self.metrics = ResourceMetrics(self.resource_pool)
        self.alert_manager = AlertManager(self.metrics)
        self.health_monitor = SystemHealthMonitor(
            self.resource_pool, self.metrics, self.alert_manager
        )
    
    def test_health_monitor_creation(self):
        """Test creating health monitor"""
        assert self.health_monitor.resource_pool is self.resource_pool
        assert self.health_monitor.metrics is self.metrics
        assert self.health_monitor.alert_manager is self.alert_manager
        assert not self.health_monitor._running
        assert len(self.health_monitor._health_checks) > 0
    
    def test_add_remove_health_check(self):
        """Test adding and removing health checks"""
        def custom_check():
            return HealthStatus.HEALTHY, "All good"
        
        self.health_monitor.add_health_check("custom_check", custom_check)
        assert "custom_check" in self.health_monitor._health_checks
        
        success = self.health_monitor.remove_health_check("custom_check")
        assert success
        assert "custom_check" not in self.health_monitor._health_checks
        
        success = self.health_monitor.remove_health_check("nonexistent")
        assert not success
    
    def test_health_monitor_lifecycle(self):
        """Test health monitor start/stop"""
        assert not self.health_monitor._running
        
        self.health_monitor.start_monitoring()
        assert self.health_monitor._running
        assert self.health_monitor._monitor_thread is not None
        
        time.sleep(0.1)  # Let it run briefly
        
        self.health_monitor.stop_monitoring()
        assert not self.health_monitor._running
    
    def test_perform_health_checks(self):
        """Test performing health checks"""
        # Mock health checks
        def healthy_check():
            return HealthStatus.HEALTHY, "Everything is fine"
        
        def degraded_check():
            return HealthStatus.DEGRADED, "Performance degraded"
        
        self.health_monitor._health_checks = {
            "healthy": healthy_check,
            "degraded": degraded_check
        }
        
        self.health_monitor._perform_health_checks()
        
        assert len(self.health_monitor._health_status_history) == 1
        status = self.health_monitor._health_status_history[0]
        
        assert status["overall_status"] == HealthStatus.DEGRADED.value
        assert "healthy" in status["checks"]
        assert "degraded" in status["checks"]
        assert status["checks"]["healthy"]["status"] == "healthy"
        assert status["checks"]["degraded"]["status"] == "degraded"
    
    def test_check_resource_availability(self):
        """Test resource availability health check"""
        with patch.object(self.resource_pool, 'get_resource_utilization') as mock_util:
            # Healthy resource usage
            mock_util.return_value = {
                ResourceType.CPU: {"utilization_percent": 50.0},
                ResourceType.MEMORY: {"utilization_percent": 60.0}
            }
            
            status, message = self.health_monitor._check_resource_availability()
            assert status == HealthStatus.HEALTHY
            assert "healthy" in message.lower()
            
            # High resource usage
            mock_util.return_value = {
                ResourceType.CPU: {"utilization_percent": 90.0},
                ResourceType.MEMORY: {"utilization_percent": 60.0}
            }
            
            status, message = self.health_monitor._check_resource_availability()
            assert status == HealthStatus.DEGRADED
            assert "high resource usage" in message.lower()
            
            # Critical resource usage
            mock_util.return_value = {
                ResourceType.CPU: {"utilization_percent": 98.0}
            }
            
            status, message = self.health_monitor._check_resource_availability()
            assert status == HealthStatus.CRITICAL
            assert "critical resource usage" in message.lower()
    
    def test_check_job_processing_health(self):
        """Test job processing health check"""
        # No job data
        status, message = self.health_monitor._check_job_processing_health()
        assert status == HealthStatus.HEALTHY
        
        # Good job processing
        self.metrics.record_metric("total_jobs_completed", 90.0)
        self.metrics.record_metric("total_jobs_failed", 10.0)
        
        status, message = self.health_monitor._check_job_processing_health()
        assert status == HealthStatus.HEALTHY
        assert "10.0%" in message
        
        # High failure rate
        self.metrics.record_metric("total_jobs_completed", 10.0)
        self.metrics.record_metric("total_jobs_failed", 90.0)
        
        status, message = self.health_monitor._check_job_processing_health()
        assert status == HealthStatus.CRITICAL
        assert "90.0%" in message
    
    def test_check_alert_status(self):
        """Test alert status health check"""
        # No alerts
        status, message = self.health_monitor._check_alert_status()
        assert status == HealthStatus.HEALTHY
        assert "no active alerts" in message.lower()
        
        # Mock active alerts
        critical_alert = Alert(
            "alert1", "critical_alert", AlertSeverity.CRITICAL, "Critical issue",
            "test_metric", 100.0, 110.0, datetime.now(timezone.utc)
        )
        
        self.alert_manager._active_alerts["critical_alert"] = critical_alert
        
        status, message = self.health_monitor._check_alert_status()
        assert status == HealthStatus.CRITICAL
        assert "critical alerts active" in message.lower()
    
    def test_get_current_health_status(self):
        """Test getting current health status"""
        # No history
        status = self.health_monitor.get_current_health_status()
        assert status["overall_status"] == "healthy"
        assert "no health data" in status["message"].lower()
        
        # Perform health checks
        self.health_monitor._perform_health_checks()
        
        status = self.health_monitor.get_current_health_status()
        assert "overall_status" in status
        assert "checks" in status
        assert "timestamp" in status
    
    def test_get_health_history(self):
        """Test getting health history"""
        # Perform some health checks
        self.health_monitor._perform_health_checks()
        time.sleep(0.1)
        self.health_monitor._perform_health_checks()
        
        history = self.health_monitor.get_health_history(24)
        assert len(history) == 2
        
        # Test duration filtering
        history = self.health_monitor.get_health_history(0)  # 0 hours
        assert len(history) == 0
    
    def test_set_monitor_interval(self):
        """Test setting monitor interval"""
        self.health_monitor.set_monitor_interval(10.0)
        assert self.health_monitor._monitor_interval == 10.0
        
        # Test minimum interval
        self.health_monitor.set_monitor_interval(0.5)
        assert self.health_monitor._monitor_interval == 1.0  # Minimum


class TestMetricsCollector:
    """Test MetricsCollector class"""
    
    def setup_method(self):
        """Setup for each test"""
        self.collector = MetricsCollector()
    
    def test_collector_creation(self):
        """Test creating metrics collector"""
        assert len(self.collector._metrics_sources) == 0
        assert len(self.collector._aggregated_metrics) == 0
        assert not self.collector._running
    
    def test_add_remove_metrics_source(self):
        """Test adding and removing metrics sources"""
        resource_pool = ResourcePool(auto_detect=False)
        metrics = ResourceMetrics(resource_pool)
        
        self.collector.add_metrics_source("test_source", metrics)
        assert "test_source" in self.collector._metrics_sources
        assert self.collector._metrics_sources["test_source"] is metrics
        
        success = self.collector.remove_metrics_source("test_source")
        assert success
        assert "test_source" not in self.collector._metrics_sources
        
        success = self.collector.remove_metrics_source("nonexistent")
        assert not success
    
    def test_add_export_handler(self):
        """Test adding export handlers"""
        exports = []
        
        def export_handler(data):
            exports.append(data)
        
        self.collector.add_export_handler(export_handler)
        assert len(self.collector._export_handlers) == 1
    
    def test_collector_lifecycle(self):
        """Test collector start/stop"""
        assert not self.collector._running
        
        self.collector.start_collection()
        assert self.collector._running
        assert self.collector._collector_thread is not None
        
        time.sleep(0.1)
        
        self.collector.stop_collection()
        assert not self.collector._running
    
    def test_collect_and_aggregate(self):
        """Test collecting and aggregating metrics"""
        # Add metrics source
        resource_pool = ResourcePool(auto_detect=False)
        metrics = ResourceMetrics(resource_pool)
        metrics.record_metric("cpu_utilization_percent", 75.0)
        metrics.record_metric("total_jobs_scheduled", 10.0)
        
        self.collector.add_metrics_source("source1", metrics)
        
        # Collect and aggregate
        self.collector._collect_and_aggregate()
        
        # Check collection history
        assert len(self.collector._collection_history) == 1
        collection = self.collector._collection_history[0]
        
        assert "timestamp" in collection
        assert "sources" in collection
        assert "aggregated" in collection
        assert "source1" in collection["sources"]
        
        source_data = collection["sources"]["source1"]
        assert "cpu_utilization_percent" in source_data
        assert source_data["cpu_utilization_percent"]["value"] == 75.0
    
    def test_aggregate_common_metrics(self):
        """Test aggregating common metrics across sources"""
        # Create two metrics sources
        resource_pool1 = ResourcePool(auto_detect=False)
        metrics1 = ResourceMetrics(resource_pool1)
        metrics1.record_metric("cpu_utilization_percent", 50.0)
        metrics1.record_metric("total_jobs_scheduled", 5.0)
        
        resource_pool2 = ResourcePool(auto_detect=False)
        metrics2 = ResourceMetrics(resource_pool2)
        metrics2.record_metric("cpu_utilization_percent", 70.0)
        metrics2.record_metric("total_jobs_scheduled", 3.0)
        
        self.collector.add_metrics_source("source1", metrics1)
        self.collector.add_metrics_source("source2", metrics2)
        
        # Collect and aggregate
        self.collector._collect_and_aggregate()
        
        collection = self.collector.get_latest_collection()
        aggregated = collection["aggregated"]
        
        # Check aggregated values
        assert "cpu_utilization_percent" in aggregated
        assert "total_jobs_scheduled" in aggregated
        
        # CPU utilization should be summed (gauge type)
        assert aggregated["cpu_utilization_percent"]["value"] == 120.0
        assert aggregated["cpu_utilization_percent"]["source_count"] == 2
        
        # Jobs scheduled should be summed (counter type)
        assert aggregated["total_jobs_scheduled"]["value"] == 8.0
    
    def test_get_latest_collection(self):
        """Test getting latest collection"""
        # No collections yet
        assert self.collector.get_latest_collection() is None
        
        # Add source and collect
        resource_pool = ResourcePool(auto_detect=False)
        metrics = ResourceMetrics(resource_pool)
        self.collector.add_metrics_source("test", metrics)
        self.collector._collect_and_aggregate()
        
        latest = self.collector.get_latest_collection()
        assert latest is not None
        assert "timestamp" in latest
    
    def test_get_collection_history(self):
        """Test getting collection history"""
        # Add source and collect multiple times
        resource_pool = ResourcePool(auto_detect=False)
        metrics = ResourceMetrics(resource_pool)
        self.collector.add_metrics_source("test", metrics)
        
        self.collector._collect_and_aggregate()
        time.sleep(0.1)
        self.collector._collect_and_aggregate()
        
        history = self.collector.get_collection_history(24)
        assert len(history) == 2
        
        # Test duration filtering
        history = self.collector.get_collection_history(0)
        assert len(history) == 0
    
    def test_query_metric(self):
        """Test querying specific metrics"""
        # Add source with metrics
        resource_pool = ResourcePool(auto_detect=False)
        metrics = ResourceMetrics(resource_pool)
        metrics.record_metric("cpu_utilization_percent", 60.0)
        
        self.collector.add_metrics_source("test_source", metrics)
        self.collector._collect_and_aggregate()
        
        # Query specific source
        results = self.collector.query_metric("cpu_utilization_percent", "test_source")
        assert len(results) == 1
        assert results[0]["value"] == 60.0
        assert results[0]["source"] == "test_source"
        
        # Query aggregated (no source specified)
        results = self.collector.query_metric("cpu_utilization_percent")
        assert len(results) == 1
        assert results[0]["source"] == "aggregated"
    
    def test_export_metrics(self):
        """Test exporting metrics"""
        # Add source and collect
        resource_pool = ResourcePool(auto_detect=False)
        metrics = ResourceMetrics(resource_pool)
        metrics.record_metric("cpu_utilization_percent", 80.0)
        
        self.collector.add_metrics_source("test", metrics)
        self.collector._collect_and_aggregate()
        
        # Export as JSON
        json_export = self.collector.export_metrics("json")
        assert isinstance(json_export, dict)
        assert "timestamp" in json_export
        
        # Export as Prometheus
        prometheus_export = self.collector.export_metrics("prometheus")
        assert isinstance(prometheus_export, dict)
        assert "prometheus_format" in prometheus_export
        
        # Invalid format
        with pytest.raises(ValueError):
            self.collector.export_metrics("invalid")
    
    def test_get_collection_stats(self):
        """Test getting collection statistics"""
        stats = self.collector.get_collection_stats()
        
        assert "sources_count" in stats
        assert "collection_history_size" in stats
        assert "collection_interval" in stats
        assert "running" in stats
        assert "export_handlers_count" in stats
        
        assert stats["sources_count"] == 0
        assert stats["running"] == False


if __name__ == "__main__":
    pytest.main([__file__])