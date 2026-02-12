"""
System Metrics Monitoring for MLflow Tracking

This module provides utilities to monitor and log system metrics during
model training and evaluation, including:
- CPU utilization
- Memory usage (RAM)
- Disk I/O
- Network I/O
- GPU utilization and memory (NVIDIA via pynvml, Apple Silicon via MPS, or PyTorch CUDA fallback)

The metrics are collected in a background thread and can be logged to MLflow.
"""

import threading
import time
import psutil
from typing import Dict, Optional, List
from pathlib import Path


class SystemMetricsMonitor:
    """
    Monitor system metrics in a background thread.

    Tracks CPU, memory, disk, network, and GPU metrics at regular intervals.
    Supports three GPU backends:
    - NVIDIA GPUs via pynvml (most detailed: utilization, memory, temp, power)
    - NVIDIA GPUs via PyTorch CUDA (fallback: memory, utilization)
    - Apple Silicon GPUs via PyTorch MPS (memory allocation)
    """

    def __init__(self, interval: float = 5.0, enable_gpu: bool = True):
        """
        Initialize the system metrics monitor.

        Args:
            interval: Time in seconds between metric collection (default: 5.0)
            enable_gpu: Whether to attempt GPU monitoring (default: True)
        """
        self.interval = interval
        self.enable_gpu = enable_gpu
        self.running = False
        self.thread = None
        self.metrics_history = []

        # GPU monitoring setup
        self.gpu_available = False
        self.gpu_backend = None  # "pynvml", "cuda", "mps", or None
        self.nvml_initialized = False

        if self.enable_gpu:
            self._init_gpu_monitoring()

        # Baseline network and disk counters
        self.baseline_net_io = psutil.net_io_counters()
        self.baseline_disk_io = psutil.disk_io_counters()
        self.start_time = None

    def _init_gpu_monitoring(self):
        """Initialize GPU monitoring, trying backends in order: pynvml > CUDA > MPS."""
        # Try pynvml first (NVIDIA, most detailed)
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml_initialized = True
            self.gpu_count = pynvml.nvmlDeviceGetCount()
            if self.gpu_count > 0:
                self.gpu_available = True
                self.gpu_backend = "pynvml"
                self.pynvml = pynvml
                print(f"GPU monitoring enabled via pynvml ({self.gpu_count} NVIDIA GPU(s))")
                return
            pynvml.nvmlShutdown()
        except (ImportError, Exception):
            pass

        # Try PyTorch CUDA (NVIDIA, less detailed)
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_available = True
                self.gpu_backend = "cuda"
                self.gpu_count = torch.cuda.device_count()
                print(f"GPU monitoring enabled via PyTorch CUDA ({self.gpu_count} GPU(s))")
                return
        except (ImportError, Exception):
            pass

        # Try PyTorch MPS (Apple Silicon)
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.gpu_available = True
                self.gpu_backend = "mps"
                self.gpu_count = 1
                print("GPU monitoring enabled via PyTorch MPS (Apple Silicon)")
                return
        except (ImportError, Exception):
            pass

        print("GPU monitoring unavailable (no supported GPU backend found)")

    def _collect_cpu_metrics(self) -> Dict[str, float]:
        """Collect CPU utilization metrics."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "cpu_count": psutil.cpu_count(),
        }

    def _collect_memory_metrics(self) -> Dict[str, float]:
        """Collect memory usage metrics."""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        return {
            "memory_used_gb": mem.used / (1024 ** 3),
            "memory_available_gb": mem.available / (1024 ** 3),
            "memory_percent": mem.percent,
            "memory_total_gb": mem.total / (1024 ** 3),
            "swap_used_gb": swap.used / (1024 ** 3),
            "swap_percent": swap.percent,
        }

    def _collect_disk_metrics(self) -> Dict[str, float]:
        """Collect disk I/O metrics."""
        try:
            disk_io = psutil.disk_io_counters()

            # Calculate delta from baseline
            read_mb = (disk_io.read_bytes - self.baseline_disk_io.read_bytes) / (1024 ** 2)
            write_mb = (disk_io.write_bytes - self.baseline_disk_io.write_bytes) / (1024 ** 2)

            return {
                "disk_read_mb": read_mb,
                "disk_write_mb": write_mb,
                "disk_read_count": disk_io.read_count - self.baseline_disk_io.read_count,
                "disk_write_count": disk_io.write_count - self.baseline_disk_io.write_count,
            }
        except Exception as e:
            print(f"Warning: Could not collect disk metrics: {e}")
            return {}

    def _collect_network_metrics(self) -> Dict[str, float]:
        """Collect network I/O metrics."""
        try:
            net_io = psutil.net_io_counters()

            # Calculate delta from baseline
            sent_mb = (net_io.bytes_sent - self.baseline_net_io.bytes_sent) / (1024 ** 2)
            recv_mb = (net_io.bytes_recv - self.baseline_net_io.bytes_recv) / (1024 ** 2)

            return {
                "network_sent_mb": sent_mb,
                "network_recv_mb": recv_mb,
                "network_packets_sent": net_io.packets_sent - self.baseline_net_io.packets_sent,
                "network_packets_recv": net_io.packets_recv - self.baseline_net_io.packets_recv,
            }
        except Exception as e:
            print(f"Warning: Could not collect network metrics: {e}")
            return {}

    def _collect_gpu_metrics(self) -> Dict[str, float]:
        """Collect GPU metrics using the appropriate backend."""
        if not self.gpu_available:
            return {}

        if self.gpu_backend == "pynvml":
            return self._collect_gpu_metrics_pynvml()
        elif self.gpu_backend == "cuda":
            return self._collect_gpu_metrics_cuda()
        elif self.gpu_backend == "mps":
            return self._collect_gpu_metrics_mps()
        return {}

    def _collect_gpu_metrics_pynvml(self) -> Dict[str, float]:
        """Collect GPU metrics via pynvml (NVIDIA, most detailed)."""
        metrics = {}
        try:
            for i in range(self.gpu_count):
                handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)

                # GPU utilization
                utilization = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics[f"gpu_{i}_utilization_percent"] = utilization.gpu
                metrics[f"gpu_{i}_memory_utilization_percent"] = utilization.memory

                # GPU memory
                mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                metrics[f"gpu_{i}_memory_used_gb"] = mem_info.used / (1024 ** 3)
                metrics[f"gpu_{i}_memory_free_gb"] = mem_info.free / (1024 ** 3)
                metrics[f"gpu_{i}_memory_total_gb"] = mem_info.total / (1024 ** 3)
                metrics[f"gpu_{i}_memory_percent"] = (mem_info.used / mem_info.total) * 100

                # GPU temperature
                try:
                    temp = self.pynvml.nvmlDeviceGetTemperature(handle, self.pynvml.NVML_TEMPERATURE_GPU)
                    metrics[f"gpu_{i}_temperature_c"] = temp
                except Exception:
                    pass

                # GPU power usage
                try:
                    power = self.pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                    metrics[f"gpu_{i}_power_watts"] = power
                except Exception:
                    pass

        except Exception as e:
            print(f"Warning: Could not collect pynvml GPU metrics: {e}")
        return metrics

    def _collect_gpu_metrics_cuda(self) -> Dict[str, float]:
        """Collect GPU metrics via PyTorch CUDA (NVIDIA fallback)."""
        import torch
        metrics = {}
        try:
            for i in range(self.gpu_count):
                # Memory metrics
                mem_allocated = torch.cuda.memory_allocated(i)
                mem_reserved = torch.cuda.memory_reserved(i)
                mem_total = torch.cuda.get_device_properties(i).total_mem
                metrics[f"gpu_{i}_memory_allocated_gb"] = mem_allocated / (1024 ** 3)
                metrics[f"gpu_{i}_memory_reserved_gb"] = mem_reserved / (1024 ** 3)
                metrics[f"gpu_{i}_memory_total_gb"] = mem_total / (1024 ** 3)
                metrics[f"gpu_{i}_memory_percent"] = (mem_allocated / mem_total) * 100 if mem_total > 0 else 0

                # Utilization (requires CUDA 11.1+)
                try:
                    utilization = torch.cuda.utilization(i)
                    metrics[f"gpu_{i}_utilization_percent"] = utilization
                except Exception:
                    pass

        except Exception as e:
            print(f"Warning: Could not collect CUDA GPU metrics: {e}")
        return metrics

    def _collect_gpu_metrics_mps(self) -> Dict[str, float]:
        """Collect GPU metrics via PyTorch MPS + macOS ioreg (Apple Silicon)."""
        import torch
        metrics = {}
        try:
            # PyTorch MPS memory metrics
            allocated = torch.mps.current_allocated_memory()
            metrics["gpu_0_memory_allocated_gb"] = allocated / (1024 ** 3)
            metrics["gpu_0_memory_allocated_mb"] = allocated / (1024 ** 2)

            driver_allocated = torch.mps.driver_allocated_memory()
            metrics["gpu_0_memory_driver_allocated_gb"] = driver_allocated / (1024 ** 3)
            metrics["gpu_0_memory_driver_allocated_mb"] = driver_allocated / (1024 ** 2)

            if driver_allocated > 0:
                metrics["gpu_0_memory_utilization_percent"] = (allocated / driver_allocated) * 100
        except Exception:
            pass

        # macOS ioreg GPU utilization and memory (works without root)
        try:
            metrics.update(self._collect_apple_gpu_ioreg())
        except Exception:
            pass

        return metrics

    def _collect_apple_gpu_ioreg(self) -> Dict[str, float]:
        """Parse Apple GPU stats from ioreg (Device/Renderer/Tiler utilization, memory)."""
        import subprocess
        import re

        metrics = {}
        result = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode != 0:
            return metrics

        output = result.stdout

        # Extract PerformanceStatistics block
        perf_match = re.search(r'"PerformanceStatistics"\s*=\s*\{([^}]+)\}', output)
        if not perf_match:
            return metrics

        perf_block = perf_match.group(1)

        # Parse key-value pairs: "Key"=Value
        for key, metric_name in [
            ("Device Utilization %", "gpu_0_device_utilization_percent"),
            ("Renderer Utilization %", "gpu_0_renderer_utilization_percent"),
            ("Tiler Utilization %", "gpu_0_tiler_utilization_percent"),
        ]:
            match = re.search(rf'"{re.escape(key)}"\s*=\s*(\d+)', perf_block)
            if match:
                metrics[metric_name] = float(match.group(1))

        # Memory stats from ioreg
        for key, metric_name in [
            ("In use system memory", "gpu_0_ioreg_in_use_memory_gb"),
            ("Alloc system memory", "gpu_0_ioreg_alloc_memory_gb"),
        ]:
            match = re.search(rf'"{re.escape(key)}"\s*=\s*(\d+)', perf_block)
            if match:
                metrics[metric_name] = int(match.group(1)) / (1024 ** 3)

        return metrics

    def collect_metrics(self) -> Dict[str, float]:
        """
        Collect all system metrics at the current moment.

        Returns:
            Dictionary of metric names to values
        """
        metrics = {}

        # Collect all metric types
        metrics.update(self._collect_cpu_metrics())
        metrics.update(self._collect_memory_metrics())
        metrics.update(self._collect_disk_metrics())
        metrics.update(self._collect_network_metrics())
        metrics.update(self._collect_gpu_metrics())

        # Add timestamp
        if self.start_time:
            metrics["elapsed_seconds"] = time.time() - self.start_time

        return metrics

    def _monitoring_loop(self):
        """Background thread loop that periodically collects metrics."""
        while self.running:
            try:
                metrics = self.collect_metrics()
                metrics["timestamp"] = time.time()
                self.metrics_history.append(metrics)
            except Exception as e:
                print(f"Error collecting metrics: {e}")

            # Sleep for interval
            time.sleep(self.interval)

    def start(self):
        """Start monitoring system metrics in a background thread."""
        if self.running:
            print("Warning: Metrics monitor already running")
            return

        self.running = True
        self.start_time = time.time()
        self.metrics_history = []

        # Reset baseline counters
        self.baseline_net_io = psutil.net_io_counters()
        try:
            self.baseline_disk_io = psutil.disk_io_counters()
        except:
            self.baseline_disk_io = None

        # Start background thread
        self.thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.thread.start()

        print(f"📊 System metrics monitoring started (interval: {self.interval}s)")

    def stop(self):
        """Stop monitoring system metrics."""
        if not self.running:
            return

        self.running = False
        if self.thread:
            self.thread.join(timeout=self.interval + 1.0)

        print("📊 System metrics monitoring stopped")

    def get_current_metrics(self) -> Dict[str, float]:
        """Get the most recent metrics snapshot."""
        if self.metrics_history:
            return self.metrics_history[-1].copy()
        return self.collect_metrics()

    def get_average_metrics(self) -> Dict[str, float]:
        """
        Calculate average metrics across all collected samples.

        Returns:
            Dictionary of metric names to average values
        """
        if not self.metrics_history:
            return {}

        # Collect all metric keys
        all_keys = set()
        for metrics in self.metrics_history:
            all_keys.update(metrics.keys())

        # Remove non-numeric keys
        all_keys.discard("timestamp")

        # Calculate averages
        averages = {}
        for key in all_keys:
            values = [m[key] for m in self.metrics_history if key in m]
            if values:
                averages[f"avg_{key}"] = sum(values) / len(values)

        return averages

    def get_max_metrics(self) -> Dict[str, float]:
        """
        Calculate maximum metrics across all collected samples.

        Returns:
            Dictionary of metric names to maximum values
        """
        if not self.metrics_history:
            return {}

        # Collect all metric keys
        all_keys = set()
        for metrics in self.metrics_history:
            all_keys.update(metrics.keys())

        # Remove non-numeric keys
        all_keys.discard("timestamp")

        # Calculate maximums
        maximums = {}
        for key in all_keys:
            values = [m[key] for m in self.metrics_history if key in m]
            if values:
                maximums[f"max_{key}"] = max(values)

        return maximums

    def get_summary_metrics(self) -> Dict[str, float]:
        """
        Get a summary of metrics including averages and maximums.

        Returns:
            Dictionary containing both average and max metrics
        """
        summary = {}
        summary.update(self.get_average_metrics())
        summary.update(self.get_max_metrics())

        # Add collection metadata
        summary["total_samples"] = len(self.metrics_history)
        if self.start_time:
            summary["monitoring_duration_seconds"] = time.time() - self.start_time

        return summary

    def __enter__(self):
        """Context manager entry - start monitoring."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop monitoring."""
        self.stop()

        # Cleanup GPU resources
        if self.nvml_initialized:
            try:
                self.pynvml.nvmlShutdown()
            except Exception:
                pass


def get_system_info() -> Dict[str, str]:
    """
    Get static system information including GPU details.

    Returns:
        Dictionary of system information
    """
    info = {
        "platform": psutil.os.name,
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
    }

    # Try pynvml first (NVIDIA)
    try:
        import pynvml
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        info["gpu_count"] = gpu_count
        info["gpu_backend"] = "nvidia_cuda"
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info[f"gpu_{i}_name"] = name
            info[f"gpu_{i}_memory_total_gb"] = round(mem.total / (1024 ** 3), 2)
        pynvml.nvmlShutdown()
        return info
    except Exception:
        pass

    # Try PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            info["gpu_count"] = gpu_count
            info["gpu_backend"] = "pytorch_cuda"
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                info[f"gpu_{i}_name"] = props.name
                info[f"gpu_{i}_memory_total_gb"] = round(props.total_mem / (1024 ** 3), 2)
            return info
    except Exception:
        pass

    # Try PyTorch MPS (Apple Silicon)
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["gpu_count"] = 1
            info["gpu_backend"] = "apple_mps"

            # Get GPU model and core count from ioreg
            try:
                import subprocess, re
                result = subprocess.run(
                    ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
                    capture_output=True, text=True, timeout=2
                )
                model_match = re.search(r'"model"\s*=\s*"([^"]+)"', result.stdout)
                cores_match = re.search(r'"gpu-core-count"\s*=\s*(\d+)', result.stdout)
                model_name = model_match.group(1) if model_match else "Apple Silicon"
                core_count = int(cores_match.group(1)) if cores_match else None
                info["gpu_0_name"] = model_name
                if core_count:
                    info["gpu_0_core_count"] = core_count
            except Exception:
                info["gpu_0_name"] = "Apple Silicon"

            return info
    except Exception:
        pass

    info["gpu_count"] = 0
    info["gpu_backend"] = "none"
    return info


if __name__ == "__main__":
    """Test the system metrics monitor."""
    print("Testing System Metrics Monitor")
    print("=" * 60)

    # Print system info
    print("\n📋 System Information:")
    for key, value in get_system_info().items():
        print(f"  {key}: {value}")

    # Test monitoring
    print("\n📊 Starting 10-second monitoring test...")
    with SystemMetricsMonitor(interval=2.0) as monitor:
        time.sleep(10)

        print("\n📈 Current Metrics:")
        current = monitor.get_current_metrics()
        for key, value in sorted(current.items()):
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

        print("\n📊 Summary Metrics:")
        summary = monitor.get_summary_metrics()
        for key, value in sorted(summary.items()):
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

    print("\n✅ Test complete!")
