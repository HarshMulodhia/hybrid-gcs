"""
Real-time Dashboard for Training Monitoring
File: hybrid_gcs/visualization/dashboard.py

Provides real-time visualization of training progress.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for dashboard."""
    
    update_interval: int = 100  # Update every N steps
    max_history: int = 1000  # Max points to keep
    metrics_to_track: List[str] = field(
        default_factory=lambda: [
            'episode_reward',
            'actor_loss',
            'critic_loss',
            'learning_rate',
        ]
    )
    save_html: bool = True
    output_dir: str = "dashboards"


class Dashboard:
    """
    Real-time training dashboard.
    
    Tracks and visualizes training metrics.
    """
    
    def __init__(self, config: DashboardConfig):
        """
        Initialize dashboard.
        
        Args:
            config: Dashboard configuration
        """
        self.config = config
        
        # Initialize metric tracking
        self.metrics_history: Dict[str, deque] = {
            metric: deque(maxlen=config.max_history)
            for metric in config.metrics_to_track
        }
        
        self.timesteps: deque = deque(maxlen=config.max_history)
        self.episodes: deque = deque(maxlen=config.max_history)
        
        self.update_count = 0
        self.start_time = datetime.now()
        
        logger.info(f"Initialized Dashboard: tracking {len(config.metrics_to_track)} metrics")
    
    def update(
        self,
        timestep: int,
        episode: int,
        metrics: Dict[str, float],
    ) -> None:
        """
        Update dashboard with new metrics.
        
        Args:
            timestep: Current timestep
            episode: Current episode
            metrics: Dictionary of metrics
        """
        self.update_count += 1
        self.timesteps.append(timestep)
        self.episodes.append(episode)
        
        for metric_name in self.config.metrics_to_track:
            if metric_name in metrics:
                self.metrics_history[metric_name].append(metrics[metric_name])
    
    def get_status(self) -> Dict[str, Any]:
        """Get current dashboard status."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        status = {
            'update_count': self.update_count,
            'elapsed_time': elapsed,
            'updates_per_second': self.update_count / max(elapsed, 1),
        }
        
        # Add current values
        for metric_name, history in self.metrics_history.items():
            if history:
                status[f"{metric_name}_current"] = history[-1]
                status[f"{metric_name}_mean"] = np.mean(list(history))
                status[f"{metric_name}_min"] = np.min(list(history))
                status[f"{metric_name}_max"] = np.max(list(history))
        
        return status
    
    def get_plot_data(self, metric: str) -> Optional[Dict[str, List]]:
        """
        Get plot data for metric.
        
        Args:
            metric: Metric name
            
        Returns:
            Dictionary with timesteps and values
        """
        if metric not in self.metrics_history:
            return None
        
        history = list(self.metrics_history[metric])
        timesteps = list(self.timesteps)
        
        if not history:
            return None
        
        return {
            'x': timesteps[-len(history):],
            'y': history,
            'name': metric,
        }
    
    def generate_html_report(self, output_file: str) -> str:
        """
        Generate HTML dashboard report.
        
        Args:
            output_file: Output file path
            
        Returns:
            Path to generated report
        """
        html = self._generate_html()
        
        with open(output_file, 'w') as f:
            f.write(html)
        
        logger.info(f"Dashboard HTML saved to {output_file}")
        return output_file
    
    def _generate_html(self) -> str:
        """Generate HTML content."""
        status = self.get_status()
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Hybrid-GCS Training Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .status {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 20px 0; }}
        .metric {{ background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .plot {{ background: white; padding: 15px; margin: 10px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-label {{ font-weight: bold; color: #2c3e50; }}
        .metric-value {{ font-size: 1.5em; color: #3498db; margin-top: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Hybrid-GCS Training Dashboard</h1>
            <p>Real-time training visualization</p>
        </div>
        
        <div class="status">
"""
        
        for key, value in status.items():
            if isinstance(value, float):
                value = f"{value:.3f}"
            html += f"""
            <div class="metric">
                <div class="metric-label">{key}</div>
                <div class="metric-value">{value}</div>
            </div>
"""
        
        html += """
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def __repr__(self) -> str:
        status = self.get_status()
        return (
            f"Dashboard(updates={status['update_count']}, "
            f"elapsed={status['elapsed_time']:.1f}s)"
        )
