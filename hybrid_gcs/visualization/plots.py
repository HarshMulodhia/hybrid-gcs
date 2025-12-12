"""
Advanced Plotting with Plotly
File: hybrid_gcs/visualization/plots.py

Creates interactive plots for analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


class PlotGenerator:
    """
    Generates publication-quality plots.
    """
    
    def __init__(self, output_dir: str = "plots"):
        """
        Initialize plot generator.
        
        Args:
            output_dir: Output directory for plots
        """
        self.output_dir = output_dir
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Initialized PlotGenerator: output_dir={output_dir}")
    
    def plot_training_curves(
        self,
        timesteps: List[int],
        rewards: List[float],
        actor_loss: List[float],
        critic_loss: List[float],
        output_file: str = "training_curves.html",
    ) -> str:
        """
        Plot training curves.
        
        Args:
            timesteps: List of timesteps
            rewards: Episode rewards
            actor_loss: Actor loss values
            critic_loss: Critic loss values
            output_file: Output file name
            
        Returns:
            Path to generated plot
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Episode Rewards",
                    "Actor Loss",
                    "Critic Loss",
                    "Combined Loss",
                ),
            )
            
            # Episode rewards
            fig.add_trace(
                go.Scatter(
                    x=timesteps,
                    y=rewards,
                    name="Episode Reward",
                    mode="lines",
                ),
                row=1, col=1
            )
            
            # Actor loss
            fig.add_trace(
                go.Scatter(
                    x=timesteps,
                    y=actor_loss,
                    name="Actor Loss",
                    mode="lines",
                    line=dict(color="orange"),
                ),
                row=1, col=2
            )
            
            # Critic loss
            fig.add_trace(
                go.Scatter(
                    x=timesteps,
                    y=critic_loss,
                    name="Critic Loss",
                    mode="lines",
                    line=dict(color="red"),
                ),
                row=2, col=1
            )
            
            # Combined loss
            combined = np.array(actor_loss) + np.array(critic_loss)
            fig.add_trace(
                go.Scatter(
                    x=timesteps,
                    y=combined,
                    name="Combined Loss",
                    mode="lines",
                    line=dict(color="purple"),
                ),
                row=2, col=2
            )
            
            fig.update_xaxes(title_text="Timestep", row=1, col=1)
            fig.update_xaxes(title_text="Timestep", row=1, col=2)
            fig.update_xaxes(title_text="Timestep", row=2, col=1)
            fig.update_xaxes(title_text="Timestep", row=2, col=2)
            
            fig.update_layout(height=600, showlegend=True)
            
            output_path = f"{self.output_dir}/{output_file}"
            fig.write_html(output_path)
            
            logger.info(f"Saved training curves plot to {output_path}")
            return output_path
            
        except ImportError:
            logger.warning("Plotly not installed, skipping plot generation")
            return ""
    
    def plot_trajectory_comparison(
        self,
        trajectories: Dict[str, np.ndarray],
        labels: Optional[Dict[str, str]] = None,
        output_file: str = "trajectory_comparison.html",
    ) -> str:
        """
        Plot trajectory comparison.
        
        Args:
            trajectories: Dict of {name: trajectory}
            labels: Optional labels
            output_file: Output file name
            
        Returns:
            Path to generated plot
        """
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            
            for i, (name, trajectory) in enumerate(trajectories.items()):
                color = colors[i % len(colors)]
                
                if trajectory.shape[1] >= 3:
                    # 3D trajectory
                    fig.add_trace(go.Scatter3d(
                        x=trajectory[:, 0],
                        y=trajectory[:, 1],
                        z=trajectory[:, 2],
                        mode='lines+markers',
                        name=labels.get(name, name) if labels else name,
                        line=dict(color=color, width=4),
                    ))
                else:
                    # 2D trajectory
                    fig.add_trace(go.Scatter(
                        x=trajectory[:, 0],
                        y=trajectory[:, 1],
                        mode='lines+markers',
                        name=labels.get(name, name) if labels else name,
                        line=dict(color=color, width=2),
                    ))
            
            fig.update_layout(
                height=600,
                title="Trajectory Comparison",
                showlegend=True,
            )
            
            output_path = f"{self.output_dir}/{output_file}"
            fig.write_html(output_path)
            
            logger.info(f"Saved trajectory comparison to {output_path}")
            return output_path
            
        except ImportError:
            logger.warning("Plotly not installed, skipping plot generation")
            return ""
    
    def plot_heatmap(
        self,
        data: np.ndarray,
        title: str = "Heatmap",
        output_file: str = "heatmap.html",
    ) -> str:
        """
        Plot heatmap.
        
        Args:
            data: (H, W) heatmap data
            title: Plot title
            output_file: Output file name
            
        Returns:
            Path to generated plot
        """
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure(data=go.Heatmap(z=data, colorscale='Viridis'))
            fig.update_layout(title=title, height=600)
            
            output_path = f"{self.output_dir}/{output_file}"
            fig.write_html(output_path)
            
            logger.info(f"Saved heatmap to {output_path}")
            return output_path
            
        except ImportError:
            logger.warning("Plotly not installed, skipping plot generation")
            return ""
