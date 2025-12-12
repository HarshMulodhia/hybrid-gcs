"""
3D Trajectory Visualization
File: hybrid_gcs/visualization/trajectory_viz.py

Visualizes trajectories in 3D configuration space.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class TrajectoryVisualizer:
    """
    Visualizes trajectories in configuration space.
    """
    
    def __init__(self, output_dir: str = "trajectories"):
        """
        Initialize trajectory visualizer.
        
        Args:
            output_dir: Output directory for visualizations
        """
        self.output_dir = output_dir
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Initialized TrajectoryVisualizer: output_dir={output_dir}")
    
    def visualize_trajectory_2d(
        self,
        trajectory: np.ndarray,
        start: Optional[np.ndarray] = None,
        goal: Optional[np.ndarray] = None,
        title: str = "Trajectory",
        output_file: str = "trajectory_2d.html",
    ) -> str:
        """
        Visualize 2D trajectory.
        
        Args:
            trajectory: (N, 2) trajectory points
            start: Start point
            goal: Goal point
            title: Plot title
            output_file: Output file name
            
        Returns:
            Path to generated visualization
        """
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            # Trajectory
            fig.add_trace(go.Scatter(
                x=trajectory[:, 0],
                y=trajectory[:, 1],
                mode='lines+markers',
                name='Trajectory',
                line=dict(color='blue', width=2),
                marker=dict(size=5),
            ))
            
            # Start point
            if start is not None:
                fig.add_trace(go.Scatter(
                    x=[start[0]],
                    y=[start[1]],
                    mode='markers',
                    name='Start',
                    marker=dict(color='green', size=15, symbol='circle'),
                ))
            
            # Goal point
            if goal is not None:
                fig.add_trace(go.Scatter(
                    x=[goal[0]],
                    y=[goal[1]],
                    mode='markers',
                    name='Goal',
                    marker=dict(color='red', size=15, symbol='star'),
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title='X',
                yaxis_title='Y',
                height=600,
                showlegend=True,
                hovermode='closest',
            )
            
            output_path = f"{self.output_dir}/{output_file}"
            fig.write_html(output_path)
            
            logger.info(f"Saved 2D trajectory visualization to {output_path}")
            return output_path
            
        except ImportError:
            logger.warning("Plotly not installed, skipping visualization")
            return ""
    
    def visualize_trajectory_3d(
        self,
        trajectory: np.ndarray,
        start: Optional[np.ndarray] = None,
        goal: Optional[np.ndarray] = None,
        regions: Optional[List[Dict]] = None,
        title: str = "3D Trajectory",
        output_file: str = "trajectory_3d.html",
    ) -> str:
        """
        Visualize 3D trajectory.
        
        Args:
            trajectory: (N, 3+) trajectory points
            start: Start point
            goal: Goal point
            regions: Optional list of region centers
            title: Plot title
            output_file: Output file name
            
        Returns:
            Path to generated visualization
        """
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            # Use first 3 dimensions
            traj_3d = trajectory[:, :3]
            
            # Trajectory
            fig.add_trace(go.Scatter3d(
                x=traj_3d[:, 0],
                y=traj_3d[:, 1],
                z=traj_3d[:, 2],
                mode='lines+markers',
                name='Trajectory',
                line=dict(color='blue', width=4),
                marker=dict(size=4),
            ))
            
            # Start point
            if start is not None:
                start_3d = start[:3]
                fig.add_trace(go.Scatter3d(
                    x=[start_3d[0]],
                    y=[start_3d[1]],
                    z=[start_3d[2]],
                    mode='markers',
                    name='Start',
                    marker=dict(color='green', size=10, symbol='circle'),
                ))
            
            # Goal point
            if goal is not None:
                goal_3d = goal[:3]
                fig.add_trace(go.Scatter3d(
                    x=[goal_3d[0]],
                    y=[goal_3d[1]],
                    z=[goal_3d[2]],
                    mode='markers+text',
                    name='Goal',
                    marker=dict(color='red', size=10, symbol='star'),
                    text=['Goal'],
                ))
            
            # Regions
            if regions:
                for i, region in enumerate(regions):
                    center = region.get('center', np.zeros(3))[:3]
                    fig.add_trace(go.Scatter3d(
                        x=[center[0]],
                        y=[center[1]],
                        z=[center[2]],
                        mode='markers',
                        name=f'Region {i}',
                        marker=dict(size=8),
                    ))
            
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                ),
                height=700,
                showlegend=True,
                hovermode='closest',
            )
            
            output_path = f"{self.output_dir}/{output_file}"
            fig.write_html(output_path)
            
            logger.info(f"Saved 3D trajectory visualization to {output_path}")
            return output_path
            
        except ImportError:
            logger.warning("Plotly not installed, skipping visualization")
            return ""
    
    def compare_trajectories(
        self,
        trajectories: Dict[str, np.ndarray],
        start: Optional[np.ndarray] = None,
        goal: Optional[np.ndarray] = None,
        title: str = "Trajectory Comparison",
        output_file: str = "trajectory_comparison.html",
    ) -> str:
        """
        Compare multiple trajectories.
        
        Args:
            trajectories: Dict of {name: trajectory}
            start: Start point
            goal: Goal point
            title: Plot title
            output_file: Output file name
            
        Returns:
            Path to generated visualization
        """
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
            
            # Add trajectories
            for i, (name, trajectory) in enumerate(trajectories.items()):
                color = colors[i % len(colors)]
                
                if trajectory.shape[1] >= 3:
                    # 3D
                    fig.add_trace(go.Scatter3d(
                        x=trajectory[:, 0],
                        y=trajectory[:, 1],
                        z=trajectory[:, 2],
                        mode='lines',
                        name=name,
                        line=dict(color=color, width=3),
                    ))
                else:
                    # 2D
                    fig.add_trace(go.Scatter(
                        x=trajectory[:, 0],
                        y=trajectory[:, 1],
                        mode='lines',
                        name=name,
                        line=dict(color=color, width=2),
                    ))
            
            # Start and goal
            if start is not None and goal is not None:
                if start.shape[0] >= 3:
                    fig.add_trace(go.Scatter3d(
                        x=[start[0]], y=[start[1]], z=[start[2]],
                        mode='markers',
                        name='Start',
                        marker=dict(color='green', size=12),
                    ))
                    fig.add_trace(go.Scatter3d(
                        x=[goal[0]], y=[goal[1]], z=[goal[2]],
                        mode='markers',
                        name='Goal',
                        marker=dict(color='red', size=12),
                    ))
            
            fig.update_layout(
                title=title,
                height=700,
                showlegend=True,
                hovermode='closest',
            )
            
            output_path = f"{self.output_dir}/{output_file}"
            fig.write_html(output_path)
            
            logger.info(f"Saved trajectory comparison to {output_path}")
            return output_path
            
        except ImportError:
            logger.warning("Plotly not installed, skipping visualization")
            return ""
    
    def create_animation(
        self,
        trajectories: Dict[str, np.ndarray],
        output_file: str = "trajectory_animation.html",
    ) -> str:
        """
        Create animated trajectory visualization.
        
        Args:
            trajectories: Dict of {name: trajectory}
            output_file: Output file name
            
        Returns:
            Path to generated animation
        """
        try:
            import plotly.graph_objects as go
            
            # Get max length
            max_len = max(len(traj) for traj in trajectories.values())
            
            frames = []
            
            for frame_idx in range(max_len):
                frame_traces = []
                
                for name, trajectory in trajectories.items():
                    if frame_idx < len(trajectory):
                        frame_traj = trajectory[:frame_idx+1]
                        
                        if trajectory.shape[1] >= 3:
                            frame_traces.append(go.Scatter3d(
                                x=frame_traj[:, 0],
                                y=frame_traj[:, 1],
                                z=frame_traj[:, 2],
                                mode='lines+markers',
                                name=name,
                            ))
                        else:
                            frame_traces.append(go.Scatter(
                                x=frame_traj[:, 0],
                                y=frame_traj[:, 1],
                                mode='lines+markers',
                                name=name,
                            ))
                
                frames.append(go.Frame(data=frame_traces, name=str(frame_idx)))
            
            # Create figure
            first_traj = next(iter(trajectories.values()))
            if first_traj.shape[1] >= 3:
                initial_trace = go.Scatter3d(x=[], y=[], z=[])
            else:
                initial_trace = go.Scatter(x=[], y=[])
            
            fig = go.Figure(data=[initial_trace], frames=frames)
            
            fig.update_layout(
                updatemenus=[{
                    'type': 'buttons',
                    'buttons': [
                        {'label': 'Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 100}}]},
                        {'label': 'Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0}}]},
                    ]
                }],
            )
            
            output_path = f"{self.output_dir}/{output_file}"
            fig.write_html(output_path)
            
            logger.info(f"Saved trajectory animation to {output_path}")
            return output_path
            
        except ImportError:
            logger.warning("Plotly not installed, skipping animation")
            return ""
