"""
GCS Decomposer - Core Planning Algorithm
File: hybrid_gcs/core/gcs_decomposer.py

This module implements the Convex Geometry Sequence (GCS) decomposition algorithm
for robot motion planning. It decomposes the configuration space into convex regions
and computes optimal trajectories through them.

Based on: "Convex Geometry Sequences for Robot Trajectory Optimization" (Deits et al.)
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import networkx as nx
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Region:
    """Represents a convex region in the configuration space."""

    id:int
    A: np.ndarray # Inequality constraint matrix (m, d)
    b: np.ndarray # Inequality constraint vector (m,)
    center: np.ndarray # Region center (d,)
    radius: float # Region radius estimate

    def contains_point(self, point: np.ndarray) -> bool:
        """
        Check if a point is contained in this convex region

        Args:
            point: Configuration point (d,)
        
        Returns:
            True if point is in region, False otherwise
        """
        if(len(point)!=self.A.shape[1]) :
            raise ValueError(f"Point dimension {len(point)} != region dimension {self.A.shape[1]}")
        
        return np.all(self.A @ point <= self.b + 1e-6)
    
    def __repr__(self) -> str:
        return f"Region(id={self.id}, dim={self.A.shape[1]}, constraints={self.A.shape[0]})"
    

@dataclass
class ReverseTimeReachability:
    "Stores reverse time reachability graph information."

    regions: List[Region]
    graph: nx.DiGraph
    edge_weights: Dict[Tuple[int, int], float]
    timestamps: List[float]

class GCSDecomposer:
    """Convex Geometry Sequence (GCS) Decomposer for Motion Planning.
    
    Decomposes configuration space into convex regions and creates a sequence 
    graph for efficient trajectory planning using convex optimization.

    Attributes:
        config_space_dim (int): Dimension of configuration space
        num_regions (int): Number of convex regions
        regions (List[Region]) : List of convex regions
        region_graph (nx.DiGraph): Adjacency graph of regions
    """

    def __init__(self, config_space_dim:int, confi_bounds: Optional[np.ndarray] = None):
        """
        Initialize GCS Decomposer.

        Args:
            config_space_dim: Dimension of configuration space
            config_bounds: (d, 2) array of [min, max] bounds per dimension
        Raises:
            ValueError: If invalid dimensions provided
        """
        if config_space_dim <= 0:
            raise ValueError(f"config_space_dim must be positive, got {config_space_dim}")
        
        self.config_space_dim = config_space_dim
        self.config_bounds = confi_bounds or np.tile([-np.pi, np.pi], (config_space_dim, 1))

        self.regions: List[Region] = []
        self.region_graph = nx.DiGraph()
        self.reverse_time_graph: Optional[ReverseTimeReachability] = None

        logger.info(f"Initialized GCSDecomposer for {config_space_dim}D space")

    def add_convex_region(self, A: np.ndarray, b: np.ndarray, region_id: Optional[int]= None,) -> Region:
        """
        Add a convex region defined by A@x <=b.

        Args:
            A: Inequality constraint matrix (m, d)
            b: Inequality constraint vector (m,)
            region_id: Optional region identifier
        Returns:
            Region object
        Raises:
            ValueError: If dimensions don't match config space
        """
        if A.shape[1] != self.config_space_dim:
            raise ValueError(f"A and b must have same number of constraints")
        
        region_id = region_id or len(self.regions)
        center = self._compute_region_center(A, b)
        radius = self._estimate_region_radius(A, b)

        region = Region(id=region_id, A=A, b=b, center=center, radius=radius)
        self.regions.append(region)
        self.region_graph.add_node(region_id)

        logger.debug(f"Added region {region_id}: {region.A.shape[0]} constraints")
        return region
    
    def add_region_transition(self, from_region:int, to_region:int, cost:float = 1.0, feasible:bool = True) -> None:
        """
        Add transition edge between regions.

        Args:
            from_region: Source region ID
            to_region: Target region ID
            cost: Cost of transition
            feasible: Whether transition is feasible

        Raises:
            ValueError: If regions don't exist
        """
        if not any(r.id == from_region for r in self.regions):
            raise ValueError(f"Region {from_region} not found")
        if not any(r.id == to_region for r in self.regions):
            raise ValueError(f"Region {to_region} not found")
        
        if feasible:
            self.region_graph.add_edge(from_region, to_region, weight=cost)
            logger.debug(f"Added transition: {from_region} -> {to_region} (cost= {cost})")

    def find_sequence(self, start_region:int, goal_region:int) -> Optional[List[int]]:
        """
        Find optimal sequence of regions from start to goal.

        Args:
            start_region: Starting region ID
            goal_region: Goal region ID

        Returns:
            Ordered list of region IDs, or None if no path exists
        """
        try:
            path = nx.shortest_path(
                self.region_graph,
                source=start_region,
                target=goal_region,
                weight="weight",
            )
            logger.info(f"Found path with {len(path)} regions")
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            logger.warning(f"No path found: {e}")
            return None
        
    def get_region_by_id(self, region_id:int) -> Optional[Region]:
        """Get region by ID"""
        for region in self.regions:
            if region.id == region_id:
                return region
        return None
    
    def _compute_region_center(self, A:np.ndarray, b:np.ndarray) -> np.ndarray:
        """Compute approximate center of convex region."""
        try:
            import cvxpy as cp
            x=cp.Variable(A.shape[1])
            objective=cp.Minimize(cp.sum_squares(x))
            constraints=[A@x <= b]
            problem=cp.Problem(objective, constraints)
            problem.solve(solver=cp.SCS, verbose=False)

            if problem.status == cp.OPTIMAL:
                return np.array(x.value)
        except ImportError:
            pass

        # Fallback: Chebyshev center approximation
        center = np.zeros(A.shape[1])
        return center
    
    def _estimate_region_radius(self, A:np.ndarray, b:np.ndarray) -> float:
        """Estimate radius of convex regions."""
        return float(np.min(b) / (np.linalg.norm(A, axis=1).max() + 1e-6))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get decomposition statistics."""
        return {
            "num_regions": len(self.regions),
            "num_transitions": self.region_graph.number_of_edges(),
            "config_space_dim": self.config_space_dim,
            "avg_constraints_per_region": np.mean([r.A.shape[0] for r in self.regions]),
        }
    
    def __repr__(self) -> str:
        return (
            f"GCSDecomposer(dim={self.config_space_dim}, "
            f"regions={len(self.regions)}, "
            f"transitions={self.region_graph.number_of_edges()})"
        )
