# API Reference

Complete API documentation for Hybrid-GCS.

## Core Module

### GCSDecomposer

```python
from hybrid_gcs.core import GCSDecomposer, ConfigSpace

class GCSDecomposer:
    """Geometric Control System Decomposer."""
    
    def __init__(
        self,
        config_space: ConfigSpace,
        max_regions: int = 10,
        max_degree: int = 2,
        **kwargs
    ):
        """Initialize decomposer.
        
        Args:
            config_space: Configuration space
            max_regions: Maximum number of regions
            max_degree: Maximum polynomial degree
        """
    
    def decompose(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: List[Tuple[np.ndarray, float]]
    ) -> Dict[str, Any]:
        """Compute decomposition.
        
        Args:
            start: Start configuration
            goal: Goal configuration
            obstacles: List of (center, radius) obstacles
            
        Returns:
            Dictionary with 'regions' and 'feasible' keys
        """
```

### ConfigSpace

```python
class ConfigSpace:
    """Configuration space definition."""
    
    def __init__(
        self,
        name: str,
        dim: int,
        bounds: List[Tuple[float, float]],
        **kwargs
    ):
        """Initialize configuration space.
        
        Args:
            name: Space name
            dim: Dimensionality
            bounds: List of (min, max) bounds
        """
```

## Training Module

### OptimizedTrainer

```python
from hybrid_gcs.training import OptimizedTrainer

class OptimizedTrainer:
    """PPO-based trainer."""
    
    def __init__(
        self,
        policy_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        batch_size: int = 32,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        **kwargs
    ):
        """Initialize trainer.
        
        Args:
            policy_dim: Policy input dimension
            action_dim: Action dimension
            learning_rate: Learning rate
            batch_size: Batch size
            gamma: Discount factor
            gae_lambda: GAE lambda
        """
    
    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Perform training update.
        
        Args:
            batch: Training batch
            
        Returns:
            Dictionary with loss values
        """
```

### CurriculumLearning

```python
class CurriculumLearning:
    """Curriculum learning scheduler."""
    
    def __init__(
        self,
        schedule_type: str = "linear",
        total_steps: int = 100000,
        **kwargs
    ):
        """Initialize curriculum.
        
        Args:
            schedule_type: "linear", "exponential", "step", or "constant"
            total_steps: Total training steps
        """
    
    def get_difficulty(self, step: int) -> float:
        """Get difficulty at step.
        
        Args:
            step: Current training step
            
        Returns:
            Difficulty level [0, 1]
        """
```

## Environment Module

### ManipulationEnvironment

```python
from hybrid_gcs.environments import ManipulationEnvironment

class ManipulationEnvironment:
    """Robot manipulation environment."""
    
    def __init__(
        self,
        state_dim: int = 25,
        action_dim: int = 6,
        task: str = "reach",
        max_steps: int = 200,
        **kwargs
    ):
        """Initialize environment.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            task: "reach", "pick", "push", or "stack"
            max_steps: Maximum steps per episode
        """
    
    def reset(self) -> np.ndarray:
        """Reset environment.
        
        Returns:
            Initial observation
        """
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action.
        
        Args:
            action: Action vector
            
        Returns:
            (observation, reward, done, info)
        """
```

## Integration Module

### HybridPolicy

```python
from hybrid_gcs.integration import HybridPolicy, HybridPolicyConfig

class HybridPolicy:
    """Hybrid GCS-RL policy."""
    
    def __init__(
        self,
        config: HybridPolicyConfig,
        **kwargs
    ):
        """Initialize hybrid policy.
        
        Args:
            config: Configuration
        """
    
    def compute_action(
        self,
        state: np.ndarray,
        goal: np.ndarray
    ) -> np.ndarray:
        """Compute action.
        
        Args:
            state: Current state
            goal: Goal state
            
        Returns:
            Action vector
        """
```

## Utilities Module

### DataBuffer

```python
from hybrid_gcs.utils import DataBuffer

class DataBuffer:
    """Replay buffer for training."""
    
    def __init__(
        self,
        max_size: int = 10000,
        data_dim: int = 20,
        **kwargs
    ):
        """Initialize buffer.
        
        Args:
            max_size: Maximum buffer size
            data_dim: Data dimensionality
        """
    
    def add(self, data: np.ndarray) -> None:
        """Add data to buffer."""
    
    def get_batch(self, batch_size: int) -> np.ndarray:
        """Get random batch."""
```

### ConfigManager

```python
from hybrid_gcs.utils import ConfigManager

class ConfigManager:
    """Configuration manager."""
    
    def __init__(self, config_path: str):
        """Load configuration from file."""
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
```

## CLI Module

```bash
# Training
python -m hybrid_gcs.cli.train_cli \
    --env manipulation \
    --episodes 1000 \
    --batch-size 32

# Evaluation
python -m hybrid_gcs.cli.eval_cli \
    --model results/models/best.pt \
    --num-episodes 100

# Visualization
python -m hybrid_gcs.cli.viz_cli \
    --data results/ \
    --mode dashboard
```

---

For more details, see [Getting Started](getting_started.md) and [Training Guide](training_guide.md).
