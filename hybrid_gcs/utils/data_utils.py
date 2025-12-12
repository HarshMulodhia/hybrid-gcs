"""
Data Utilities Module
File: hybrid_gcs/utils/data_utils.py

Provides data handling and processing utilities.
"""

import logging
from typing import Dict, List, Tuple, Optional
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class DataBuffer:
    """
    Ring buffer for storing trajectory data.
    
    Efficient storage for time-series data.
    """
    
    def __init__(self, max_size: int = 1000, data_dim: int = 3):
        """
        Initialize data buffer.
        
        Args:
            max_size: Maximum buffer size
            data_dim: Data dimension
        """
        self.max_size = max_size
        self.data_dim = data_dim
        self.buffer: deque = deque(maxlen=max_size)
        
        logger.debug(f"Initialized DataBuffer (size={max_size}, dim={data_dim})")
    
    def add(self, data: np.ndarray) -> None:
        """
        Add data to buffer.
        
        Args:
            data: Data to add (shape: (data_dim,))
        """
        if data.shape[0] != self.data_dim:
            raise ValueError(
                f"Data dimension mismatch: expected {self.data_dim}, got {data.shape[0]}"
            )
        
        self.buffer.append(data.copy())
    
    def get_array(self) -> np.ndarray:
        """
        Get buffer as array.
        
        Returns:
            Array of shape (buffer_len, data_dim)
        """
        if not self.buffer:
            return np.array([]).reshape(0, self.data_dim)
        
        return np.array(list(self.buffer))
    
    def clear(self) -> None:
        """Clear buffer."""
        self.buffer.clear()
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return len(self.buffer) == self.max_size
    
    def __len__(self) -> int:
        """Get buffer length."""
        return len(self.buffer)


def normalize_data(
    data: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize data to zero mean and unit variance.
    
    Args:
        data: Data array (shape: (N, D))
        mean: Optional precomputed mean
        std: Optional precomputed std
        
    Returns:
        (normalized_data, mean, std)
    """
    if mean is None:
        mean = np.mean(data, axis=0)
    if std is None:
        std = np.std(data, axis=0)
    
    std = np.where(std == 0, 1, std)  # Avoid division by zero
    
    normalized = (data - mean) / std
    
    logger.debug(f"Normalized data: mean={np.mean(normalized):.4f}, std={np.std(normalized):.4f}")
    
    return normalized, mean, std


def denormalize_data(
    data: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Denormalize data.
    
    Args:
        data: Normalized data
        mean: Original mean
        std: Original std
        
    Returns:
        Denormalized data
    """
    return data * std + mean


def split_data(
    data: np.ndarray,
    labels: Optional[np.ndarray] = None,
    train_ratio: float = 0.8,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/test sets.
    
    Args:
        data: Data array (shape: (N, D))
        labels: Optional labels array
        train_ratio: Fraction for training
        shuffle: Whether to shuffle data
        seed: Random seed
        
    Returns:
        (train_data, test_data, train_labels, test_labels)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_samples = len(data)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    split_idx = int(n_samples * train_ratio)
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    train_data = data[train_indices]
    test_data = data[test_indices]
    
    train_labels = labels[train_indices] if labels is not None else None
    test_labels = labels[test_indices] if labels is not None else None
    
    logger.info(
        f"Split data: {len(train_data)} train, {len(test_data)} test "
        f"(ratio={train_ratio:.1%})"
    )
    
    return train_data, test_data, train_labels, test_labels


def batch_data(
    data: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Create batches from data.
    
    Args:
        data: Data array (shape: (N, D))
        batch_size: Size of each batch
        shuffle: Whether to shuffle data
        seed: Random seed
        
    Returns:
        List of batch arrays
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_samples = len(data)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    batches = []
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        batches.append(data[batch_indices])
    
    logger.debug(f"Created {len(batches)} batches of size {batch_size}")
    
    return batches


def pad_sequences(
    sequences: List[np.ndarray],
    max_length: Optional[int] = None,
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    Pad sequences to same length.
    
    Args:
        sequences: List of sequence arrays
        max_length: Maximum length (default: max in sequences)
        pad_value: Value for padding
        
    Returns:
        Padded array (shape: (n_sequences, max_length, feature_dim))
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    n_sequences = len(sequences)
    feature_dim = sequences[0].shape[1] if len(sequences[0].shape) > 1 else 1
    
    padded = np.full((n_sequences, max_length, feature_dim), pad_value)
    
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq
    
    logger.debug(f"Padded {n_sequences} sequences to length {max_length}")
    
    return padded


def compute_statistics(data: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics for data.
    
    Args:
        data: Data array
        
    Returns:
        Dict of statistics
    """
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'median': float(np.median(data)),
    }
