"""
Policy Network - Neural Network Architectures
File: hybrid_gcs/core/policy_network.py

Defines actor and critic networks for reinforcement learning policies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class PolicyNetwork(nn.Module):
    """
    Base policy network class.
    
    Provides shared functionality for actor and critic networks.
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = None, activation: str = "relu", 
                 dropout_rate: float = 0.0):
        """
        Initialize policy network.
        
        Args:
            input_size: Size of input features
            hidden_sizes: Sizes of hidden layers
            activation: Activation function ("relu", "tanh", "elu")
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes or [256, 256]
        self.activation_name = activation
        self.dropout_rate = dropout_rate
        
        # Select activation
        self.activation = self._get_activation(activation)
        
        logger.info(
            f"Initialized PolicyNetwork: "
            f"input={input_size}, hidden={self.hidden_sizes}"
        )
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(0.1),
        }
        return activations.get(name, nn.ReLU())
    
    def _build_layers(self, layer_sizes: List[int], output_size: Optional[int] = None) -> nn.Sequential:
        """Build network layers."""
        layers = []
        
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # No activation after last layer
                layers.append(self.activation)
                if self.dropout_rate > 0:
                    layers.append(nn.Dropout(self.dropout_rate))
        
        if output_size is not None:
            layers.append(nn.Linear(layer_sizes[-1], output_size))
        
        return nn.Sequential(*layers)


class ActorNetwork(PolicyNetwork):
    """
    Actor Network for Policy Gradient Methods.
    
    Outputs mean and log std of action distribution.
    """
    
    def __init__(self, input_size: int, action_size: int, hidden_sizes: List[int] = None, activation: str = "relu",
        dropout_rate: float = 0.0, log_std_init: float = -0.5, log_std_min: float = -20.0, log_std_max: float = 2.0):
        """
        Initialize actor network.
        
        Args:
            input_size: Size of state observation
            action_size: Size of action
            hidden_sizes: Hidden layer sizes
            activation: Activation function
            dropout_rate: Dropout rate
            log_std_init: Initial log standard deviation
            log_std_min: Minimum log std (for stability)
            log_std_max: Maximum log std (for stability)
        """
        super().__init__(input_size, hidden_sizes, activation, dropout_rate)
        
        self.action_size = action_size
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Build network
        layer_sizes = [input_size] + self.hidden_sizes
        self.backbone = self._build_layers(layer_sizes)
        
        # Output layers
        self.mean_head = nn.Linear(self.hidden_sizes[-1], action_size)
        self.log_std_head = nn.Linear(self.hidden_sizes[-1], action_size)
        
        # Initialize log_std
        self.log_std_head.weight.data.fill_(log_std_init)
        self.log_std_head.bias.data.fill_(log_std_init)
        
        logger.info(f"Initialized ActorNetwork: action_size={action_size}")
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to get action distribution parameters.
        
        Args:
            state: State observation (batch_size, input_size)
            
        Returns:
            mean: Action mean (batch_size, action_size)
            log_std: Log standard deviation (batch_size, action_size)
        """
        x = self.backbone(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            state: State observation
            deterministic: If True, use mean action
            
        Returns:
            action: Sampled action
            log_prob: Log probability of action
        """
        mean, log_std = self(state)
        std = log_std.exp()
        
        if deterministic:
            return mean, torch.zeros(1)
        
        distribution = Normal(mean, std)
        action = distribution.rsample()  # Reparameterization trick
        log_prob = distribution.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob


class CriticNetwork(PolicyNetwork):
    """
    Critic Network for Value Function Estimation.
    
    Outputs state value or action-value estimate.
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = None, activation: str = "relu", 
                 dropout_rate: float = 0.0, output_size: int = 1):
        """
        Initialize critic network.
        
        Args:
            input_size: Size of input (state or state-action)
            hidden_sizes: Hidden layer sizes
            activation: Activation function
            dropout_rate: Dropout rate
            output_size: Output size (usually 1 for V or Q function)
        """
        super().__init__(input_size, hidden_sizes, activation, dropout_rate)
        
        self.output_size = output_size
        
        # Build network
        layer_sizes = [input_size] + self.hidden_sizes
        self.backbone = self._build_layers(layer_sizes)
        
        # Output layer
        self.value_head = nn.Linear(self.hidden_sizes[-1], output_size)
        
        logger.info(f"Initialized CriticNetwork: input={input_size}")
    
    def forward(self, state_action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get value estimate.
        
        Args:
            state_action: State or state-action concatenation
            
        Returns:
            value: Value estimate (batch_size, output_size)
        """
        x = self.backbone(state_action)
        value = self.value_head(x)
        return value


class DuelingNetwork(PolicyNetwork):
    """
    Dueling Network Architecture for Q-learning.
    
    Separates value and advantage streams.
    """
    
    def __init__(self, input_size: int, action_size: int, hidden_sizes: List[int] = None,
        activation: str = "relu", dropout_rate: float = 0.0):
        """
        Initialize dueling network.
        
        Args:
            input_size: Size of state
            action_size: Number of actions
            hidden_sizes: Hidden layer sizes
            activation: Activation function
            dropout_rate: Dropout rate
        """
        super().__init__(input_size, hidden_sizes, activation, dropout_rate)
        
        self.action_size = action_size
        
        # Common backbone
        layer_sizes = [input_size] + self.hidden_sizes
        self.backbone = self._build_layers(layer_sizes)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(self.hidden_sizes[-1], 128),
            self.activation,
            nn.Linear(128, 1),
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.hidden_sizes[-1], 128),
            self.activation,
            nn.Linear(128, action_size),
        )
        
        logger.info(f"Initialized DuelingNetwork: actions={action_size}")
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for dueling Q-network.
        
        Args:
            state: State observation
            
        Returns:
            q_values: Q-values for each action
        """
        x = self.backbone(state)
        value = self.value_stream(x)
        advantages = self.advantage_stream(x)
        
        # Dueling combination: Q = V + (A - mean(A))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values