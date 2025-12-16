"""
Hybrid-GCS Training Script

File: scripts/train.py

Description:
This script provides a production-grade command-line interface for training
Hybrid-GCS models using the OptimizedTrainer and ConfigLoader.

Features:
- Load configurations from YAML files or presets.
- Override any configuration parameter via CLI.
- Comprehensive logging and command-line feedback.
- Seed management for reproducibility.
- Automatic directory creation for results.
- Full integration with TensorBoard.
"""

import argparse
import logging
import os
import sys
import numpy as np
import torch

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from hybrid_gcs.training.configs.config_loader import ConfigLoader, TrainingConfig
from hybrid_gcs.training.optimized_trainer import OptimizedTrainer
from hybrid_gcs.environments import ManipulationEnvironment, DroneNavigationEnvironment
from hybrid_gcs.core import ActorNetwork, CriticNetwork

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def create_environment(config: TrainingConfig):
    """Creates an environment based on the configuration."""
    if config.env_name == 'manipulation':
        return ManipulationEnvironment(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            task=config.task
        )
    elif config.env_name == 'drone':
        return DroneNavigationEnvironment()
    else:
        raise ValueError(f"Unknown environment: {config.env_name}")


def create_models(config: TrainingConfig, device: torch.device):
    """Creates actor and critic models."""
    actor = ActorNetwork(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        hidden_sizes=config.actor_hidden_sizes,
        activation=config.activation
    ).to(device)

    critic = CriticNetwork(
        state_dim=config.state_dim,
        hidden_sizes=config.critic_hidden_sizes,
        activation=config.activation
    ).to(device)
    
    return actor, critic


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train Hybrid-GCS Models")

    # --- Configuration Arguments ---
    parser.add_argument('--config', type=str, help="Path to YAML config file.")
    parser.addargument('--preset', type=str, help="Use a preset configuration (e.g., 'standard', 'high_quality').")
    
    # --- Key Training Parameter Overrides ---
    parser.add_argument('--total_timesteps', type=int, help="Override total training timesteps.")
    parser.add_argument('--env_name', type=str, help="Environment name (e.g., 'manipulation').")
    parser.add_argument('--batch_size', type=int, help="Override batch size.")
    parser.add_argument('--learning_rate', type=float, help="Override learning rate.")
    
    # --- Output & Logging ---
    parser.add_argument('--save_dir', type=str, default='results', help="Directory to save models and logs.")
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    # --- Reproducibility ---
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    # --- Set Logging Level ---
    logger.setLevel(getattr(logging, args.log_level.upper()))
    
    # --- Set Seed ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    logger.info(f"Using random seed: {args.seed}")

    # --- Load Configuration ---
    config_loader = ConfigLoader()
    
    # Create an override dictionary from CLI arguments
    overrides = {
        'total_timesteps': args.total_timesteps,
        'env_name': args.env_name,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'save_dir': args.save_dir,
    }
    # Filter out None values so they don't override defaults
    cli_overrides = {k: v for k, v in overrides.items() if v is not None}

    try:
        config_dict = config_loader.load_config(
            path=args.config,
            preset=args.preset,
            overrides=cli_overrides
        )
        # Convert dictionary to TrainingConfig dataclass
        config = TrainingConfig(**config_dict)
        logger.info(f"Configuration loaded successfully. Preset: '{args.preset}', Path: '{args.config}'.")
        
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Configuration Error: {e}")
        sys.exit(1)

    # --- Prepare for Training ---
    # Ensure directories exist
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(os.path.join(config.save_dir, 'checkpoints'), exist_ok=True)
    config.log_dir = os.path.join(config.save_dir, 'logs') # Set log_dir within save_dir
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    # --- Initialize Components ---
    try:
        env = create_environment(config)
        actor, critic = create_models(config, device)
        
        trainer = OptimizedTrainer(
            config=config,
            actor=actor,
            critic=critic,
            env=env,
            callbacks=[]  # Add any callbacks here
        )
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)

    # --- Start Training ---
    logger.info("Starting training...")
    try:
        final_metrics = trainer.train()
        logger.info("Training finished successfully.")
        logger.info(f"Final Metrics: {final_metrics}")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        sys.exit(1)
        
    # --- Save Final Model ---
    final_model_path = os.path.join(config.save_dir, 'models', 'final_model.pt')
    trainer.save_checkpoint(path=final_model_path)
    logger.info(f"Final model saved to {final_model_path}")


if __name__ == '__main__':
    main()
