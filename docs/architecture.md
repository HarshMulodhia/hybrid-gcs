# Architecture Guide

System architecture and component overview.

## System Overview

```
┌─────────────────────────────────────────────────────┐
│           Hybrid-GCS System Architecture            │
├─────────────────────────────────────────────────────┤
│                                                       │
│  ┌───────────────────────────────────────────────┐  │
│  │     Core Layer (GCS Decomposition)            │  │
│  │  - ConfigSpace Management                     │  │
│  │  - GCSDecomposer                              │  │
│  │  - Trajectory Optimization                    │  │
│  └───────────────────────────────────────────────┘  │
│                      ↓                              │
│  ┌───────────────────────────────────────────────┐  │
│  │     Training Layer (RL)                       │  │
│  │  - Policy Networks (Actor-Critic)             │  │
│  │  - PPO/SAC Algorithms                         │  │
│  │  - Curriculum Learning                        │  │
│  └───────────────────────────────────────────────┘  │
│                      ↓                              │
│  ┌───────────────────────────────────────────────┐  │
│  │     Integration Layer (Hybrid)                │  │
│  │  - Hybrid Policy                              │  │
│  │  - Feature Extraction                         │  │
│  │  - Action Blending                            │  │
│  └───────────────────────────────────────────────┘  │
│                      ↓                              │
│  ┌───────────────────────────────────────────────┐  │
│  │     Environment & Evaluation                  │  │
│  │  - Multiple Environments                      │  │
│  │  - Metrics & Analysis                         │  │
│  │  - Visualization                              │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Core Components

### 1. Configuration Space (`core/config_space.py`)
- Manages state and action spaces
- Bounds validation
- Multi-dimensional support

### 2. GCS Decomposer (`core/gcs_decomposer.py`)
- Trajectory planning
- Obstacle avoidance
- Region generation

### 3. Policy Network (`core/policy_network.py`)
- Actor network (π(a|s))
- Critic network (V(s))
- Network utilities

## Training Components

### 1. Optimized Trainer (`training/optimized_trainer.py`)
- PPO algorithm
- Advantage computation
- Policy updates

### 2. Curriculum Learning (`training/curriculum.py`)
- 4 schedule types
- Difficulty adaptation
- Progressive learning

### 3. Reward Shaping (`training/reward_shaping.py`)
- 7 reward strategies
- Intrinsic motivation
- Shaped rewards

## Integration Components

### 1. Hybrid Policy (`integration/hybrid_policy.py`)
- Policy blending
- 3 blend methods
- Weight adjustment

### 2. Feature Extractor (`integration/feature_extractor.py`)
- State features
- Trajectory features
- Combined features

### 3. Corridor Planning (`integration/corridor_planning.py`)
- Safe corridors
- Obstacle avoidance
- Clearance optimization

## Environment Layer

### Supported Environments
- YCB Grasping (single & dual-arm)
- Drone Navigation (single & multi-agent)
- Manipulation (reach, pick, push, stack)

## Module Dependencies

```
hybrid_gcs/
├── core/              (GCS algorithms)
│   └── Dependencies: numpy, scipy
├── training/          (RL algorithms)
│   └── Dependencies: core, torch, numpy
├── environments/      (Simulators)
│   └── Dependencies: gym, pybullet
├── evaluation/        (Analysis)
│   └── Dependencies: core, training, numpy
├── integration/       (Hybrid)
│   └── Dependencies: core, training
├── cli/               (Command-line)
│   └── Dependencies: all above
└── utils/             (Helpers)
    └── Dependencies: core
```

## Data Flow

### Training Flow
```
Environment
    ↓
Observation
    ↓
Policy Network
    ↓
Action
    ↓
Environment Step
    ↓
Reward + Next State
    ↓
PPO Update
    ↓
Policy Improvement
```

### Hybrid Execution Flow
```
State
    ↓
├─→ GCS Planner → GCS Action
│
├─→ RL Policy → RL Action
│
Blend Actions
    ↓
Hybrid Action
    ↓
Execute on Robot
```

## Key Design Principles

1. **Modularity**: Independent components
2. **Type Safety**: Full type hints
3. **Documentation**: Complete docstrings
4. **Testing**: 30+ test cases
5. **Scalability**: Multi-agent support
6. **Safety**: Built-in safety mechanisms

---

See [API Reference](api_reference.md) for detailed component documentation.
