# Real-World Deployment Guide

Deploy Hybrid-GCS to real robots.

## UR5 Arm Integration

```python
from hybrid_gcs.integration import RobotController
import numpy as np

# Connect to UR5
controller = RobotController(
    robot_type="ur5",
    ip_address="192.168.1.100",
    port=30003
)

# Load trained policy
policy = load_policy("results/models/best.pt")

# Main control loop
try:
    while True:
        # Get robot state
        joint_angles = controller.get_joint_angles()
        tcp_pose = controller.get_tcp_pose()
        
        # Create state observation
        state = np.concatenate([
            joint_angles,
            tcp_pose.flatten()
        ])
        
        # Compute action
        action = policy.compute_action(state)
        
        # Apply action (with safety limits)
        safe_action = apply_safety_limits(action)
        controller.set_joint_velocities(safe_action)
        
        # Check status
        if controller.emergency_stop_pressed():
            break
            
except Exception as e:
    print(f"Error: {e}")
finally:
    controller.disconnect()
```

## Crazyflie Drone Integration

```python
from hybrid_gcs.environments import CrazyflieDrone
from crazyflie_py import Crazyflie
import numpy as np

# Connect to Crazyflie
cf = Crazyflie()
cf.connect_uri("radio://0/80/2M")

# Load policy
policy = load_policy("results/models/drone_policy.pt")

# Flight parameters
max_speed = 1.0  # m/s
hover_thrust = 0.5

# Control loop
for t in range(1000):
    # Get state
    state = np.array([
        cf.position[0],
        cf.position[1],
        cf.position[2],
        cf.velocity[0],
        cf.velocity[1],
        cf.velocity[2],
    ])
    
    # Compute action
    action = policy.compute_action(state)
    
    # Apply thrust
    thrust = hover_thrust + action[0] * 0.2
    cf.send_thrust_command(thrust)
    
    # Handle crashes
    if cf.position[2] < 0.1:
        cf.send_emergency_stop()
        break

cf.disconnect()
```

## Safety Considerations

### Emergency Stop

```python
class SafetyMonitor:
    """Monitor safety during deployment."""
    
    def __init__(self, robot_controller):
        """Initialize safety monitor."""
        self.robot = robot_controller
        self.velocity_limit = 1.0  # m/s
        self.position_limits = {
            'x': [-2, 2],
            'y': [-2, 2],
            'z': [0, 2]
        }
    
    def check_safety(self, state, action):
        """Check if action is safe."""
        # Check joint limits
        joint_angles = state[:6]
        if np.any(np.abs(joint_angles) > np.pi):
            return False, "Joint limit exceeded"
        
        # Check velocity
        if np.linalg.norm(action) > self.velocity_limit:
            return False, "Velocity limit exceeded"
        
        # Check position
        position = state[6:9]
        for axis, limits in self.position_limits.items():
            if not (limits[0] <= position[0] <= limits[1]):
                return False, f"Position limit exceeded on {axis}"
        
        return True, "Safe"
    
    def emergency_stop(self):
        """Trigger emergency stop."""
        self.robot.stop_all_motion()
        self.robot.disable_motors()
```

### Monitoring

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Monitor:
    """Monitor deployment."""
    
    def __init__(self, log_file):
        """Initialize monitor."""
        self.log_file = log_file
    
    def log_state(self, state, action, reward):
        """Log robot state."""
        logger.info(f"State: {state}, Action: {action}, Reward: {reward}")
        
        with open(self.log_file, 'a') as f:
            f.write(f"{state},{action},{reward}\n")
    
    def check_performance(self):
        """Check performance metrics."""
        # Read log file and compute metrics
        pass
```

## Integration Checklist

Before deployment:

- [ ] Model trained and validated
- [ ] Safety limits configured
- [ ] Emergency stop tested
- [ ] Monitoring system active
- [ ] Logging enabled
- [ ] Network connectivity verified
- [ ] Robot calibrated
- [ ] All sensors working

## Troubleshooting

### Robot Not Responding

```python
# Check connection
controller.ping()

# Reset connection
controller.disconnect()
controller.reconnect()
```

### Unstable Motion

```python
# Reduce action scale
action = policy.compute_action(state) * 0.5

# Enable safety filtering
action = safety_monitor.filter_action(action)
```

---

See [Deployment Guide](../deployment.md) for more options.
