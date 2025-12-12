# Deployment Guide

Deploy Hybrid-GCS to production environments.

## Local Deployment

```bash
# 1. Install package
pip install hybrid-gcs

# 2. Verify
python -c "from hybrid_gcs import GCSDecomposer; print('✅ Ready')"

# 3. Run training
python scripts/train.py --config configs/training/production.yaml

# 4. Evaluate
python scripts/evaluate.py --model results/models/best.pt

# 5. Deploy
python scripts/deploy.py --deploy-dir production/
```

## Docker Deployment

```bash
# Build image
docker build -t hybrid-gcs:2.0.0 .

# Run container
docker run -it hybrid-gcs:2.0.0 \
    python scripts/train.py --env manipulation

# Or use Docker Compose
docker-compose up -d

# Monitor
docker-compose logs -f hybrid-gcs
```

## Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hybrid-gcs
spec:
  replicas: 1
  selector:
    matchLabels:
      app: hybrid-gcs
  template:
    metadata:
      labels:
        app: hybrid-gcs
    spec:
      containers:
      - name: hybrid-gcs
        image: hybrid-gcs:2.0.0
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: models
          mountPath: /app/models
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
      - name: logs
        persistentVolumeClaim:
          claimName: logs-pvc
```

## Real Robot Integration

### UR5 Robot

```python
from hybrid_gcs.integration import RealRobotController
from ur_robot_driver.ur_driver import URDriver

# Connect to robot
robot = URDriver("192.168.1.100")

# Create controller
controller = RealRobotController(robot)

# Execute policy
state = controller.get_state()
action = policy.compute_action(state)
controller.execute_action(action)
```

### Crazyflie Drone

```python
from hybrid_gcs.environments import CrazyflieDrone
from crazyflie_py import Crazyflie

# Connect to drone
cf = Crazyflie()
cf.connect_uri("radio://0/80/2M")

# Execute policy
action = policy.compute_action(state)
cf.send_command(action)
```

## Monitoring & Logging

```bash
# Enable logging
export LOG_LEVEL=INFO

# Monitor in real-time
tail -f results/logs/deployment.log

# TensorBoard
tensorboard --logdir results/logs/

# Health check
curl http://localhost:5000/health
```

## Safety Checklist

Before deployment:

- [ ] Model trained and evaluated
- [ ] Safety mechanisms enabled
- [ ] Collision detection working
- [ ] Emergency stop functional
- [ ] Velocity limits enforced
- [ ] Torque limits set
- [ ] Logging enabled
- [ ] Monitoring active

## Scaling

For multi-robot deployment:

```bash
# Run multiple instances
for i in {1..4}; do
    python scripts/deploy.py --robot-id $i --deploy-dir deployment/$i &
done
```

## Troubleshooting

### Connection Issues
```bash
# Check connectivity
ping robot_ip
# Check network
netstat -an | grep ESTABLISHED
```

### Performance Issues
```bash
# Check system resources
top -b -n 1
# Check GPU usage
nvidia-smi
```

---

See [Troubleshooting](troubleshooting.md) for common issues.
