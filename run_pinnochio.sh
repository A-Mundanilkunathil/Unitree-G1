#!/bin/bash
# Wrapper to run g1_mediapipe_arms_pinocchio.py with proper environment

# Use unitree SDK's bundled cyclonedds (remove CYCLONEDDS_HOME so it auto-detects)
export LD_LIBRARY_PATH=/home/unitree/unitree_sdk2/thirdparty/lib/aarch64:$LD_LIBRARY_PATH

# Add unitree SDK to Python path
export PYTHONPATH=/home/unitree/unitree_sdk2_python:$PYTHONPATH

# Run with micromamba tv environment python
exec $HOME/.local/share/mamba/envs/tv/bin/python3 /home/unitree/unitree_sdk2_python/g1_mediapipe_arms_pinocchio.py
