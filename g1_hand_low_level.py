"""
Unitree G1 Hand Low-Level Control
Demonstrates correct SDK initialization and basic motor commands for the hand
"""

import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

# Import low-level commands if available (SDK may expose these differently)
# Adjust imports based on your SDK version
try:
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
    from unitree_sdk2py.comm.motion_switcher import MotionSwitcher
    print("✓ Low-level imports available")
except ImportError as e:
    print(f"Low-level imports not available: {e}")
    LowCmd_ = None


def initialize_sdk():
    """
    Initialize the Unitree SDK DDS subsystem
    MUST be called before creating any clients (SportClient, etc.)
    """
    print("Initializing Unitree SDK DDS subsystem...")
    try:
        # Initialize the ChannelFactory with domain ID (typically 0)
        ChannelFactoryInitialize(0)
        print("✓ SDK initialized successfully")
        return True
    except Exception as e:
        print(f"✗ SDK initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hand_commands():
    """
    Test basic hand motor commands
    This is a placeholder - actual commands depend on SDK version
    """
    print("\n=== Hand Command Test ===")
    
    # Hand joint indices (G1 specific - verify with your robot docs)
    # Typically hand joints are at higher indices (e.g., 20-29)
    LEFT_HAND_START = 20   # example
    RIGHT_HAND_START = 25  # example
    
    print("Hand motor control commands:")
    print("  - Joint position: cmd.q = <angle_rad>")
    print("  - Joint velocity: cmd.dq = <velocity_rad_s>")
    print("  - Joint torque: cmd.tau = <torque_Nm>")
    print("  - PD gains: cmd.Kp, cmd.Kd")
    
    # Example command structure (conceptual - adapt to your SDK)
    print("\nExample low-level command structure:")
    print("  motor_cmd = MotorCmd()")
    print("  motor_cmd.mode = 0x01  # Position/torque mode")
    print("  motor_cmd.q = 0.5      # Target angle (rad)")
    print("  motor_cmd.dq = 0.0     # Target velocity")
    print("  motor_cmd.tau = 0.0    # Feedforward torque")
    print("  motor_cmd.Kp = 10.0    # Position gain")
    print("  motor_cmd.Kd = 1.0     # Damping gain")
    
    return True


def basic_hand_movement_demo():
    """
    Demonstrates basic hand movement sequence
    NOTE: This is a safe demonstration - adapt for your specific hand
    """
    print("\n=== Basic Hand Movement Demo ===")
    
    # Common hand commands for G1
    commands = {
        "open_hand": {
            "description": "Open hand (extend fingers)",
            "joint_angles": [0.0, 0.0, 0.0, 0.0, 0.0],  # example
            "duration": 2.0
        },
        "close_hand": {
            "description": "Close hand (curl fingers)",
            "joint_angles": [1.2, 1.2, 1.2, 1.2, 1.2],  # example
            "duration": 2.0
        },
        "neutral": {
            "description": "Neutral position",
            "joint_angles": [0.5, 0.5, 0.5, 0.5, 0.5],  # example
            "duration": 1.5
        }
    }
    
    for cmd_name, cmd_data in commands.items():
        print(f"\n{cmd_data['description']}:")
        print(f"  Target angles: {cmd_data['joint_angles']}")
        print(f"  Duration: {cmd_data['duration']}s")
        # In real implementation: send_motor_commands(cmd_data['joint_angles'])
        time.sleep(0.5)
    
    return True


def main():
    """Main entry point"""
    print("=" * 60)
    print("Unitree G1 Hand Low-Level Control")
    print("=" * 60)
    
    # Step 1: Initialize SDK (CRITICAL - must be first)
    if not initialize_sdk():
        print("\n✗ Cannot proceed without SDK initialization")
        print("\nTroubleshooting:")
        print("1. Ensure cyclonedds.xml is valid or use default config")
        print("2. Check network interface is up")
        print("3. Verify SDK installation: pip3 show unitree-sdk2py")
        return 1
    
    # Step 2: Test command knowledge
    test_hand_commands()
    
    # Step 3: Demo movement sequence
    basic_hand_movement_demo()
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
