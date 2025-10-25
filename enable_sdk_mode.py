#!/usr/bin/env python3
"""
Unitree G1 - Enable SDK Mode and Test Basic Commands

This script helps you verify the robot is ready for SDK control.
"""

import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient


def main():
    print("=" * 70)
    print("Unitree G1 - SDK Mode Verification")
    print("=" * 70)
    print()
    
    print("BEFORE running this script, ensure:")
    print("  1. Robot is powered on")
    print("  2. eth0 is connected to robot")
    print("  3. Robot is in SDK/Developer mode")
    print()
    print("HOW TO ENABLE SDK MODE:")
    print("  Method 1: Use Unitree app")
    print("    - Open Unitree app")
    print("    - Go to Settings → Control Mode")
    print("    - Select 'SDK Mode' or 'Developer Mode'")
    print()
    print("  Method 2: Use controller")
    print("    - Hold L2 + A buttons simultaneously")
    print("    - Wait for confirmation on robot display")
    print()
    print("  Method 3: Robot display/button")
    print("    - Check robot's onboard display")
    print("    - Navigate to SDK/Developer mode")
    print()
    
    input("Press ENTER when robot is in SDK mode...")
    print()
    
    # Initialize SDK
    print("Initializing SDK on eth0...")
    try:
        ChannelFactoryInitialize(0, "eth0")
        print("✓ SDK initialized")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return 1
    
    # Create client
    print("Creating SportClient...")
    try:
        client = SportClient()
        client.Init()
        print("✓ SportClient created")
    except Exception as e:
        print(f"✗ SportClient failed: {e}")
        return 1
    
    # Test heartbeat
    print("\nTesting communication with robot...")
    print("Sending 5 heartbeats (watch for errors):")
    
    errors = 0
    for i in range(5):
        try:
            client.HeartBeat()
            print(f"  ✓ Heartbeat {i+1}/5 sent")
            time.sleep(0.5)
        except Exception as e:
            errors += 1
            print(f"  ✗ Heartbeat {i+1}/5 failed: {e}")
    
    print()
    
    if errors > 0:
        print("⚠️  COMMUNICATION ERRORS DETECTED")
        print()
        print("Troubleshooting:")
        print("  1. [ClientStub] send error = Robot not in SDK mode")
        print("     → Enable SDK mode using app or controller")
        print()
        print("  2. Check robot display shows 'SDK Mode'")
        print()
        print("  3. Try rebooting robot and enabling SDK mode again")
        print()
        print("  4. Verify with working SDK examples:")
        print("     cd ~/unitree_sdk2_python/example")
        print("     python3 <example_script>.py")
        print()
        return 1
    else:
        print("✓ SUCCESS! Robot is responding to SDK commands")
        print()
        print("You can now run your control scripts.")
        print()
        
        # Optional: Test a simple command
        print("Testing stand command (robot should respond)...")
        try:
            client.StandUp()
            print("✓ Stand command sent")
            time.sleep(2)
        except Exception as e:
            print(f"✗ Stand command failed: {e}")
        
        return 0


if __name__ == "__main__":
    exit(main())
