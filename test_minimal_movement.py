"""
Minimal G1 Movement Test
Tests if Move() commands work at all
"""

import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

print("="*60)
print("MINIMAL G1 MOVEMENT TEST")
print("="*60)

# Initialize SDK
print("\n1. Initializing SDK...")
ChannelFactoryInitialize(0, "eth0")
print("✓ SDK initialized")

# Create loco client (for G1 humanoid robot)
print("\n2. Creating LocoClient for G1...")
client = LocoClient()
client.SetTimeout(10.0)
client.Init()
print("✓ LocoClient created")

# Check if robot is already standing
print("\n3. Checking robot state...")
print("   ⚠️  Assuming robot is ALREADY STANDING")
print("   (If robot is NOT standing, manually stand it up first)")
time.sleep(1)
print("✓ Robot ready for movement commands")

# Test 1: Simple forward walk
print("\n4. TEST 1: Walk forward for 3 seconds...")
print("   Sending continuous Move(0.2, 0, 0) commands...")
start_time = time.time()
while time.time() - start_time < 3.0:
    client.Move(0.2, 0.0, 0.0)
    time.sleep(0.1)  # Send every 100ms
print("   ✓ Forward walk complete!")

# Stop
print("\n5. Stopping...")
client.StopMove()
time.sleep(2)  # Longer pause to ensure robot stops
print("✓ Should have stopped")

# Test 2: Strafe left (PURE strafe, no forward)
print("\n6. TEST 2: Strafe left for 3 seconds (NO FORWARD)...")
print("   Sending continuous Move(0, 0.3, 0) commands...")
print("   ⚠️  Watch carefully - robot should move LEFT only!")
start_time = time.time()
while time.time() - start_time < 3.0:
    client.Move(0.0, 0.3, 0.0)  # vx=0, vy=0.3 (left)
    time.sleep(0.1)  # Send every 100ms
print("   ✓ Strafe left complete!")

# Stop
print("\n7. Stopping...")
client.StopMove()
time.sleep(2)  # Longer pause
print("✓ Should have stopped")

# Test 3: Strafe right (opposite direction)
print("\n8. TEST 3: Strafe right for 3 seconds (NO FORWARD)...")
print("   Sending continuous Move(0, -0.3, 0) commands...")
print("   ⚠️  Watch carefully - robot should move RIGHT only!")
start_time = time.time()
while time.time() - start_time < 3.0:
    client.Move(0.0, -0.3, 0.0)  # vx=0, vy=-0.3 (right)
    time.sleep(0.1)  # Send every 100ms
print("   ✓ Strafe right complete!")

# Stop
print("\n9. Stopping...")
client.StopMove()
time.sleep(2)  # Longer pause
print("✓ Should have stopped")

# Test 4: Turn in place
print("\n10. TEST 4: Turn right for 2 seconds...")
print("    Sending continuous Move(0, 0, -0.3) commands...")
start_time = time.time()
while time.time() - start_time < 2.0:
    client.Move(0.0, 0.0, -0.3)
    time.sleep(0.1)  # Send every 100ms
print("    ✓ Turn right complete!")

# Stop
print("\n11. Stopping...")
client.StopMove()
time.sleep(1)
print("✓ Should have stopped")

print("\n" + "="*60)
print("TEST COMPLETE - Check robot movements!")
print("="*60)
print("\n✅ The robot should have:")
print("   1. Walked FORWARD (vx=+0.2)")
print("   2. Strafed LEFT (vy=+0.3)")
print("   3. Strafed RIGHT (vy=-0.3)")
print("   4. Turned RIGHT (vyaw=-0.3)")
print("\n💡 CRITICAL INSIGHTS:")
print("   - G1 needs CONTINUOUS Move() commands (send every 100ms)")
print("   - This test assumes robot is ALREADY STANDING")
print("   - If robot drops, it may have been in an unstable state")
print("   - For standup sequence, use: Damp() → Squat2StandUp() → wait 8s")
print("\n📋 COORDINATE SYSTEM (from robot's view):")
print("   • vx  > 0 → FORWARD  | vx  < 0 → BACKWARD")
print("   • vy  > 0 → LEFT     | vy  < 0 → RIGHT")
print("   • vyaw > 0 → Rotate LEFT | vyaw < 0 → Rotate RIGHT")




