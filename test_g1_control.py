"""
Test file for Unitree G1 High-Level Control
Run this to test various control functions
"""

import time
import sys
from g1_high_level_control import G1HighLevelController


def test_basic_movements(controller):
    """Test basic movement commands"""
    print("\n" + "="*50)
    print("TEST 1: Basic Movements")
    print("="*50)
    
    try:
        # Stand up
        print("\n1. Standing up...")
        controller.stand_up()
        time.sleep(4)
        
        # Walk forward
        print("\n2. Walking forward...")
        controller.walk_forward(speed=0.2, duration=3)
        time.sleep(1)
        
        # Walk backward
        print("\n3. Walking backward...")
        controller.walk_backward(speed=0.2, duration=3)
        time.sleep(1)
        
        # Stop
        print("\n4. Stopping...")
        controller.stop_move()
        time.sleep(1)
        
        print("\n✓ Basic movements test completed")
        return True
        
    except Exception as e:
        print(f"\n✗ Basic movements test failed: {e}")
        return False


def test_rotation(controller):
    """Test rotation commands"""
    print("\n" + "="*50)
    print("TEST 2: Rotation")
    print("="*50)
    
    try:
        # Turn left
        print("\n1. Turning left...")
        controller.turn_left(speed=0.3, duration=2)
        time.sleep(1)
        
        # Turn right
        print("\n2. Turning right...")
        controller.turn_right(speed=0.3, duration=2)
        time.sleep(1)
        
        # Stop
        controller.stop_move()
        time.sleep(1)
        
        print("\n✓ Rotation test completed")
        return True
        
    except Exception as e:
        print(f"\n✗ Rotation test failed: {e}")
        return False


def test_strafing(controller):
    """Test strafing movements"""
    print("\n" + "="*50)
    print("TEST 3: Strafing")
    print("="*50)
    
    try:
        # Strafe left
        print("\n1. Strafing left...")
        controller.strafe_left(speed=0.15, duration=2)
        time.sleep(1)
        
        # Strafe right
        print("\n2. Strafing right...")
        controller.strafe_right(speed=0.15, duration=2)
        time.sleep(1)
        
        # Stop
        controller.stop_move()
        time.sleep(1)
        
        print("\n✓ Strafing test completed")
        return True
        
    except Exception as e:
        print(f"\n✗ Strafing test failed: {e}")
        return False


def test_pose_control(controller):
    """Test body pose control"""
    print("\n" + "="*50)
    print("TEST 4: Pose Control")
    print("="*50)
    
    try:
        # Raise body
        print("\n1. Raising body height...")
        controller.pose(body_height=0.02)
        time.sleep(2)
        
        # Lower body
        print("\n2. Lowering body height...")
        controller.pose(body_height=-0.05)
        time.sleep(2)
        
        # Pitch forward
        print("\n3. Pitching forward...")
        controller.pose(pitch=0.1)
        time.sleep(2)
        
        # Return to neutral
        print("\n4. Returning to neutral pose...")
        controller.pose(body_height=0.0, pitch=0.0)
        time.sleep(2)
        
        print("\n✓ Pose control test completed")
        return True
        
    except Exception as e:
        print(f"\n✗ Pose control test failed: {e}")
        return False


def test_circle_walk(controller):
    """Test circular walking pattern"""
    print("\n" + "="*50)
    print("TEST 5: Circle Walk")
    print("="*50)
    
    try:
        print("\n1. Walking in a circle...")
        controller.circle_walk(radius=1.0, speed=0.25, duration=8)
        time.sleep(1)
        
        controller.stop_move()
        
        print("\n✓ Circle walk test completed")
        return True
        
    except Exception as e:
        print(f"\n✗ Circle walk test failed: {e}")
        return False


def test_gait_switching(controller):
    """Test different gait patterns"""
    print("\n" + "="*50)
    print("TEST 6: Gait Switching")
    print("="*50)
    
    try:
        # Trot gait
        print("\n1. Switching to trot gait...")
        controller.switch_gait(1)
        time.sleep(2)
        
        controller.walk_forward(speed=0.3, duration=3)
        time.sleep(1)
        
        # Back to idle
        print("\n2. Returning to idle gait...")
        controller.switch_gait(0)
        time.sleep(2)
        
        print("\n✓ Gait switching test completed")
        return True
        
    except Exception as e:
        print(f"\n✗ Gait switching test failed: {e}")
        return False


def test_balance_modes(controller):
    """Test different balance and standing modes"""
    print("\n" + "="*50)
    print("TEST 7: Balance Modes")
    print("="*50)
    
    try:
        # Balance stand
        print("\n1. Testing balance stand...")
        controller.balance_stand()
        time.sleep(3)
        
        # Euler stand
        print("\n2. Testing euler stand...")
        controller.euler_stand()
        time.sleep(3)
        
        print("\n✓ Balance modes test completed")
        return True
        
    except Exception as e:
        print(f"\n✗ Balance modes test failed: {e}")
        return False


def test_combined_movement(controller):
    """Test combined movements (diagonal, curved paths)"""
    print("\n" + "="*50)
    print("TEST 8: Combined Movement")
    print("="*50)
    
    try:
        # Diagonal forward-left
        print("\n1. Moving diagonally forward-left...")
        controller.move(vx=0.2, vy=0.1, vyaw=0.0)
        time.sleep(3)
        
        # Diagonal backward-right with rotation
        print("\n2. Moving diagonally backward-right with rotation...")
        controller.move(vx=-0.15, vy=-0.1, vyaw=0.2)
        time.sleep(3)
        
        # Stop
        controller.stop_move()
        time.sleep(1)
        
        print("\n✓ Combined movement test completed")
        return True
        
    except Exception as e:
        print(f"\n✗ Combined movement test failed: {e}")
        return False


def run_all_tests():
    """Run all test sequences"""
    print("\n" + "="*50)
    print("UNITREE G1 HIGH-LEVEL CONTROL TEST SUITE")
    print("="*50)
    print("\nInitializing controller...")
    
    controller = G1HighLevelController()
    
    # Connect to robot
    if not controller.connect():
        print("\n✗ Failed to connect to robot!")
        return
    
    print("\n⚠️  WARNING: Robot will move!")
    print("Make sure the robot has enough space and is safe to operate.")
    
    response = input("\nProceed with tests? (yes/no): ").lower()
    if response != 'yes':
        print("Tests cancelled.")
        return
    
    # Run tests
    tests = [
        test_basic_movements,
        test_rotation,
        test_strafing,
        test_pose_control,
        test_gait_switching,
        test_balance_modes,
        test_combined_movement,
        # test_circle_walk,  # Commented out as it takes longer
    ]
    
    results = []
    for i, test_func in enumerate(tests, 1):
        print(f"\n\nRunning test {i}/{len(tests)}...")
        try:
            result = test_func(controller)
            results.append((test_func.__name__, result))
        except KeyboardInterrupt:
            print("\n\n⚠️  Tests interrupted by user!")
            controller.emergency_stop()
            break
        except Exception as e:
            print(f"\n✗ Unexpected error in {test_func.__name__}: {e}")
            results.append((test_func.__name__, False))
    
    # Stand down at end
    print("\n\nTests complete. Standing down...")
    controller.stand_down()
    time.sleep(3)
    
    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    controller.shutdown()


def run_simple_test():
    """Run a simple quick test"""
    print("\n" + "="*50)
    print("SIMPLE TEST - Basic Stand and Walk")
    print("="*50)
    
    controller = G1HighLevelController()
    controller.connect()
    
    try:
        print("\n1. Standing up...")
        controller.stand_up()
        time.sleep(4)
        
        print("\n2. Walking forward for 2 seconds...")
        controller.walk_forward(speed=0.15, duration=2)
        time.sleep(1)
        
        print("\n3. Stopping...")
        controller.stop_move()
        time.sleep(1)
        
        print("\n4. Standing down...")
        controller.stand_down()
        time.sleep(3)
        
        print("\n✓ Simple test completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted!")
        controller.emergency_stop()
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        controller.emergency_stop()
    finally:
        controller.shutdown()


def interactive_mode():
    """Interactive control mode"""
    print("\n" + "="*50)
    print("INTERACTIVE CONTROL MODE")
    print("="*50)
    
    controller = G1HighLevelController()
    controller.connect()
    
    print("\nCommands:")
    print("  u - Stand up")
    print("  d - Stand down")
    print("  w - Walk forward")
    print("  s - Walk backward")
    print("  a - Turn left")
    print("  d - Turn right")
    print("  q - Strafe left")
    print("  e - Strafe right")
    print("  space - Stop")
    print("  x - Emergency stop")
    print("  h - Help")
    print("  exit - Quit")
    
    try:
        controller.stand_up()
        time.sleep(3)
        
        while True:
            cmd = input("\nEnter command: ").strip().lower()
            
            if cmd == 'exit':
                break
            elif cmd == 'u':
                controller.stand_up()
            elif cmd == 'd':
                controller.stand_down()
            elif cmd == 'w':
                controller.walk_forward(speed=0.2, duration=1)
            elif cmd == 's':
                controller.walk_backward(speed=0.2, duration=1)
            elif cmd == 'a':
                controller.turn_left(speed=0.3, duration=1)
            elif cmd == 'e':
                controller.turn_right(speed=0.3, duration=1)
            elif cmd == 'q':
                controller.strafe_left(speed=0.15, duration=1)
            elif cmd == 'r':
                controller.strafe_right(speed=0.15, duration=1)
            elif cmd == ' ' or cmd == 'space':
                controller.stop_move()
            elif cmd == 'x':
                controller.emergency_stop()
            elif cmd == 'h':
                print("\nCommands: u/d/w/s/a/e/q/r/space/x/h/exit")
            else:
                print("Unknown command. Type 'h' for help.")
    
    except KeyboardInterrupt:
        print("\n\nExiting interactive mode...")
    finally:
        controller.stand_down()
        time.sleep(2)
        controller.shutdown()


if __name__ == "__main__":
    print("\nUnitree G1 Control Test Options:")
    print("1. Run all tests")
    print("2. Run simple test")
    print("3. Interactive mode")
    print("4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1':
        run_all_tests()
    elif choice == '2':
        run_simple_test()
    elif choice == '3':
        interactive_mode()
    elif choice == '4':
        print("Exiting...")
    else:
        print("Invalid option. Running simple test...")
        run_simple_test()
