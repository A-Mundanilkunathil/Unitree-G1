"""
Unitree G1 High-Level Control
Provides high-level commands for walking, standing, and movement control
"""

import time
import numpy as np
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.go2.sport.sport_client import SportClient


class G1HighLevelController:
    """High-level controller for Unitree G1 robot"""
    
    def __init__(self):
        """Initialize the high-level controller"""
        self.sport_client = SportClient()
        self.sport_client.Init()
        
        self.state = None
        self.is_connected = False
        
        # Movement parameters
        self.max_velocity = 1.5  # m/s
        self.max_yaw_speed = 1.0  # rad/s
        
        print("G1 High-Level Controller initialized")
    
    def connect(self):
        """Connect to the robot"""
        try:
            self.is_connected = True
            print("Connected to G1 robot")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    def stand_up(self):
        """Command robot to stand up"""
        try:
            self.sport_client.StandUp()
            print("Standing up...")
            return True
        except Exception as e:
            print(f"Stand up failed: {e}")
            return False
    
    def stand_down(self):
        """Command robot to sit/lie down"""
        try:
            self.sport_client.StandDown()
            print("Standing down...")
            return True
        except Exception as e:
            print(f"Stand down failed: {e}")
            return False
    
    def damp(self):
        """Enable damping mode (passive)"""
        try:
            self.sport_client.Damp()
            print("Damping mode enabled")
            return True
        except Exception as e:
            print(f"Damp mode failed: {e}")
            return False
    
    def balance_stand(self):
        """Enable balance stand mode"""
        try:
            self.sport_client.BalanceStand()
            print("Balance stand mode enabled")
            return True
        except Exception as e:
            print(f"Balance stand failed: {e}")
            return False
    
    def move(self, vx=0.0, vy=0.0, vyaw=0.0):
        """
        Move the robot with specified velocities
        
        Args:
            vx: Forward/backward velocity (m/s), positive = forward
            vy: Left/right velocity (m/s), positive = left
            vyaw: Yaw rotation velocity (rad/s), positive = counter-clockwise
        """
        try:
            # Clamp velocities to safe limits
            vx = np.clip(vx, -self.max_velocity, self.max_velocity)
            vy = np.clip(vy, -self.max_velocity, self.max_velocity)
            vyaw = np.clip(vyaw, -self.max_yaw_speed, self.max_yaw_speed)
            
            self.sport_client.Move(vx, vy, vyaw)
            return True
        except Exception as e:
            print(f"Move command failed: {e}")
            return False
    
    def stop_move(self):
        """Stop all movement"""
        return self.move(0.0, 0.0, 0.0)
    
    def pose(self, body_height=0.0, roll=0.0, pitch=0.0, yaw=0.0):
        """
        Set body pose
        
        Args:
            body_height: Body height adjustment (m), range: -0.18 to 0.03
            roll: Roll angle (rad)
            pitch: Pitch angle (rad)
            yaw: Yaw angle (rad)
        """
        try:
            self.sport_client.Pose(body_height, roll, pitch, yaw)
            return True
        except Exception as e:
            print(f"Pose command failed: {e}")
            return False
    
    def switch_gait(self, gait_type=0):
        """
        Switch gait pattern
        
        Args:
            gait_type: 0=idle, 1=trot, 2=trot running, 3=climb stair, 4=trot obstacle
        """
        try:
            self.sport_client.SwitchGait(gait_type)
            print(f"Switched to gait type: {gait_type}")
            return True
        except Exception as e:
            print(f"Switch gait failed: {e}")
            return False
    
    def get_state(self):
        """Get current robot state"""
        try:
            # This would return state information
            # Implementation depends on SDK version
            return self.state
        except Exception as e:
            print(f"Get state failed: {e}")
            return None
    
    def recover_stand(self):
        """Recovery stand after falling"""
        try:
            self.sport_client.RecoveryStand()
            print("Executing recovery stand...")
            return True
        except Exception as e:
            print(f"Recovery stand failed: {e}")
            return False
    
    def euler_stand(self):
        """Stand with Euler angle control"""
        try:
            self.sport_client.Euler()
            print("Euler stand mode enabled")
            return True
        except Exception as e:
            print(f"Euler stand failed: {e}")
            return False
    
    def heart_beat(self):
        """Send heartbeat to maintain connection"""
        try:
            self.sport_client.HeartBeat()
            return True
        except Exception as e:
            print(f"Heartbeat failed: {e}")
            return False
    
    def set_velocity_limits(self, max_vel=1.5, max_yaw=1.0):
        """Set maximum velocity limits"""
        self.max_velocity = max_vel
        self.max_yaw_speed = max_yaw
        print(f"Velocity limits set: {max_vel} m/s, {max_yaw} rad/s")
    
    def walk_forward(self, speed=0.3, duration=2.0):
        """Walk forward for specified duration"""
        print(f"Walking forward at {speed} m/s for {duration} seconds")
        self.move(vx=speed)
        time.sleep(duration)
        self.stop_move()
    
    def walk_backward(self, speed=0.3, duration=2.0):
        """Walk backward for specified duration"""
        print(f"Walking backward at {speed} m/s for {duration} seconds")
        self.move(vx=-speed)
        time.sleep(duration)
        self.stop_move()
    
    def strafe_left(self, speed=0.2, duration=2.0):
        """Strafe left for specified duration"""
        print(f"Strafing left at {speed} m/s for {duration} seconds")
        self.move(vy=speed)
        time.sleep(duration)
        self.stop_move()
    
    def strafe_right(self, speed=0.2, duration=2.0):
        """Strafe right for specified duration"""
        print(f"Strafing right at {speed} m/s for {duration} seconds")
        self.move(vy=-speed)
        time.sleep(duration)
        self.stop_move()
    
    def turn_left(self, speed=0.3, duration=2.0):
        """Turn left for specified duration"""
        print(f"Turning left at {speed} rad/s for {duration} seconds")
        self.move(vyaw=speed)
        time.sleep(duration)
        self.stop_move()
    
    def turn_right(self, speed=0.3, duration=2.0):
        """Turn right for specified duration"""
        print(f"Turning right at {speed} rad/s for {duration} seconds")
        self.move(vyaw=-speed)
        time.sleep(duration)
        self.stop_move()
    
    def circle_walk(self, radius=1.0, speed=0.3, duration=5.0):
        """Walk in a circle"""
        print(f"Walking in circle with radius {radius}m")
        # Calculate angular velocity for circular motion
        vyaw = speed / radius
        self.move(vx=speed, vyaw=vyaw)
        time.sleep(duration)
        self.stop_move()
    
    def emergency_stop(self):
        """Emergency stop - damp all motors"""
        print("EMERGENCY STOP!")
        self.stop_move()
        time.sleep(0.1)
        self.damp()
    
    def shutdown(self):
        """Safely shutdown the controller"""
        print("Shutting down controller...")
        self.stop_move()
        time.sleep(0.5)


# Convenience functions for quick commands
def quick_test():
    """Quick test of basic functions"""
    controller = G1HighLevelController()
    controller.connect()
    
    print("\n=== Quick Test Sequence ===")
    
    # Stand up
    controller.stand_up()
    time.sleep(3)
    
    # Walk forward
    controller.walk_forward(speed=0.2, duration=2)
    time.sleep(1)
    
    # Stop and stand down
    controller.stand_down()
    time.sleep(2)
    
    controller.shutdown()
    print("Test complete!")


if __name__ == "__main__":
    quick_test()
