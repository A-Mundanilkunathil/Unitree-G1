"""
G1 Punch Mirror - High-Level Motion Tracking
Uses OpenCV to track hand movements and mirrors them on the robot's arms
"""

import time
import sys
import threading
import cv2
import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

kPi = 3.141592654
kPi_2 = 1.57079632

class G1JointIndex:
    # Left leg
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5

    # Right leg
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11

    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14

    # Left arm
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21

    # Right arm
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28

    kNotUsedJoint = 29  # Weight for arm_sdk control

class HandTracker:
    """OpenCV-based hand tracking for punch mirroring"""
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        self.running = False
        self.lock = threading.Lock()
        
        # Track hand positions (normalized to -1 to 1)
        self.left_hand_pos = (0.0, 0.0)
        self.right_hand_pos = (0.0, 0.0)
        
        # Color tracking for hands (red/orange)
        self.hand_color_lower = np.array([0, 100, 100])
        self.hand_color_upper = np.array([20, 255, 255])
        
    def update_hands(self, left_pos, right_pos):
        """Update tracked hand positions"""
        with self.lock:
            self.left_hand_pos = left_pos
            self.right_hand_pos = right_pos
    
    def get_hands(self):
        """Get current hand positions"""
        with self.lock:
            return self.left_hand_pos, self.right_hand_pos
    
    def track_frame(self, frame):
        """Track hands in frame"""
        if frame is None:
            return None, None
            
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hand_color_lower, self.hand_color_upper)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hands = []
        hand_positions = []
        
        # Sort by area and take top 2
        areas = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        
        h, w = frame.shape[:2]
        
        for contour in areas:
            if cv2.contourArea(contour) > 500:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Normalize to [-1, 1]
                    norm_x = 2.0 * (cx / w) - 1.0
                    norm_y = 2.0 * (cy / h) - 1.0
                    
                    hand_positions.append((norm_x, norm_y))
                    hands.append((cx, cy))
        
        # Determine left and right
        left_hand = None
        right_hand = None
        
        if len(hand_positions) >= 1:
            sorted_hands = sorted(zip(hand_positions, hands), key=lambda x: x[0][0])
            if len(sorted_hands) >= 1:
                left_hand = sorted_hands[0][1]
                left_pos = sorted_hands[0][0]
            if len(sorted_hands) >= 2:
                right_hand = sorted_hands[-1][1]
                right_pos = sorted_hands[-1][0]
            elif len(sorted_hands) == 1:
                right_hand = sorted_hands[0][1]
                right_pos = sorted_hands[0][0]
                left_pos = (0.0, 0.0)
            
            if left_hand is not None:
                self.update_hands(left_pos, right_pos if right_hand else (0.0, 0.0))
        
        return left_hand, right_hand
    
    def run(self):
        """Run tracking loop"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"Failed to open camera {self.camera_id}")
            return
        
        self.running = True
        print("Hand tracking started. Wear red/orange gloves or hold colored markers.")
        print("Press 'q' to quit tracking.")
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            left_hand, right_hand = self.track_frame(frame)
            
            # Visualize
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), 2)
            
            if left_hand:
                cv2.circle(frame, left_hand, 20, (255, 0, 0), -1)
                cv2.putText(frame, "LEFT", (left_hand[0] + 25, left_hand[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            if right_hand:
                cv2.circle(frame, right_hand, 20, (0, 0, 255), -1)
                cv2.putText(frame, "RIGHT", (right_hand[0] + 25, right_hand[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Punch Mirror - Wear Red/Orange Gloves', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def stop(self):
        """Stop tracking"""
        self.running = False


class PunchMirror:
    """High-level punch mirroring control"""
    def __init__(self):
        self.time_ = 0.0
        self.control_dt_ = 0.02
        self.duration_ = 3.0
        self.counter_ = 0
        
        # Arm control parameters
        self.kp = 60.0
        self.kd = 1.5
        
        self.mode_machine_ = 0
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.first_update = False
        self.crc = CRC()
        
        # Hand tracking
        self.hand_tracker = HandTracker()
        self.tracking_thread = None
        self.use_tracking = True
        
        # Arm joints to control
        self.arm_joints = [
            G1JointIndex.LeftShoulderPitch, G1JointIndex.LeftShoulderRoll,
            G1JointIndex.LeftShoulderYaw, G1JointIndex.LeftElbow,
            G1JointIndex.LeftWristRoll,
            G1JointIndex.RightShoulderPitch, G1JointIndex.RightShoulderRoll,
            G1JointIndex.RightShoulderYaw, G1JointIndex.RightElbow,
            G1JointIndex.RightWristRoll,
            G1JointIndex.WaistYaw, G1JointIndex.WaistPitch
        ]

    def Init(self):
        """Initialize SDK and channels"""
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        # Release any existing mode
        status, result = self.msc.CheckMode()
        while result['name']:
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

        # Create publisher for arm SDK
        self.arm_sdk_publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.arm_sdk_publisher.Init()

        # Create subscriber for state
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

    def Start(self):
        """Start control and tracking"""
        # Start hand tracking
        if self.use_tracking:
            self.tracking_thread = threading.Thread(target=self.hand_tracker.run, daemon=True)
            self.tracking_thread.start()
            time.sleep(2)
        
        # Start control loop
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=self.control_dt_, target=self.LowCmdWrite, name="control"
        )
        
        while not self.first_update:
            time.sleep(0.1)
        
        if self.first_update:
            self.lowCmdWriteThreadPtr.Start()

    def LowStateHandler(self, msg: LowState_):
        """Handle low state updates"""
        self.low_state = msg
        
        if not self.first_update:
            self.first_update = True
    
    def LowCmdWrite(self):
        """Main control loop"""
        self.time_ += self.control_dt_

        # Stage 1: Initialize
        if self.time_ < self.duration_:
            self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1
            
            for i, joint in enumerate(self.arm_joints):
                ratio = np.clip(self.time_ / self.duration_, 0.0, 1.0)
                self.low_cmd.motor_cmd[joint].tau = 0.0
                self.low_cmd.motor_cmd[joint].q = (1.0 - ratio) * self.low_state.motor_state[joint].q
                self.low_cmd.motor_cmd[joint].dq = 0.0
                self.low_cmd.motor_cmd[joint].kp = self.kp
                self.low_cmd.motor_cmd[joint].kd = self.kd

        # Stage 2: Punch mirroring
        else:
            if self.use_tracking:
                left_hand, right_hand = self.hand_tracker.get_hands()
                
                # Max angles for smooth control
                max_sp = kPi * 60.0 / 180.0
                max_sr = kPi * 30.0 / 180.0
                max_sy = kPi * 45.0 / 180.0
                max_elbow = kPi * 90.0 / 180.0
                max_wrist = kPi * 30.0 / 180.0
                
                # Map right hand (human) -> robot's left arm
                if right_hand[0] != 0.0 or right_hand[1] != 0.0:
                    L_SP = kPi_2 - right_hand[1] * max_sp
                    L_SR = right_hand[0] * max_sr
                    L_SY = right_hand[0] * max_sy
                    L_ELBOW = kPi_2 + right_hand[1] * max_elbow * 0.5
                    L_WR = -right_hand[0] * max_wrist
                else:
                    L_SP = kPi_2
                    L_SR = 0
                    L_SY = 0
                    L_ELBOW = kPi_2
                    L_WR = 0
                
                # Map left hand (human) -> robot's right arm
                if left_hand[0] != 0.0 or left_hand[1] != 0.0:
                    R_SP = kPi_2 - left_hand[1] * max_sp
                    R_SR = -left_hand[0] * max_sr
                    R_SY = -left_hand[0] * max_sy
                    R_ELBOW = kPi_2 + left_hand[1] * max_elbow * 0.5
                    R_WR = left_hand[0] * max_wrist
                else:
                    R_SP = kPi_2
                    R_SR = 0
                    R_SY = 0
                    R_ELBOW = kPi_2
                    R_WR = 0
                
                # Apply arm positions
                self.low_cmd.motor_cmd[G1JointIndex.LeftShoulderPitch].q = L_SP
                self.low_cmd.motor_cmd[G1JointIndex.LeftShoulderRoll].q = L_SR
                self.low_cmd.motor_cmd[G1JointIndex.LeftShoulderYaw].q = L_SY
                self.low_cmd.motor_cmd[G1JointIndex.LeftElbow].q = L_ELBOW
                self.low_cmd.motor_cmd[G1JointIndex.LeftWristRoll].q = L_WR
                
                self.low_cmd.motor_cmd[G1JointIndex.RightShoulderPitch].q = R_SP
                self.low_cmd.motor_cmd[G1JointIndex.RightShoulderRoll].q = R_SR
                self.low_cmd.motor_cmd[G1JointIndex.RightShoulderYaw].q = R_SY
                self.low_cmd.motor_cmd[G1JointIndex.RightElbow].q = R_ELBOW
                self.low_cmd.motor_cmd[G1JointIndex.RightWristRoll].q = R_WR
                
                # Control waist
                waist_pitch = kPi_2 + (left_hand[1] + right_hand[1]) * 0.5 * kPi * 15.0 / 180.0
                waist_yaw = (left_hand[0] + right_hand[0]) * 0.5 * kPi * 20.0 / 180.0
                self.low_cmd.motor_cmd[G1JointIndex.WaistPitch].q = waist_pitch
                self.low_cmd.motor_cmd[G1JointIndex.WaistYaw].q = waist_yaw

        # Send command
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.arm_sdk_publisher.Write(self.low_cmd)


if __name__ == '__main__':
    print("=" * 60)
    print("G1 Punch Mirror - High-Level Motion Tracking")
    print("=" * 60)
    print("WARNING: Ensure no obstacles around the robot!")
    print("\nMirror Mode: Your right hand controls robot's left arm")
    print("             Your left hand controls robot's right arm")
    print("\nSetup: Wear red/orange gloves or hold colored markers")
    print("Press 'q' in tracking window to stop tracking")
    print("=" * 60)
    
    input("Press Enter to start...")

    # Parse network interface if provided
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    controller = PunchMirror()
    controller.Init()
    controller.Start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        controller.hand_tracker.stop()
        if controller.tracking_thread:
            controller.tracking_thread.join(timeout=2)
        print("Done!")

