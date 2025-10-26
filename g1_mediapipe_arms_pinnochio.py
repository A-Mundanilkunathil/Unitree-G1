"""
G1 MediaPipe Arm Control with Pinocchio IK
Uses legitimate inverse kinematics via Pinocchio library
"""

import time
import sys
import math
import numpy as np
import cv2
import threading
import os

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

try:
    import mediapipe as mp
except Exception:
    print("MediaPipe not installed. Install with: pip install mediapipe")
    sys.exit(1)

try:
    import pinocchio as pin
except Exception:
    print("Pinocchio not found. Run: micromamba activate tv")
    print("Or install with: micromamba create -n tv python=3.10 pinocchio=3.1.0 casadi=3.6.5 numpy=1.26.4 -c conda-forge")
    sys.exit(1)

G1_NUM_MOTOR = 29

Kp = [
    60, 60, 60, 100, 40, 40,      # legs
    60, 60, 60, 100, 40, 40,      # legs
    60, 40, 40,                   # waist
    20, 20, 20, 20,  20, 20, 20,  # arms (lower for smoother motion)
    20, 20, 20, 20,  20, 20, 20   # arms
]

Kd = [
    1, 1, 1, 2, 1, 1,     # legs
    1, 1, 1, 2, 1, 1,     # legs
    1, 1, 1,              # waist
    2, 2, 2, 2, 2, 2, 2,  # arms (higher damping for smoothness)
    2, 2, 2, 2, 2, 2, 2   # arms 
]

class G1JointIndex:
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristYaw = 19
    LeftWristRoll = 20
    LeftWristPitch = 21
    
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristYaw = 26
    RightWristRoll = 27
    RightWristPitch = 28

class PinocchioRightArmIK:
    """Inverse kinematics solver for G1 right arm using Pinocchio"""
    
    def __init__(self, urdf_path=None):
        self.urdf_path = urdf_path or "/home/unitree/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"
        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"URDF not found at {self.urdf_path}")

        # Load full model and create reduced model with only right arm
        self.model_full = pin.buildModelFromUrdf(self.urdf_path)
        self.q_reference = pin.neutral(self.model_full)

        keep_joints = {
            'universe',
            'right_shoulder_pitch_joint',
            'right_shoulder_roll_joint',
            'right_shoulder_yaw_joint',
            'right_elbow_joint',
            'right_wrist_roll_joint',
            'right_wrist_pitch_joint',
            'right_wrist_yaw_joint',
        }

        lock_ids = [
            self.model_full.getJointId(name)
            for name in self.model_full.names
            if name not in keep_joints
        ]

        self.model = pin.buildReducedModel(self.model_full, lock_ids, self.q_reference)
        self.data = self.model.createData()
        self.q_current = pin.neutral(self.model)

        # Get frame IDs for end effectors
        self.frame_wrist = self.model.getFrameId('right_wrist_yaw_link')
        self.frame_elbow = self.model.getFrameId('right_elbow_joint')
        self.shoulder_joint = self.model.getJointId('right_shoulder_pitch_joint')

        # Compute initial FK to get arm segment lengths
        pin.forwardKinematics(self.model, self.data, self.q_current)
        pin.updateFramePlacements(self.model, self.data)

        shoulder_pos = self.data.oMi[self.shoulder_joint].translation.copy()
        elbow_pos = self.data.oMf[self.frame_elbow].translation.copy()
        wrist_pos = self.data.oMf[self.frame_wrist].translation.copy()

        self.shoulder_origin = shoulder_pos
        self.upper_len = np.linalg.norm(elbow_pos - shoulder_pos)  # ~0.193m
        self.fore_len = np.linalg.norm(wrist_pos - elbow_pos)      # ~0.184m

        print(f"✓ Pinocchio IK initialized:")
        print(f"  Upper arm length: {self.upper_len:.3f}m")
        print(f"  Forearm length: {self.fore_len:.3f}m")
        print(f"  Shoulder origin: {self.shoulder_origin}")

        self.last_elbow_target = elbow_pos
        self.last_wrist_target = wrist_pos

        self.lower_limits = np.array(self.model.lowerPositionLimit, dtype=float)
        self.upper_limits = np.array(self.model.upperPositionLimit, dtype=float)

        # IK solver parameters
        self.damping = 1e-3
        self.max_step = 0.15
        self.max_iters = 80
        self.tol = 1e-3

    def _map_direction(self, vec):
        """Map MediaPipe coordinate system to robot coordinate system"""
        norm = np.linalg.norm(vec)
        if norm < 1e-5:
            return None
        hv = vec / norm
        # MediaPipe: X=right, Y=down, Z=toward_camera
        # Robot: X=forward, Y=left, Z=up
        # Mapping: robot_x=-mp_z, robot_y=mp_x, robot_z=-mp_y
        mapped = np.array([-hv[2], hv[0], -hv[1]])
        mapped_norm = np.linalg.norm(mapped)
        if mapped_norm < 1e-5:
            return None
        return mapped / mapped_norm

    def _targets_from_landmarks(self, shoulder, elbow, wrist):
        """Convert MediaPipe landmarks to robot frame target positions"""
        upper_dir = self._map_direction(elbow - shoulder)
        fore_dir = self._map_direction(wrist - elbow)
        if upper_dir is None or fore_dir is None:
            return self.last_elbow_target, self.last_wrist_target

        # Reconstruct arm in robot frame with proper segment lengths
        elbow_target = self.shoulder_origin + upper_dir * self.upper_len
        wrist_target = elbow_target + fore_dir * self.fore_len

        self.last_elbow_target = elbow_target
        self.last_wrist_target = wrist_target
        return elbow_target, wrist_target

    def solve(self, shoulder, elbow, wrist):
        """Solve IK for given MediaPipe landmark positions"""
        target_elbow, target_wrist = self._targets_from_landmarks(shoulder, elbow, wrist)
        q = self.q_current.copy()
        success = False

        # Damped least squares IK
        for iteration in range(self.max_iters):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            wrist_curr = self.data.oMf[self.frame_wrist].translation.copy()
            elbow_curr = self.data.oMf[self.frame_elbow].translation.copy()
            
            # Combined error: prioritize wrist, use elbow as guide
            error = np.hstack((wrist_curr - target_wrist, 0.5 * (elbow_curr - target_elbow)))

            if np.linalg.norm(error) < self.tol:
                success = True
                break

            # Compute Jacobians
            J_wrist = pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                self.frame_wrist,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:3, :]

            J_elbow = pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                self.frame_elbow,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:3, :]

            # Stack Jacobians (wrist + damped elbow)
            J = np.vstack((J_wrist, 0.5 * J_elbow))
            
            # Damped least squares: delta_q = J^T (J J^T + lambda I)^-1 error
            A = J @ J.T + self.damping * np.eye(J.shape[0])
            delta = -J.T @ np.linalg.solve(A, error)

            # Limit step size
            step_norm = np.linalg.norm(delta)
            if step_norm > self.max_step:
                delta *= self.max_step / (step_norm + 1e-9)

            # Update and clamp to joint limits
            q = np.clip(q + delta, self.lower_limits, self.upper_limits)

        if not success:
            # Final FK even if not converged
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

        self.q_current = q
        return q

    def to_command_dict(self, q):
        """Convert joint configuration to command dictionary"""
        return {
            'R_sh_pitch': float(q[0]),
            'R_sh_roll': float(q[1]),
            'R_sh_yaw': float(q[2]),
            'R_elbow': float(q[3]),
            'R_wrist_roll': float(q[4]),
            'R_wrist_pitch': float(q[5]),
            'R_wrist_yaw': float(q[6]),
        }

class MediaPipeArmController:
    def __init__(self, camera_index=6):
        self.camera_index = camera_index
        self.time_ = 0.0
        self.control_dt_ = 0.002  # 2ms like low-level example
        self.mode_machine_ = 0
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.update_mode_machine_ = False
        self.crc = CRC()
        
        # Initialize Pinocchio IK solver
        self.ik_solver = PinocchioRightArmIK()
        
        # MediaPipe target commands (will be updated by camera thread)
        self.target_cmds = self.ik_solver.to_command_dict(self.ik_solver.q_current)
        
        # Current actual commands being sent (for smooth interpolation)
        self.current_cmds = None
        self.initialized = False
        
        # Max velocity for each joint (radians per second)
        self.max_velocities = {
            'R_sh_pitch': 2.0,
            'R_sh_roll': 2.0,
            'R_sh_yaw': 1.5,
            'R_elbow': 2.5,
            'R_wrist_roll': 2.0,
            'R_wrist_pitch': 2.0,
            'R_wrist_yaw': 2.0,
        }
        
        self.has_pose = False
        self.lock = threading.Lock()
        
        # For visualization
        self.display_angles = {}
        
        # MediaPipe pose with LOWER thresholds for better detection
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        self.camera_running = False

    def Init(self):
        """Initialize SDK (like low-level example)"""
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result['name']:
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

        # create publisher
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        # create subscriber
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

    def Start(self):
        """Start control loop (like low-level example)"""
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=self.control_dt_, target=self.LowCmdWrite, name="control"
        )
        
        # Wait for first state message
        while self.update_mode_machine_ == False:
            time.sleep(0.1)

        if self.update_mode_machine_ == True:
            self.lowCmdWriteThreadPtr.Start()
            print("✓ Motor control thread started")

    def LowStateHandler(self, msg: LowState_):
        """Handle state updates (like low-level example)"""
        self.low_state = msg
        if self.update_mode_machine_ == False:
            self.mode_machine_ = self.low_state.mode_machine
            self.update_mode_machine_ = True
            print(f"✓ Robot mode_machine: {self.mode_machine_}")
            print(f"✓ Right arm current positions: "
                  f"ShPitch={msg.motor_state[G1JointIndex.RightShoulderPitch].q:.3f} "
                  f"ShRoll={msg.motor_state[G1JointIndex.RightShoulderRoll].q:.3f} "
                  f"Elbow={msg.motor_state[G1JointIndex.RightElbow].q:.3f}")
            
            # Initialize to neutral position from IK solver
            home_position = self.ik_solver.to_command_dict(self.ik_solver.q_current)
            
            self.current_cmds = home_position.copy()
            # Also initialize targets to home position
            with self.lock:
                self.target_cmds = home_position.copy()
            self.initialized = True
            print(f"✓ Initialized to HOME position (Pinocchio neutral)")
            print(f"✓ Robot will smoothly move to home, then track your pose...")

    def process_landmarks(self, landmarks):
        """Process MediaPipe landmarks and calculate RIGHT ARM joint angles using Pinocchio IK"""
        lm = landmarks
        
        R_SHOULDER = 12
        R_ELBOW = 14
        R_WRIST = 16

        def to_xyz(p):
            return np.array([p.x, p.y, p.z], dtype=float)

        r_sh = to_xyz(lm[R_SHOULDER])
        r_el = to_xyz(lm[R_ELBOW])
        r_wr = to_xyz(lm[R_WRIST])

        # Solve IK using Pinocchio
        q = self.ik_solver.solve(r_sh, r_el, r_wr)
        if q is None:
            return None

        commands = self.ik_solver.to_command_dict(q)

        # Store for display (in degrees for readability)
        self.display_angles = {
            'sh_pitch': math.degrees(commands['R_sh_pitch']),
            'sh_roll': math.degrees(commands['R_sh_roll']),
            'sh_yaw': math.degrees(commands['R_sh_yaw']),
            'elbow': math.degrees(commands['R_elbow']),
            'wr_roll': math.degrees(commands['R_wrist_roll']),
            'wr_pitch': math.degrees(commands['R_wrist_pitch']),
            'wr_yaw': math.degrees(commands['R_wrist_yaw']),
        }

        return commands

    def camera_thread_func(self):
        """Run camera capture in separate thread"""
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"✗ Cannot open camera {self.camera_index}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        time.sleep(1.0)
        
        print(f"✓ Camera {self.camera_index} opened (640x480 @ 30fps)")
        
        has_display = os.environ.get('DISPLAY') is not None
        if has_display:
            print("✓ Visualization window enabled")
        else:
            print("✓ Running headless, angles printed to console")
        
        self.camera_running = True
        frame_count = 0
        no_pose_count = 0

        while self.camera_running:
            ret, frame = cap.read()
            if not ret:
                print("✗ Failed to read frame")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                no_pose_count = 0
                
                cmds = self.process_landmarks(results.pose_landmarks.landmark)
                if cmds:
                    with self.lock:
                        self.target_cmds = cmds
                        self.has_pose = True
                
                if frame_count % 15 == 0:
                    pitch_dir = "FWD" if self.display_angles.get('sh_pitch', 0) > 10 else "BACK" if self.display_angles.get('sh_pitch', 0) < -10 else "SIDE"
                    roll_val = self.display_angles.get('sh_roll', 0)
                    roll_dir = "UP" if roll_val > 45 else "MID" if roll_val > 15 else "DOWN"
                    
                    print(f"👁 [Pinocchio IK] | "
                          f"Pitch={self.display_angles.get('sh_pitch', 0):6.1f}° ({pitch_dir}) "
                          f"Roll={roll_val:6.1f}° ({roll_dir}) "
                          f"Elbow={self.display_angles.get('elbow', 0):6.1f}°")
                    
                if has_display:
                    try:
                        self.mp_drawing.draw_landmarks(
                            frame,
                            results.pose_landmarks,
                            self.mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                        )
                        
                        y_offset = 30
                        for joint_name, angle in self.display_angles.items():
                            text = f"{joint_name}: {angle:.1f}°"
                            cv2.putText(frame, text, (10, y_offset), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            y_offset += 25
                        
                        cv2.imshow('MediaPipe + Pinocchio IK', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            self.camera_running = False
                            break
                    except Exception as e:
                        has_display = False
                        print(f"Display disabled: {e}")
            else:
                with self.lock:
                    self.has_pose = False
                
                no_pose_count += 1
                if no_pose_count % 60 == 0:
                    print("❌ No pose detected - robot holding position")

        cap.release()
        if has_display:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        print("✓ Camera closed")

    def LowCmdWrite(self):
        """Write low-level commands at 500Hz with velocity limiting"""
        if self.low_state is None or not self.initialized:
            return

        with self.lock:
            targets = self.target_cmds.copy()
            has_pose = self.has_pose
        
        if not has_pose:
            targets = self.current_cmds.copy()
        
        # Velocity-limited interpolation
        for joint_name in targets:
            target = targets[joint_name]
            current = self.current_cmds[joint_name]
            max_vel = self.max_velocities[joint_name]
            
            max_delta = max_vel * self.control_dt_
            delta = target - current
            
            if delta > max_delta:
                delta = max_delta
            elif delta < -max_delta:
                delta = -max_delta
            
            self.current_cmds[joint_name] = current + delta
        
        # CRITICAL: Initialize all motors with current state positions
        for i in range(G1_NUM_MOTOR):
            self.low_cmd.motor_cmd[i].mode = 0x01
            self.low_cmd.motor_cmd[i].q = float(self.low_state.motor_state[i].q)
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = Kp[i]
            self.low_cmd.motor_cmd[i].kd = Kd[i]
            self.low_cmd.motor_cmd[i].tau = 0.0
        
        self.low_cmd.mode_pr = self.mode_machine_
        self.low_cmd.mode_machine = self.mode_machine_

        # Override RIGHT ARM with Pinocchio IK commands
        self.low_cmd.motor_cmd[G1JointIndex.RightShoulderPitch].q = float(self.current_cmds['R_sh_pitch'])
        self.low_cmd.motor_cmd[G1JointIndex.RightShoulderRoll].q = float(self.current_cmds['R_sh_roll'])
        self.low_cmd.motor_cmd[G1JointIndex.RightShoulderYaw].q = float(self.current_cmds['R_sh_yaw'])
        self.low_cmd.motor_cmd[G1JointIndex.RightElbow].q = float(self.current_cmds['R_elbow'])
        self.low_cmd.motor_cmd[G1JointIndex.RightWristRoll].q = float(self.current_cmds['R_wrist_roll'])
        self.low_cmd.motor_cmd[G1JointIndex.RightWristPitch].q = float(self.current_cmds['R_wrist_pitch'])
        self.low_cmd.motor_cmd[G1JointIndex.RightWristYaw].q = float(self.current_cmds['R_wrist_yaw'])

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)

if __name__ == '__main__':
    print("=" * 70)
    print("G1 MediaPipe + Pinocchio IK - RIGHT ARM Control")
    print("=" * 70)
    print("FEATURES:")
    print("  ✓ Legitimate inverse kinematics via Pinocchio library")
    print("  ✓ Damped least squares IK solver")
    print("  ✓ Proper coordinate system transformation")
    print("  ✓ Uses G1 URDF model with accurate segment lengths")
    print()
    print("IMPORTANT: Stand facing the camera naturally.")
    print("  - Raise YOUR RIGHT ARM → Robot raises its right arm")
    print("  - Move forward → Robot arm moves forward")
    print("  - Raise to side → Robot arm raises to side")
    print()
    print("Press Ctrl+C to stop.")
    print("=" * 70)
    print()
    input("Press Enter to start...")
    
    ChannelFactoryInitialize(0)
    
    controller = MediaPipeArmController(camera_index=6)
    controller.Init()
    controller.Start()
    
    camera_thread = threading.Thread(target=controller.camera_thread_func)
    camera_thread.start()
    
    print()
    print("✓ System running! Move your RIGHT arm...")
    print()
    
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down...")
        controller.camera_running = False
        camera_thread.join()
        print("Done!")
