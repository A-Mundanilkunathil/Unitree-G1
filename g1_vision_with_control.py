"""
Unitree G1 Vision + Movement Control
Combines vision detection with robot movement for object tracking and following
"""

import cv2
import numpy as np
import time
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from g1_vision_detection import G1VisionDetector

# Import SDK for robot control
try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    print("⚠️  Unitree SDK not available - robot control will be disabled")


class SimpleRobotController:
    """Simple robot controller using LocoClient for G1"""
    
    def __init__(self, domain_id=0, network_interface="eth0"):
        """Initialize robot controller"""
        if not SDK_AVAILABLE:
            raise ImportError("Unitree SDK not available")
        
        print(f"Initializing robot controller (domain={domain_id}, interface={network_interface})...")
        
        # Initialize SDK
        ChannelFactoryInitialize(domain_id, network_interface)
        print("✓ SDK initialized")
        
        # Create loco client for G1
        self.sport_client = LocoClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()
        print("✓ LocoClient created")
        
        self.max_velocity = 1.5  # m/s
        self.max_yaw_speed = 1.0  # rad/s
        self.current_head_pitch = 0.0  # Track head position
        self.current_head_yaw = 0.0
        self.is_standing = False
        
        # Movement command caching - for continuous sending
        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_vyaw = 0.0
        self.command_rate = 0.02  # Send commands every 20ms (50Hz)
    
    def stand_up(self):
        """Check if robot is standing - assumes already standing"""
        print("⚠️  Assuming robot is ALREADY STANDING")
        print("   (If not, manually stand up robot before running)")
        self.is_standing = True
        time.sleep(0.5)
        print("✓ Robot ready")
        return True
    
    def balance_stand(self):
        """Balance stand not needed - robot already stable"""
        print("⚠️  Skipping balance stand (robot already standing)")
        return True
    
    def switch_gait(self, gait_type=1):
        """Switch gait before movement - NOT AVAILABLE on G1"""
        # G1 doesn't have SwitchGait method like Go2
        # Skip this step for G1
        print(f"⚠️  Gait switching not available on G1 (skipping)")
        return True
    
    def move(self, vx, vy, vyaw):
        """
        Send movement command (matches g1_high_level_control.py implementation)
        vx: forward/backward velocity (m/s)
        vy: left/right velocity (m/s)
        vyaw: rotation velocity (rad/s)
        
        NOTE: G1 robots require CONTINUOUS Move() commands to keep moving!
        This method caches the command for continuous sending.
        """
        try:
            # Clip to safe limits (from g1_high_level_control.py)
            vx = np.clip(vx, -self.max_velocity, self.max_velocity)
            vy = np.clip(vy, -self.max_velocity, self.max_velocity)
            vyaw = np.clip(vyaw, -self.max_yaw_speed, self.max_yaw_speed)
            
            # Cache the command
            self.last_vx = vx
            self.last_vy = vy
            self.last_vyaw = vyaw
            
            # Send command
            self.sport_client.Move(vx, vy, vyaw)
            return True
        except Exception as e:
            print(f"Move failed: {e}")
            return False
    
    def send_continuous_move(self):
        """Send the last move command continuously - CRITICAL for G1 movement"""
        try:
            self.sport_client.Move(self.last_vx, self.last_vy, self.last_vyaw)
        except:
            pass  # Silently fail to avoid spam
    
    def pose(self, body_height=0.0, roll=0.0, pitch=0.0, yaw=0.0):
        """Set body pose - DISABLED (G1 API incompatible)"""
        # G1 Pose() API is different from Go2 - takes single pose object
        # Not using this for now
        return False
    
    def move_head(self, pitch, yaw):
        """
        Move robot head/upper body to track target
        pitch: up/down angle (rad), positive = look up
        yaw: left/right angle (rad), positive = look left
        """
        # Clip to safe ranges
        pitch = np.clip(pitch, -0.5, 0.5)  # ~±30 degrees
        yaw = np.clip(yaw, -0.8, 0.8)  # ~±45 degrees
        
        self.current_head_pitch = pitch
        self.current_head_yaw = yaw
        
        # Use Pose to adjust upper body orientation
        try:
            self.pose(0.0, 0.0, pitch, yaw)
        except:
            pass  # Silently fail if not supported
    
    def stop_move(self):
        """Stop all movement"""
        try:
            self.sport_client.StopMove()
        except Exception as e:
            pass  # Silently fail
    
    def wave_hand(self, duration=3.0):
        """Make robot wave/handshake gesture - DISABLED (Pose API incompatible)"""
        print(f"\n⚠️  Wave gesture disabled - Pose() API incompatible with G1")
        print("   G1 Pose() takes different parameters than Go2")
        print("   Skipping wave test...")
        return True


class G1VisionControl:
    """Integrated vision and movement control for G1"""
    
    def __init__(self, robot_controller=None):
        """
        Initialize vision control
        
        Args:
            robot_controller: Instance of SimpleRobotController or compatible
        """
        self.robot = robot_controller
        self.detector = None
        
        # Tracking parameters
        self.target_class = None
        self.frame_center = (320, 240)  # Default center
        self.deadzone = 30  # SMALLER deadzone for more precise tracking
        self.safe_distance_min = 15000  # Minimum target area -> farther than desired
        self.safe_distance_max = 40000  # Maximum target area -> closer than desired
        self.safe_distance_target = 27500  # Ideal area (sweet spot)
        self.last_known_position = None  # Remember last position
        self.frames_since_detection = 0
        self.search_pattern_step = 0
        
        # Motion detection parameters
        self.prev_frame = None
        self.motion_threshold = 500  # Minimum pixels of motion to track
        
        # Head tracking parameters
        self.use_head_tracking = True
        self.head_pitch = 0.0
        self.head_yaw = 0.0
        
        print("G1 Vision Control initialized")
    
    def setup_vision(self, model_type='cascade', camera_source=0, use_multi_detect=False, use_motion_tracking=True):
        """
        Setup vision detection
        
        Args:
            model_type: Detection model to use
            camera_source: Camera index or URL
            use_multi_detect: Use multiple detection methods simultaneously
            use_motion_tracking: Enable motion-based tracking as fallback
        """
        self.detector = G1VisionDetector(model_type=model_type, confidence_threshold=0.25)  # Lower threshold
        self.use_multi_detect = use_multi_detect
        self.use_motion_tracking = use_motion_tracking
        
        # Create additional detectors for multi-detection mode
        if use_multi_detect:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            print("✓ Multi-detection mode enabled (face + profile + eyes)")
        
        if use_motion_tracking:
            print("✓ Motion tracking enabled as fallback")
        
        if not self.detector.connect_camera(camera_source):
            print("✗ Failed to connect camera")
            return False
        
        self.detector.start_capture()
        time.sleep(1)
        
        # Get actual frame center
        with self.detector.frame_lock:
            if self.detector.frame is not None:
                h, w = self.detector.frame.shape[:2]
                self.frame_center = (w // 2, h // 2)
        
        print(f"✓ Vision system ready (frame center: {self.frame_center})")
        return True
    
    def track_object(self, target_class='person', duration=30, speed=0.2, display=True):
        """
        Track and follow a specific object class
        
        Args:
            target_class: Object class to track (e.g., 'person', 'face', 'bottle')
            duration: How long to track (seconds)
            speed: Movement speed multiplier
            display: Show display (set False for headless mode)
        """
        if self.detector is None:
            print("✗ Vision system not setup. Call setup_vision() first.")
            return
        
        self.target_class = target_class
        print(f"\n{'='*60}")
        print(f"TRACKING: {target_class}")
        print(f"Duration: {duration}s | Speed: {speed}")
        print(f"Mode: {'Display' if display else 'Headless (saving to files)'}")
        print(f"{'='*60}\n")
        
        # Keep threaded capture running - just read from shared frame
        start_time = time.time()
        last_command_time = time.time()
        no_detection_count = 0
        frame_count = 0
        save_interval = 30  # Save every 30 frames (~1 sec)
        
        try:
            while time.time() - start_time < duration:
                # Read from detector's shared frame (thread-safe)
                with self.detector.frame_lock:
                    if self.detector.frame is None:
                        time.sleep(0.01)
                        continue
                    frame = self.detector.frame.copy()
                
                # Detect objects - use multiple methods for better detection
                if self.use_multi_detect and target_class == 'face':
                    target_detections = self._multi_face_detect(frame)
                elif self.use_motion_tracking and target_class == 'motion':
                    target_detections = self._detect_motion(frame)
                else:
                    detections = self.detector.detect_objects(frame)
                    # Filter for target class
                    target_detections = [d for d in detections if d[0] == target_class]
                
                # If no detections and motion tracking enabled, try motion as fallback
                if not target_detections and self.use_motion_tracking and target_class != 'motion':
                    motion_detections = self._detect_motion(frame)
                    if motion_detections:
                        target_detections = motion_detections
                
                if target_detections:
                    no_detection_count = 0
                    self.frames_since_detection = 0
                    
                    # Get closest/largest target
                    target = max(target_detections, key=lambda x: x[2][2] * x[2][3])  # By area
                    class_name, confidence, (x, y, w, h) = target
                    
                    # Calculate target center
                    target_center = (x + w // 2, y + h // 2)
                    self.last_known_position = (target_center, w * h)  # Save position and size
                    
                    # Calculate offset from frame center
                    dx = target_center[0] - self.frame_center[0]
                    dy = target_center[1] - self.frame_center[1]
                    
                    # Control robot based on offset - CONTINUOUS TRACKING
                    self._control_from_offset(dx, dy, w * h, speed)
                    
                    # Draw visualization
                    annotated = self.detector.draw_detections(frame, [target])
                    cv2.circle(annotated, target_center, 5, (0, 0, 255), -1)
                    cv2.circle(annotated, self.frame_center, 5, (0, 255, 0), -1)
                    cv2.line(annotated, self.frame_center, target_center, (255, 0, 0), 2)
                    
                    status = f"LOCKED ON: {class_name} ({confidence:.2f})"
                    cv2.putText(annotated, status, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    if display:
                        cv2.imshow('G1 Object Tracking', annotated)
                    elif frame_count % save_interval == 0:
                        filename = f"tracking_{int(time.time())}.jpg"
                        cv2.imwrite(filename, annotated)
                    
                    frame_count += 1
                    
                else:
                    no_detection_count += 1
                    frame_count += 1
                    self.frames_since_detection += 1
                    
                    # Stop robot when no target
                    if self.robot:
                        self.robot.stop_move()
                    
                    # Show searching status (less verbose)
                    if self.frames_since_detection < 90:
                        search_msg = f"SEARCHING... frames lost: {self.frames_since_detection}"
                    else:
                        search_msg = f"LOST target: {target_class}"
                        self.last_known_position = None
                    
                    if display:
                        cv2.putText(frame, search_msg, (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.imshow('G1 Object Tracking', frame)
                
                if display and cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Small delay for ~30 fps
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\n\nTracking stopped by user")
        
        finally:
            if self.robot:
                self.robot.stop_move()
            if display:
                cv2.destroyAllWindows()
            print("\n✓ Tracking complete")
    
    def _multi_face_detect(self, frame):
        """
        Enhanced face detection using multiple cascades
        Returns detections in format: [(class_name, confidence, (x, y, w, h)), ...]
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = []
        
        # Frontal face detection - MORE SENSITIVE
        frontal_faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,  # Smaller steps for better detection
            minNeighbors=3,     # Lower threshold
            minSize=(20, 20),   # Smaller minimum size
            maxSize=(400, 400)  # Larger maximum size
        )
        for (x, y, w, h) in frontal_faces:
            detections.append(('face', 0.95, (int(x), int(y), int(w), int(h))))
        
        # Profile face detection
        profile_faces = self.profile_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05, 
            minNeighbors=3, 
            minSize=(20, 20),
            maxSize=(400, 400)
        )
        for (x, y, w, h) in profile_faces:
            detections.append(('face', 0.85, (int(x), int(y), int(w), int(h))))
        
        # Eye detection - helps with faces at angles
        if len(detections) == 0:
            eyes = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(15, 15))
            if len(eyes) >= 2:
                # Estimate face from eye positions
                eyes_sorted = sorted(eyes, key=lambda e: e[0])  # Sort by x position
                left_eye = eyes_sorted[0]
                right_eye = eyes_sorted[1]
                
                # Calculate face bounding box from eyes
                eye_distance = abs(right_eye[0] - left_eye[0])
                face_w = int(eye_distance * 2.5)
                face_h = int(face_w * 1.3)
                face_x = max(0, left_eye[0] - int(face_w * 0.2))
                face_y = max(0, left_eye[1] - int(face_h * 0.3))
                
                detections.append(('face', 0.7, (face_x, face_y, face_w, face_h)))
        
        return detections
    
    def _detect_motion(self, frame):
        """
        Detect motion in frame and return largest moving region
        Returns detections in format: [(class_name, confidence, (x, y, w, h)), ...]
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return []
        
        # Calculate frame difference
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Update previous frame
        self.prev_frame = gray
        
        detections = []
        for contour in contours:
            if cv2.contourArea(contour) < self.motion_threshold:
                continue
            
            (x, y, w, h) = cv2.boundingRect(contour)
            detections.append(('motion', 0.6, (x, y, w, h)))
        
        return detections
    
    def _control_from_offset(self, dx, dy, area, speed):
        """
        Control robot based on target detection - ACTIVE TRACKING WITH MOVEMENT
        
        Args:
            dx: Horizontal offset (pixels)
            dy: Vertical offset (pixels)
            area: Target area (for distance estimation)
            speed: Speed multiplier
        """
        if self.robot is None:
            print(f"\r⚠️  No robot controller - dx:{dx:4.0f} area:{area:6.0f}  ", end='', flush=True)
            return
        
        # Calculate movement commands based on target position
        vx = 0.0  # Forward/backward
        vy = 0.0  # Strafe left/right
        vyaw = 0.0  # Rotation
        
        # HORIZONTAL CONTROL - Strafe to keep target centered
        if abs(dx) > self.deadzone:
            # Positive dx = target is RIGHT of center, strafe RIGHT (negative vy)
            # Negative dx = target is LEFT of center, strafe LEFT (positive vy)
            vy = -np.clip(dx / 200.0, -0.3, 0.3) * speed
            print(f"\r🏃 STRAFING: vy={vy:+.3f} | dx={dx:4.0f}  ", end='', flush=True)
        
        # DISTANCE CONTROL - Move forward/backward to maintain safe distance
        if area > self.safe_distance_max:
            # Target occupies large area -> too close, back up
            vx = -0.18 * speed
            print(f"\r⬅️  TOO CLOSE! Backing up: vx={vx:+.3f} | area={area:6.0f}  ", end='', flush=True)
        elif area < self.safe_distance_min and area > 0:
            # Target small -> too far, advance
            vx = 0.25 * speed
            print(f"\r➡️  TOO FAR! Moving forward: vx={vx:+.3f} | area={area:6.0f}  ", end='', flush=True)
        else:
            # Good distance - just track sideways
            vx = 0.0
        
        # Send movement command
        if abs(vx) > 0.01 or abs(vy) > 0.01 or abs(vyaw) > 0.01:
            self.robot.move(vx, vy, vyaw)
            print(f"\r🎯 TRACKING: vx={vx:+.3f}, vy={vy:+.3f}, vyaw={vyaw:+.3f} | dx={dx:4.0f}, area={area:6.0f}  ", 
                  end='', flush=True)
        else:
            # Target is centered and at good distance - stop
            self.robot.stop_move()
            print(f"\r✓ LOCKED ON CENTER! | dx={dx:4.0f}, area={area:6.0f}  ", end='', flush=True)
    
    def count_objects(self, duration=10, display=True):
        """
        Count and classify objects in view
        
        Args:
            duration: How long to count (seconds)
            display: Show live display
        """
        if self.detector is None:
            print("✗ Vision system not setup")
            return
        
        print(f"\n{'='*60}")
        print(f"OBJECT COUNTING ({duration}s)")
        print(f"{'='*60}\n")
        
        all_detections = []
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                with self.detector.frame_lock:
                    if self.detector.frame is None:
                        time.sleep(0.1)
                        continue
                    frame = self.detector.frame.copy()
                
                detections = self.detector.detect_objects(frame)
                all_detections.extend(detections)
                
                if display:
                    annotated = self.detector.draw_detections(frame, detections)
                    summary = self.detector.get_detection_summary(detections)
                    
                    y_offset = 30
                    for class_name, info in summary['by_class'].items():
                        text = f"{class_name}: {info['count']}"
                        cv2.putText(annotated, text, (10, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        y_offset += 30
                    
                    cv2.imshow('Object Counting', annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n\nCounting stopped")
        
        # Calculate statistics
        class_counts = {}
        for class_name, conf, bbox in all_detections:
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print("\n" + "="*60)
        print("COUNTING RESULTS")
        print("="*60)
        print(f"Total detections: {len(all_detections)}")
        print(f"\nBy class:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"  {class_name}: {count}")
        
        cv2.destroyAllWindows()
        return class_counts
    
    def shutdown(self):
        """Cleanup resources"""
        if self.detector:
            self.detector.release()
        if self.robot:
            self.robot.stop_move()
        print("✓ Vision control shutdown")


def demo_tracking():
    """Demo: Track a face using enhanced detection and aggressive following"""
    print("\n" + "="*60)
    print("G1 ENHANCED FACE TRACKING DEMO")
    print("="*60)
    
    # Initialize robot controller
    robot = None
    if SDK_AVAILABLE:
        try:
            print("\nInitializing robot controller...")
            robot = SimpleRobotController(domain_id=0, network_interface="eth0")
            print("✓ Robot controller initialized")
            print("⚠️  Ensure robot is ALREADY STANDING and clear the area")

            # Check robot state - assumes already standing
            if hasattr(robot, 'stand_up'):
                print("\n🤖 Checking robot state...")
                robot.stand_up()  # Just confirms, doesn't actually stand up
            
            print("\n✓ Robot ready for tracking!")
            print("   (Robot assumed standing - no standup sequence needed)")
                
        except Exception as e:
            print(f"\n❌ Could not initialize robot controller: {e}")
            print("Vision will work but robot won't move")
            robot = None
    else:
        print("\n❌ Unitree SDK not available")
        print("Vision will work but robot won't move")
    
    # Setup vision control with robot
    vision_control = G1VisionControl(robot_controller=robot)
    
    # Try working cameras (2 or 4 from diagnostic)
    camera_source = 4
    print(f"\nTrying camera {camera_source}...")
    print("Using ENHANCED MULTI-DETECTION (face + profile + eyes + MOTION)...")
    
    # Use multi-detection with motion tracking for narrow FOV camera
    if not vision_control.setup_vision(
        model_type='cascade', 
        camera_source=camera_source, 
        use_multi_detect=True,
        use_motion_tracking=True
    ):
        print(f"Camera {camera_source} failed, trying camera 2...")
        camera_source = 2
        if not vision_control.setup_vision(
            model_type='cascade', 
            camera_source=camera_source, 
            use_multi_detect=True,
            use_motion_tracking=True
        ):
            print("✗ No working cameras found")
            print("Run: python3 find_cameras.py")
            return
    
    print("\n✓ Vision ready with ULTRA-SENSITIVE detection")
    print("\n" + "="*60)
    print("TRACKING MODE: ACTIVE MOVEMENT TRACKING")
    print("="*60)
    if robot:
        print("🤖 Robot is STANDING and READY:")
        print("   ✓ Robot will STRAFE LEFT/RIGHT to center target")
        print("   ✓ Robot will MOVE FORWARD if target too far")
        print("   ✓ Robot will BACK UP if target too close")
        print("   🎯 Multi-cascade detection (face + profile + eyes)")
        print("   🎯 Motion tracking fallback (follows ANY movement)")
        print("\n   ⚠️  ROBOT WILL MOVE - Clear space around robot!")
        print("\n🚀 Active tracking - robot WILL follow faces!")
    else:
        print("❌ Robot movement disabled - controller not initialized")
        print("   Only vision detection will work")
    print("\n TIP: Stand in front of robot - it will track and follow you!")
    print("   Robot will strafe left/right and move forward/backward")
    print("\n⚠️  MAKE SURE ROBOT HAS CLEAR SPACE TO MOVE!")
    print("\nStarting tracking in 2 seconds...")
    print("Press Ctrl+C to stop\n")
    
    # Wait a moment before starting
    time.sleep(2)
    
    print("🚀 TRACKING STARTING NOW!\n")
    
    # Track with MAXIMUM speed for narrow FOV
    vision_control.track_object(target_class='face', duration=180, speed=0.8, display=False)
    
    # Stop and sit down when done
    if robot:
        print("\n\nStopping robot...")
        robot.stop_move()
        time.sleep(1)
    
    vision_control.shutdown()


def demo_counting():
    """Demo: Count objects in view"""
    print("\n" + "="*60)
    print("G1 OBJECT COUNTING DEMO")
    print("="*60)
    
    vision_control = G1VisionControl()
    
    # Try working cameras (2 or 4 from diagnostic)
    camera_source = 4
    print(f"\nTrying camera {camera_source}...")
    if not vision_control.setup_vision(model_type='cascade', camera_source=camera_source):
        print(f"Camera {camera_source} failed, trying camera 4...")
        camera_source = 2
        if not vision_control.setup_vision(model_type='cascade', camera_source=camera_source):
            print("✗ No working cameras found")
            print("Run: python3 find_cameras.py")
            return
    
    print("\n✓ Vision ready")
    print("\nCounting objects for 10 seconds...")
    print("Running in HEADLESS mode - no display")
    print("Press Ctrl+C to stop early\n")
    
    vision_control.count_objects(duration=10, display=False)
    vision_control.shutdown()


if __name__ == "__main__":
    print("\nSelect demo:")
    print("1. Object tracking (follows target)")
    print("2. Object counting")
    
    choice = input("\nChoice [1]: ").strip() or "1"
    
    if choice == "1":
        demo_tracking()
    else:
        demo_counting()
