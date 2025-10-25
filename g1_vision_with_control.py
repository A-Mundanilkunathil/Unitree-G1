"""
Unitree G1 Vision + Movement Control
Combines vision detection with robot movement for object tracking and following
"""

import cv2
import numpy as np
import time
from g1_vision_detection import G1VisionDetector


class G1VisionControl:
    """Integrated vision and movement control for G1"""
    
    def __init__(self, robot_controller=None):
        """
        Initialize vision control
        
        Args:
            robot_controller: Instance of G1HighLevelController or compatible
        """
        self.robot = robot_controller
        self.detector = None
        
        # Tracking parameters
        self.target_class = None
        self.frame_center = (320, 240)  # Default center
        self.deadzone = 50  # Pixels deadzone around center
        
        print("G1 Vision Control initialized")
    
    def setup_vision(self, model_type='cascade', camera_source=0):
        """
        Setup vision detection
        
        Args:
            model_type: Detection model to use
            camera_source: Camera index or URL
        """
        self.detector = G1VisionDetector(model_type=model_type, confidence_threshold=0.5)
        
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
    
    def track_object(self, target_class='person', duration=30, speed=0.2):
        """
        Track and follow a specific object class
        
        Args:
            target_class: Object class to track (e.g., 'person', 'face', 'bottle')
            duration: How long to track (seconds)
            speed: Movement speed multiplier
        """
        if self.detector is None:
            print("✗ Vision system not setup. Call setup_vision() first.")
            return
        
        self.target_class = target_class
        print(f"\n{'='*60}")
        print(f"TRACKING: {target_class}")
        print(f"Duration: {duration}s | Speed: {speed}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        no_detection_count = 0
        
        try:
            while time.time() - start_time < duration:
                # Get current frame
                with self.detector.frame_lock:
                    if self.detector.frame is None:
                        time.sleep(0.1)
                        continue
                    frame = self.detector.frame.copy()
                
                # Detect objects
                detections = self.detector.detect_objects(frame)
                
                # Filter for target class
                target_detections = [d for d in detections if d[0] == target_class]
                
                if target_detections:
                    no_detection_count = 0
                    
                    # Get closest/largest target
                    target = max(target_detections, key=lambda x: x[2][2] * x[2][3])  # By area
                    class_name, confidence, (x, y, w, h) = target
                    
                    # Calculate target center
                    target_center = (x + w // 2, y + h // 2)
                    
                    # Calculate offset from frame center
                    dx = target_center[0] - self.frame_center[0]
                    dy = target_center[1] - self.frame_center[1]
                    
                    # Control robot based on offset
                    self._control_from_offset(dx, dy, w * h, speed)
                    
                    # Draw visualization
                    annotated = self.detector.draw_detections(frame, [target])
                    cv2.circle(annotated, target_center, 5, (0, 0, 255), -1)
                    cv2.circle(annotated, self.frame_center, 5, (0, 255, 0), -1)
                    
                    status = f"TRACKING: {class_name} ({confidence:.2f})"
                    cv2.putText(annotated, status, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow('G1 Object Tracking', annotated)
                    
                else:
                    no_detection_count += 1
                    
                    # Stop if lost target for too long
                    if no_detection_count > 30:  # ~1 second at 30fps
                        if self.robot:
                            self.robot.stop_move()
                        print(f"\rLost target: {target_class}          ", end='', flush=True)
                    
                    cv2.putText(frame, f"SEARCHING: {target_class}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow('G1 Object Tracking', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                time.sleep(0.03)  # ~30 fps
        
        except KeyboardInterrupt:
            print("\n\nTracking stopped by user")
        
        finally:
            if self.robot:
                self.robot.stop_move()
            print("\n✓ Tracking complete")
    
    def _control_from_offset(self, dx, dy, area, speed):
        """
        Control robot movement based on target offset
        
        Args:
            dx: Horizontal offset (pixels)
            dy: Vertical offset (pixels)
            area: Target area (for distance estimation)
            speed: Speed multiplier
        """
        if self.robot is None:
            return
        
        # Calculate velocities
        vx = 0.0  # Forward/backward
        vy = 0.0  # Strafe (usually not used)
        vyaw = 0.0  # Rotation
        
        # Horizontal control (rotation)
        if abs(dx) > self.deadzone:
            # Turn towards target
            vyaw = -np.clip(dx / 200.0, -1.0, 1.0) * speed * 0.5
        
        # Distance control (forward/backward)
        # Larger area = closer, smaller area = farther
        target_area = 20000  # Adjust based on your needs
        area_diff = target_area - area
        
        if abs(area_diff) > 5000:
            # Move forward if too far, backward if too close
            vx = np.clip(area_diff / 30000.0, -0.5, 0.5) * speed
        
        # Send movement command
        self.robot.move(vx, vy, vyaw)
        
        # Debug output
        print(f"\rTracking - dx:{dx:4.0f} area:{area:6.0f} | vx:{vx:+.2f} vyaw:{vyaw:+.2f}  ", 
              end='', flush=True)
    
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
    """Demo: Track a person using vision"""
    print("\n" + "="*60)
    print("G1 VISION TRACKING DEMO")
    print("="*60)
    
    # Setup vision control
    vision_control = G1VisionControl()
    
    # Setup vision (cascade is easiest, no model files needed)
    if not vision_control.setup_vision(model_type='cascade', camera_source=0):
        return
    
    print("\n✓ Vision ready")
    print("\nStarting face tracking...")
    print("The robot would move to center the face in frame")
    print("(Robot movement disabled in this demo)")
    print("\nPress 'q' in the window to stop\n")
    
    # Track faces for 30 seconds
    vision_control.track_object(target_class='face', duration=30, speed=0.3)
    
    vision_control.shutdown()


def demo_counting():
    """Demo: Count objects in view"""
    print("\n" + "="*60)
    print("G1 OBJECT COUNTING DEMO")
    print("="*60)
    
    vision_control = G1VisionControl()
    
    if not vision_control.setup_vision(model_type='cascade', camera_source=0):
        return
    
    print("\n✓ Vision ready")
    print("\nCounting objects for 10 seconds...")
    print("Press 'q' to stop early\n")
    
    vision_control.count_objects(duration=10, display=True)
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
