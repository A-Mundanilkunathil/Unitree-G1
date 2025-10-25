"""
Unitree G1 Vision Object Detection and Classification
Supports multiple detection models and camera feeds
"""

import cv2
import numpy as np
import time
from threading import Thread, Lock
from collections import deque


class G1VisionDetector:
    """Vision-based object detection and classification for Unitree G1"""
    
    def __init__(self, model_type='yolo', confidence_threshold=0.5):
        """
        Initialize vision detector
        
        Args:
            model_type: 'yolo', 'mobilenet', or 'cascade' (face detection)
            confidence_threshold: Minimum confidence for detections (0.0-1.0)
        """
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        
        # Camera settings
        self.frame = None
        self.frame_lock = Lock()
        self.running = False
        self.camera_thread = None
        
        # Detection results
        self.detections = []
        self.detection_lock = Lock()
        
        # Initialize model
        self.net = None
        self.classes = []
        self.colors = []
        
        print(f"Initializing {model_type} vision detector...")
        self._load_model()
        
    def _load_model(self):
        """Load detection model"""
        try:
            if self.model_type == 'yolo':
                self._load_yolo()
            elif self.model_type == 'mobilenet':
                self._load_mobilenet()
            elif self.model_type == 'cascade':
                self._load_cascade()
            else:
                print(f"Unknown model type: {self.model_type}, using cascade")
                self._load_cascade()
                
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to cascade classifier...")
            self._load_cascade()
    
    def _load_yolo(self):
        """Load YOLO model for object detection"""
        import os
        import urllib.request
        
        try:
            # YOLO v4-tiny weights and config
            model_dir = 'models'
            weights_path = os.path.join(model_dir, 'yolov4-tiny.weights')
            config_path = os.path.join(model_dir, 'yolov4-tiny.cfg')
            names_path = os.path.join(model_dir, 'coco.names')
            
            # Create models directory if it doesn't exist
            os.makedirs(model_dir, exist_ok=True)
            
            # Download URLs
            weights_url = 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights'
            config_url = 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg'
            names_url = 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names'
            
            # Download files if they don't exist
            if not os.path.exists(weights_path):
                print("Downloading YOLO weights (~23 MB)...")
                try:
                    urllib.request.urlretrieve(weights_url, weights_path)
                    print("✓ Weights downloaded")
                except Exception as e:
                    print(f"✗ Failed to download weights: {e}")
                    raise
            
            if not os.path.exists(config_path):
                print("Downloading YOLO config...")
                try:
                    urllib.request.urlretrieve(config_url, config_path)
                    print("✓ Config downloaded")
                except Exception as e:
                    print(f"✗ Failed to download config: {e}")
                    raise
            
            if not os.path.exists(names_path):
                print("Downloading class names...")
                try:
                    urllib.request.urlretrieve(names_url, names_path)
                    print("✓ Class names downloaded")
                except Exception as e:
                    print(f"✗ Failed to download class names: {e}")
                    raise
            
            # Load model
            try:
                print("Loading YOLO model...")
                self.net = cv2.dnn.readNet(weights_path, config_path)
                with open(names_path, 'r') as f:
                    self.classes = [line.strip() for line in f.readlines()]
                print(f"✓ YOLO model loaded with {len(self.classes)} classes")
            except Exception as e:
                print(f"⚠️  Failed to load YOLO: {e}")
                print("Using simplified detection")
                self.net = None
                self.classes = ['person', 'chair', 'bottle', 'cup', 'laptop']
                
            # Generate colors for each class
            self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
            
        except Exception as e:
            print(f"YOLO loading error: {e}")
            print("Falling back to cascade classifier")
            self._load_cascade()
    
    def _load_mobilenet(self):
        """Load MobileNet SSD for object detection"""
        import os
        import urllib.request
        
        try:
            # MobileNet SSD paths
            model_dir = 'models'
            prototxt = os.path.join(model_dir, 'MobileNetSSD_deploy.prototxt')
            model = os.path.join(model_dir, 'MobileNetSSD_deploy.caffemodel')
            
            # Create models directory if it doesn't exist
            os.makedirs(model_dir, exist_ok=True)
            
            # Download URLs
            prototxt_url = 'https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt'
            model_url = 'https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc'
            # Alternative model URL
            model_url_alt = 'https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel'
            
            # Download prototxt if it doesn't exist
            if not os.path.exists(prototxt):
                print("Downloading MobileNet prototxt...")
                try:
                    urllib.request.urlretrieve(prototxt_url, prototxt)
                    print("✓ Prototxt downloaded")
                except Exception as e:
                    print(f"✗ Failed to download prototxt: {e}")
                    raise
            
            # Download model if it doesn't exist
            if not os.path.exists(model):
                print("Downloading MobileNet model (~23 MB)...")
                print("⚠️  Note: MobileNet download may be slow or fail.")
                print("   If it fails, download manually from:")
                print("   https://github.com/chuanqi305/MobileNet-SSD")
                try:
                    # Try alternative URL
                    urllib.request.urlretrieve(model_url_alt, model)
                    print("✓ Model downloaded")
                except Exception as e:
                    print(f"✗ Failed to download model: {e}")
                    print("Falling back to cascade")
                    self._load_cascade()
                    return
            
            # Load model
            try:
                print("Loading MobileNet SSD...")
                self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
                self.classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                               "sofa", "train", "tvmonitor"]
                print(f"✓ MobileNet SSD loaded with {len(self.classes)} classes")
            except Exception as e:
                print(f"⚠️  Failed to load MobileNet: {e}")
                print("Using simplified detection")
                self.net = None
                self.classes = ['person', 'car', 'bottle']
                
            self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
            
        except Exception as e:
            print(f"MobileNet loading error: {e}")
            print("Falling back to cascade classifier")
            self._load_cascade()
    
    def _load_cascade(self):
        """Load Haar Cascade for face/object detection"""
        try:
            # Load face cascade (built into OpenCV)
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            self.classes = ['face', 'eye']
            self.colors = [(255, 0, 0), (0, 255, 0)]
            print("✓ Cascade classifier loaded")
        except Exception as e:
            print(f"Cascade loading error: {e}")
    
    def connect_camera(self, camera_index=0):
        """
        Connect to camera
        
        Args:
            camera_index: Camera index (0 for default, or IP camera URL)
        """
        try:
            # For G1 robot cameras, use appropriate stream
            # Common G1 camera URLs:
            # "rtsp://192.168.123.161:8554/main_stream"
            # "http://192.168.123.161:8080/video"
            
            if isinstance(camera_index, str):
                print(f"Connecting to camera stream: {camera_index}")
                self.cap = cv2.VideoCapture(camera_index)
            else:
                print(f"Connecting to camera index: {camera_index}")
                self.cap = cv2.VideoCapture(camera_index)
            
            if not self.cap.isOpened():
                print("✗ Failed to open camera")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            print("✓ Camera connected")
            return True
            
        except Exception as e:
            print(f"✗ Camera connection error: {e}")
            return False
    
    def start_capture(self):
        """Start camera capture thread"""
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            print("✗ Camera not connected. Call connect_camera() first.")
            return False
        
        self.running = True
        self.camera_thread = Thread(target=self._capture_loop, daemon=True)
        self.camera_thread.start()
        print("✓ Camera capture started")
        return True
    
    def _capture_loop(self):
        """Continuous camera capture loop"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.frame = frame.copy()
            time.sleep(0.03)  # ~30 FPS
    
    def detect_objects(self, frame=None):
        """
        Detect objects in frame
        
        Args:
            frame: Image frame (if None, uses latest camera frame)
            
        Returns:
            List of detections: [(class_name, confidence, (x, y, w, h)), ...]
        """
        if frame is None:
            with self.frame_lock:
                if self.frame is None:
                    return []
                frame = self.frame.copy()
        
        if self.model_type == 'yolo':
            return self._detect_yolo(frame)
        elif self.model_type == 'mobilenet':
            return self._detect_mobilenet(frame)
        elif self.model_type == 'cascade':
            return self._detect_cascade(frame)
        
        return []
    
    def _detect_yolo(self, frame):
        """YOLO detection"""
        detections = []
        
        if self.net is None:
            return detections
        
        try:
            height, width = frame.shape[:2]
            
            # Create blob from image
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            
            # Get output layer names
            layer_names = self.net.getLayerNames()
            output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            
            # Forward pass
            outputs = self.net.forward(output_layers)
            
            # Process detections
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > self.confidence_threshold:
                        # Get bounding box coordinates
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        class_name = self.classes[class_id] if class_id < len(self.classes) else f"class_{class_id}"
                        detections.append((class_name, float(confidence), (x, y, w, h)))
            
        except Exception as e:
            print(f"YOLO detection error: {e}")
        
        return detections
    
    def _detect_mobilenet(self, frame):
        """MobileNet SSD detection"""
        detections = []
        
        if self.net is None:
            return detections
        
        try:
            height, width = frame.shape[:2]
            
            # Create blob
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            self.net.setInput(blob)
            
            # Detection
            outputs = self.net.forward()
            
            for i in range(outputs.shape[2]):
                confidence = outputs[0, 0, i, 2]
                
                if confidence > self.confidence_threshold:
                    class_id = int(outputs[0, 0, i, 1])
                    
                    # Get bounding box
                    x = int(outputs[0, 0, i, 3] * width)
                    y = int(outputs[0, 0, i, 4] * height)
                    w = int(outputs[0, 0, i, 5] * width) - x
                    h = int(outputs[0, 0, i, 6] * height) - y
                    
                    class_name = self.classes[class_id] if class_id < len(self.classes) else f"class_{class_id}"
                    detections.append((class_name, float(confidence), (x, y, w, h)))
        
        except Exception as e:
            print(f"MobileNet detection error: {e}")
        
        return detections
    
    def _detect_cascade(self, frame):
        """Cascade classifier detection (faces, eyes)"""
        detections = []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                detections.append(('face', 1.0, (x, y, w, h)))
                
                # Detect eyes in face region
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    detections.append(('eye', 1.0, (x+ex, y+ey, ew, eh)))
        
        except Exception as e:
            print(f"Cascade detection error: {e}")
        
        return detections
    
    def draw_detections(self, frame, detections):
        """
        Draw detection boxes and labels on frame
        
        Args:
            frame: Image frame
            detections: List of detections from detect_objects()
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for class_name, confidence, (x, y, w, h) in detections:
            # Get color for this class
            color_idx = list(self.classes).index(class_name) if class_name in self.classes else 0
            color = tuple(map(int, self.colors[color_idx % len(self.colors)]))
            
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
            cv2.putText(annotated, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def get_detection_summary(self, detections):
        """
        Get summary of detections
        
        Returns:
            Dictionary with counts and details
        """
        summary = {
            'total_objects': len(detections),
            'by_class': {},
            'highest_confidence': None,
            'detections': detections
        }
        
        for class_name, confidence, bbox in detections:
            if class_name not in summary['by_class']:
                summary['by_class'][class_name] = {'count': 0, 'avg_confidence': 0.0, 'confidences': []}
            
            summary['by_class'][class_name]['count'] += 1
            summary['by_class'][class_name]['confidences'].append(confidence)
        
        # Calculate average confidences
        for class_name in summary['by_class']:
            confidences = summary['by_class'][class_name]['confidences']
            summary['by_class'][class_name]['avg_confidence'] = sum(confidences) / len(confidences)
        
        # Find highest confidence detection
        if detections:
            summary['highest_confidence'] = max(detections, key=lambda x: x[1])
        
        return summary
    
    def stop_capture(self):
        """Stop camera capture"""
        self.running = False
        if self.camera_thread:
            self.camera_thread.join(timeout=2)
        print("✓ Camera capture stopped")
    
    def release(self):
        """Release all resources"""
        self.stop_capture()
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        print("✓ Resources released")


def test_vision_detection():
    """Test vision detection system"""
    print("\n" + "="*60)
    print("G1 VISION DETECTION TEST")
    print("="*60)
    
    print("\nSelect detection model:")
    print("1. Cascade (face/eye detection) - No external files needed")
    print("2. YOLO (object detection) - Requires model files")
    print("3. MobileNet SSD (object detection) - Requires model files")
    
    choice = input("\nChoice [1]: ").strip() or "1"
    
    model_map = {'1': 'cascade', '2': 'yolo', '3': 'mobilenet'}
    model_type = model_map.get(choice, 'cascade')
    
    # Create detector
    detector = G1VisionDetector(model_type=model_type, confidence_threshold=0.5)
    
    print("\nSelect camera source:")
    print("1. G1 Camera 1 (index 2) - Recommended")
    print("2. G1 Camera 2 (index 4)")
    print("3. Custom camera index")
    print("4. Custom camera URL")
    
    cam_choice = input("\nChoice [1]: ").strip() or "1"
    
    if cam_choice == "1":
        camera_source = 2  # Working camera from diagnostic
    elif cam_choice == "2":
        camera_source = 4  # Alternative working camera
    elif cam_choice == "3":
        camera_source = int(input("Enter camera index: ").strip())
    elif cam_choice == "4":
        camera_source = input("Enter camera URL: ").strip()
    else:
        camera_source = 2
    
    # Connect to camera
    if not detector.connect_camera(camera_source):
        print("\n✗ Failed to connect to camera")
        print("\nTrying fallback cameras...")
        # Try known working cameras
        for fallback in [2, 4]:
            print(f"Trying camera {fallback}...")
            if detector.connect_camera(fallback):
                break
        else:
            print("✗ No camera available")
            print("\nRun: python3 find_cameras.py")
            return
    
    # Start capture
    detector.start_capture()
    time.sleep(1)  # Wait for camera to initialize
    
    print("\n✓ Detection system ready")
    print("\nPress 'q' to quit, 's' to save snapshot")
    print("="*60 + "\n")
    
    try:
        while True:
            with detector.frame_lock:
                if detector.frame is None:
                    time.sleep(0.1)
                    continue
                frame = detector.frame.copy()
            
            # Detect objects
            detections = detector.detect_objects(frame)
            
            # Draw detections
            annotated = detector.draw_detections(frame, detections)
            
            # Get summary
            summary = detector.get_detection_summary(detections)
            
            # Add summary text to frame
            y_offset = 30
            cv2.putText(annotated, f"Objects: {summary['total_objects']}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            for class_name, info in summary['by_class'].items():
                y_offset += 30
                text = f"{class_name}: {info['count']} ({info['avg_confidence']:.2f})"
                cv2.putText(annotated, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('G1 Vision Detection', annotated)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"detection_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated)
                print(f"✓ Saved: {filename}")
            
            # Print detections to console
            if detections:
                print(f"\rDetected: {summary['total_objects']} objects | ", end='')
                for class_name, info in summary['by_class'].items():
                    print(f"{class_name}:{info['count']} ", end='')
                print("          ", end='', flush=True)
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    finally:
        detector.release()
        print("\n✓ Test complete")


if __name__ == "__main__":
    test_vision_detection()
