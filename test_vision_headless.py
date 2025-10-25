#!/usr/bin/env python3
"""
Unitree G1 Vision Detection - Headless Mode
Saves snapshots to files instead of displaying windows
"""

import cv2
import numpy as np
import time
from g1_vision_detection import G1VisionDetector


def test_vision_headless():
    """Test vision detection without display (headless mode)"""
    print("\n" + "="*60)
    print("G1 VISION DETECTION - HEADLESS MODE")
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
    
    cam_choice = input("\nChoice [1]: ").strip() or "1"
    
    if cam_choice == "1":
        camera_source = 2
    elif cam_choice == "2":
        camera_source = 4
    elif cam_choice == "3":
        camera_source = int(input("Enter camera index: ").strip())
    else:
        camera_source = 2
    
    # Connect to camera
    if not detector.connect_camera(camera_source):
        print("\n✗ Failed to connect to camera")
        print("\nTrying fallback cameras...")
        for fallback in [2, 4]:
            print(f"Trying camera {fallback}...")
            if detector.connect_camera(fallback):
                break
        else:
            print("✗ No camera available")
            return
    
    # Start capture
    detector.start_capture()
    time.sleep(1)
    
    print("\n✓ Detection system ready (headless mode)")
    print("\nOptions:")
    print("  1. Capture single frame")
    print("  2. Capture 10 frames (1 per second)")
    print("  3. Continuous capture (Ctrl+C to stop)")
    
    mode = input("\nChoice [1]: ").strip() or "1"
    
    print("\nStarting capture...")
    print("Frames will be saved to: detection_<timestamp>.jpg")
    print("="*60 + "\n")
    
    frame_count = 0
    
    try:
        if mode == "1":
            # Single frame
            time.sleep(0.5)
            with detector.frame_lock:
                if detector.frame is not None:
                    frame = detector.frame.copy()
                    
                    detections = detector.detect_objects(frame)
                    annotated = detector.draw_detections(frame, detections)
                    summary = detector.get_detection_summary(detections)
                    
                    filename = f"detection_{int(time.time())}.jpg"
                    cv2.imwrite(filename, annotated)
                    
                    print(f"✓ Saved: {filename}")
                    print(f"  Detected: {summary['total_objects']} objects")
                    for class_name, info in summary['by_class'].items():
                        print(f"    - {class_name}: {info['count']} ({info['avg_confidence']:.2f})")
        
        elif mode == "2":
            # 10 frames
            for i in range(10):
                with detector.frame_lock:
                    if detector.frame is not None:
                        frame = detector.frame.copy()
                        
                        detections = detector.detect_objects(frame)
                        annotated = detector.draw_detections(frame, detections)
                        summary = detector.get_detection_summary(detections)
                        
                        filename = f"detection_{int(time.time())}_{i:02d}.jpg"
                        cv2.imwrite(filename, annotated)
                        
                        print(f"✓ Frame {i+1}/10: {filename}")
                        if summary['total_objects'] > 0:
                            print(f"  Detected: {summary['total_objects']} objects")
                            for class_name, info in summary['by_class'].items():
                                print(f"    - {class_name}: {info['count']}")
                
                time.sleep(1)
        
        else:
            # Continuous
            print("Capturing continuously (Ctrl+C to stop)...")
            print("Saving one frame every 2 seconds\n")
            
            while True:
                with detector.frame_lock:
                    if detector.frame is not None:
                        frame = detector.frame.copy()
                        
                        detections = detector.detect_objects(frame)
                        annotated = detector.draw_detections(frame, detections)
                        summary = detector.get_detection_summary(detections)
                        
                        filename = f"detection_{int(time.time())}.jpg"
                        cv2.imwrite(filename, annotated)
                        frame_count += 1
                        
                        status = f"Frame {frame_count}: {filename} | Objects: {summary['total_objects']}"
                        if summary['by_class']:
                            status += " | " + ", ".join([f"{k}:{v['count']}" for k, v in summary['by_class'].items()])
                        print(status)
                
                time.sleep(2)
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    finally:
        detector.release()
        print(f"\n✓ Saved {frame_count} frames")
        print("✓ Test complete")


if __name__ == "__main__":
    test_vision_headless()
