#!/usr/bin/env python3
"""
Find all available cameras on the G1 robot
"""

import cv2
import subprocess
import os


def check_video_devices():
    """Check for /dev/video* devices"""
    print("=" * 60)
    print("1. Checking Video Devices")
    print("=" * 60)
    
    devices = []
    for i in range(10):
        device = f"/dev/video{i}"
        if os.path.exists(device):
            devices.append(device)
            print(f"  ✓ Found: {device}")
    
    if not devices:
        print("  ✗ No /dev/video* devices found")
        print("  → Robot may not have USB cameras attached")
    
    print()
    return devices


def test_opencv_cameras():
    """Test OpenCV camera indices"""
    print("=" * 60)
    print("2. Testing OpenCV Camera Indices")
    print("=" * 60)
    
    working_cameras = []
    
    for i in range(10):
        print(f"  Testing camera index {i}...", end=" ")
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"✓ Working ({w}x{h})")
                working_cameras.append(i)
            else:
                print("✗ Opens but no frame")
        else:
            print("✗ Cannot open")
        
        cap.release()
    
    print()
    if not working_cameras:
        print("  ✗ No working OpenCV cameras found")
    else:
        print(f"  ✓ Found {len(working_cameras)} working camera(s): {working_cameras}")
    
    print()
    return working_cameras


def check_rtsp_streams():
    """Check common G1 RTSP camera streams"""
    print("=" * 60)
    print("3. Checking G1 Robot RTSP Streams")
    print("=" * 60)
    
    # Common G1 camera IPs and ports
    streams = [
        "rtsp://192.168.123.161:8554/main_stream",
        "rtsp://192.168.123.161:8554/sub_stream",
        "rtsp://192.168.123.164:8554/main_stream",
        "rtsp://192.168.123.164:8554/sub_stream",
        "rtsp://192.168.123.15:8554/main_stream",   # Go2 default
    ]
    
    working_streams = []
    
    for stream in streams:
        print(f"  Testing: {stream}")
        print(f"    Trying connection...", end=" ")
        
        cap = cv2.VideoCapture(stream, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)  # 3 second timeout
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"✓ Working ({w}x{h})")
                working_streams.append(stream)
            else:
                print("✗ Opens but no frame")
        else:
            print("✗ Connection refused")
        
        cap.release()
    
    print()
    if not working_streams:
        print("  ✗ No RTSP streams accessible")
        print("  → Check robot IP and camera service")
    else:
        print(f"  ✓ Found {len(working_streams)} working stream(s):")
        for s in working_streams:
            print(f"    - {s}")
    
    print()
    return working_streams


def check_http_streams():
    """Check common HTTP/MJPEG streams"""
    print("=" * 60)
    print("4. Checking HTTP/MJPEG Streams")
    print("=" * 60)
    
    streams = [
        "http://192.168.123.161:8080/video",
        "http://192.168.123.164:8080/video",
        "http://192.168.123.161:8000/stream.mjpg",
    ]
    
    working_streams = []
    
    for stream in streams:
        print(f"  Testing: {stream}...", end=" ")
        
        cap = cv2.VideoCapture(stream)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"✓ Working ({w}x{h})")
                working_streams.append(stream)
            else:
                print("✗ Opens but no frame")
        else:
            print("✗ Cannot open")
        
        cap.release()
    
    print()
    if working_streams:
        print(f"  ✓ Found {len(working_streams)} working stream(s):")
        for s in working_streams:
            print(f"    - {s}")
    else:
        print("  ✗ No HTTP streams accessible")
    
    print()
    return working_streams


def check_network():
    """Check network connectivity to robot"""
    print("=" * 60)
    print("5. Checking Network to Robot")
    print("=" * 60)
    
    ips = ["192.168.123.161", "192.168.123.164", "192.168.123.15"]
    
    reachable = []
    for ip in ips:
        print(f"  Pinging {ip}...", end=" ")
        result = subprocess.run(['ping', '-c', '1', '-W', '1', ip],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Reachable")
            reachable.append(ip)
        else:
            print("✗ Not reachable")
    
    print()
    if not reachable:
        print("  ✗ Robot not reachable on network")
        print("  → Check eth0 configuration and robot power")
    else:
        print(f"  ✓ Robot reachable at: {', '.join(reachable)}")
    
    print()
    return reachable


def main():
    print("\n" + "=" * 60)
    print("UNITREE G1 CAMERA DISCOVERY")
    print("=" * 60)
    print()
    
    # Run all checks
    video_devices = check_video_devices()
    opencv_cameras = test_opencv_cameras()
    rtsp_streams = check_rtsp_streams()
    http_streams = check_http_streams()
    reachable_ips = check_network()
    
    # Summary
    print("=" * 60)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 60)
    print()
    
    if opencv_cameras:
        print("✓ Use OpenCV camera:")
        for cam in opencv_cameras:
            print(f"  detector.connect_camera({cam})")
        print()
    
    if rtsp_streams:
        print("✓ Use RTSP stream:")
        for stream in rtsp_streams:
            print(f'  detector.connect_camera("{stream}")')
        print()
    
    if http_streams:
        print("✓ Use HTTP stream:")
        for stream in http_streams:
            print(f'  detector.connect_camera("{stream}")')
        print()
    
    if not opencv_cameras and not rtsp_streams and not http_streams:
        print("✗ NO CAMERAS FOUND!")
        print()
        print("Troubleshooting:")
        print("  1. Check if robot camera service is running:")
        print("     systemctl status unitree-camera  # or similar")
        print()
        print("  2. Check robot IP and network:")
        if reachable_ips:
            print(f"     Robot reachable at: {reachable_ips[0]}")
        else:
            print("     sudo ip addr add 192.168.123.2/24 dev eth0")
            print("     ping 192.168.123.161")
        print()
        print("  3. For USB camera, plug in and check:")
        print("     ls -l /dev/video*")
        print()
        print("  4. Try robot SDK camera examples:")
        print("     cd ~/unitree_sdk2_python/example")
        print("     python3 <camera_example>.py")
        print()
    
    print("=" * 60)


if __name__ == "__main__":
    main()
