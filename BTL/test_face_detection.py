#!/usr/bin/env python3
"""
Test script to verify face detection and landmark tracking is working correctly.
Run this to check if MediaPipe Face Mesh and visualization functions are available.
"""

import sys
import os

# Add path
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test if all required libraries import correctly."""
    print("\n" + "="*60)
    print("TESTING IMPORTS")
    print("="*60)
    
    try:
        import cv2
        print("✓ cv2 (OpenCV) imported successfully")
    except ImportError as e:
        print(f"✗ cv2 (OpenCV) import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✓ torch imported successfully (version: {torch.__version__})")
    except ImportError as e:
        print(f"✗ torch import failed: {e}")
        return False
    
    try:
        import mediapipe
        print(f"✓ mediapipe imported successfully (version: {mediapipe.__version__})")
    except ImportError as e:
        print(f"✗ mediapipe import failed: {e}")
        print("  → Install with: pip install mediapipe")
        return False
    
    return True


def test_face_mesh():
    """Test if MediaPipe Face Mesh loads correctly."""
    print("\n" + "="*60)
    print("TESTING MEDIAPIPE FACE MESH")
    print("="*60)
    
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✓ MediaPipe Face Mesh initialized successfully")
        print(f"  → Model configured for up to 1 face")
        print(f"  → Will detect 468 3D landmarks")
        print(f"  → Static mode: False (video input)")
        return face_mesh
    except Exception as e:
        print(f"✗ MediaPipe Face Mesh initialization failed: {e}")
        return None


def test_landmark_functions():
    """Test if landmark visualization functions are defined."""
    print("\n" + "="*60)
    print("TESTING LANDMARK VISUALIZATION FUNCTIONS")
    print("="*60)
    
    functions_to_check = [
        "get_landmark_region",
        "draw_face_landmarks",
        "draw_face_bounding_box",
        "get_key_landmarks",
        "draw_key_landmarks_text",
        "draw_landmark_statistics",
    ]
    
    # Try to import functions from web_cam
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("web_cam", "web_cam.py")
        web_cam = importlib.util.module_from_spec(spec)
        
        # We can't fully execute due to environment setup, but check if functions exist
        with open("web_cam.py", "r") as f:
            content = f.read()
        
        all_found = True
        for func_name in functions_to_check:
            if f"def {func_name}(" in content:
                print(f"✓ {func_name}() is defined")
            else:
                print(f"✗ {func_name}() is NOT defined")
                all_found = False
        
        return all_found
    
    except Exception as e:
        print(f"✗ Error checking functions: {e}")
        return False


def test_color_scheme():
    """Display the color scheme for landmarks."""
    print("\n" + "="*60)
    print("LANDMARK COLOR SCHEME (BGR)")
    print("="*60)
    
    colors = {
        "Lips": (0, 0, 255),
        "Eyes": (255, 0, 0),
        "Eyebrows": (0, 165, 255),
        "Nose": (0, 255, 0),
        "Face Contour": (255, 255, 0),
        "Left Cheek": (255, 0, 255),
        "Right Cheek": (128, 0, 128),
    }
    
    for region, (b, g, r) in colors.items():
        print(f"  ● {region:15s} → BGR({b:3d}, {g:3d}, {r:3d})")
    
    return True


def test_key_landmarks():
    """Display the key landmarks that will be extracted."""
    print("\n" + "="*60)
    print("KEY LANDMARKS (7 POINTS)")
    print("="*60)
    
    landmarks = {
        "nose_tip": 4,
        "left_eye": 33,
        "right_eye": 263,
        "left_mouth": 61,
        "right_mouth": 291,
        "chin": 152,
        "forehead": 10,
    }
    
    for name, idx in landmarks.items():
        print(f"  • {name:15s} (index {idx:3d})")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "█"*60)
    print("  FACE DETECTION & LANDMARK TRACKING - VERIFICATION TEST")
    print("█"*60)
    
    results = {
        "Imports": test_imports(),
    }
    
    if results["Imports"]:
        results["MediaPipe Face Mesh"] = test_face_mesh() is not None
        results["Landmark Functions"] = test_landmark_functions()
        test_color_scheme()
        test_key_landmarks()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {test_name}")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    
    if all(results.values()):
        print("✓ All tests passed!")
        print("\nRun the webcam with landmarks:")
        print("  python web_cam.py --model advanced")
        print("\nYou should see:")
        print("  • Green bounding box around detected face")
        print("  • 468 colored landmark points")
        print("  • Face mesh connections between landmarks")
        print("  • Key landmark coordinates (top-right)")
        print("  • Detection statistics (bottom-left)")
        print("  • Head orientation angles (yaw/pitch/roll)")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("  1. pip install mediapipe")
        print("  2. pip install opencv-python")
        print("  3. pip install torch torchvision")
    
    print("\n" + "█"*60 + "\n")


if __name__ == "__main__":
    main()
