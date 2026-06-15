# Face Detection & Landmark Tracking - Example Reference

**Quick Reference for Common Tasks**

---

## 🎯 Common Use Cases

### 1. Display All Landmarks with Visualization

```python
# In realtime_webcam loop, after face detection:

if last_landmarks is not None:
    # Draw all 468 landmarks with connections
    frame = draw_face_landmarks(frame, last_landmarks, 
                                draw_connections=True,
                                landmark_size=2)
    
    # Draw bounding box
    frame = draw_face_bounding_box(frame, last_landmarks,
                                   bbox_color=(0, 255, 0),
                                   thickness=2)
```

**Output**: Sees 468 colored dots connected by lines, organized by region

---

### 2. Extract Key Points Only

```python
# Get 7 most important landmarks

key_lms = get_key_landmarks(last_landmarks)

# Access specific key points
nose_x = key_lms["nose_tip"]["x"]
nose_y = key_lms["nose_tip"]["y"]
nose_z = key_lms["nose_tip"]["z"]
nose_visibility = key_lms["nose_tip"]["visibility"]

print(f"Nose at: ({nose_x:.3f}, {nose_y:.3f}, {nose_z:.3f})")
print(f"Confidence: {nose_visibility:.2f}")
```

**Output**: 
```
Nose at: (0.500, 0.300, -0.100)
Confidence: 0.95
```

---

### 3. Get Head Orientation (3D Pose)

```python
# After face detection

if last_orientation is not None:
    yaw = last_orientation["yaw"]      # Left/right rotation
    pitch = last_orientation["pitch"]  # Up/down tilt
    roll = last_orientation["roll"]    # Sideways tilt
    
    print(f"Head Pose: Yaw={yaw:.1f}°, Pitch={pitch:.1f}°, Roll={roll:.1f}°")
    
    # Interpret the values
    if abs(yaw) > 30:
        print("Head turned significantly")
    if pitch > 20:
        print("Head tilted down")
    if roll > 15:
        print("Head tilted to the side")
```

**Output**:
```
Head Pose: Yaw=15.2°, Pitch=-10.5°, Roll=3.2°
Head turned significantly
```

---

### 4. Process All Landmarks by Region

```python
# Analyze landmarks by facial region

regions_data = {
    "lips": [],
    "eyes": [],
    "nose": [],
    "cheeks": [],
}

for idx, landmark in enumerate(last_landmarks.landmark):
    if landmark.visibility < 0.5:
        continue
    
    region, color = get_landmark_region(idx)
    
    # Store landmark coordinates
    point = {
        "index": idx,
        "x": landmark.x,
        "y": landmark.y,
        "z": landmark.z,
        "visibility": landmark.visibility,
        "color": color,
    }
    
    if region == "lips":
        regions_data["lips"].append(point)
    elif region == "eyes":
        regions_data["eyes"].append(point)
    # ... etc for other regions
```

---

### 5. Extract ROI from Specific Region

```python
# Get forehead region only for rPPG analysis

forehead_indices = [10, 67, 103, 109, 338]
h, w, _ = frame.shape

# Get all forehead landmark coordinates
forehead_coords = []
for idx in forehead_indices:
    lm = last_landmarks.landmark[idx]
    if lm.visibility > 0.5:
        x = int(lm.x * w)
        y = int(lm.y * h)
        forehead_coords.append((x, y))

if forehead_coords:
    # Get bounding box of forehead region
    xs = [coord[0] for coord in forehead_coords]
    ys = [coord[1] for coord in forehead_coords]
    
    x1 = max(min(xs) - 20, 0)
    y1 = max(min(ys) - 20, 0)
    x2 = min(max(xs) + 20, w)
    y2 = min(max(ys) + 20, h)
    
    # Extract ROI
    forehead_roi = frame[y1:y2, x1:x2]
    
    # Use for signal processing
    signal = process_rppg(forehead_roi)
```

---

### 6. Quality Check from Statistics

```python
# Check if detection quality is good

if last_landmarks is not None:
    visibilities = [lm.visibility for lm in last_landmarks.landmark]
    avg_visibility = np.mean(visibilities)
    confident_count = sum(1 for v in visibilities if v > 0.5)
    
    quality_percentage = (confident_count / 468) * 100
    
    if quality_percentage > 90 and avg_visibility > 0.80:
        print("✓ Excellent quality - ready for rPPG")
    elif quality_percentage > 70 and avg_visibility > 0.60:
        print("⚠ Good quality - acceptable for rPPG")
    else:
        print("❌ Poor quality - improve lighting/angle")
    
    print(f"Detected: {confident_count}/468 landmarks")
    print(f"Avg Visibility: {avg_visibility:.2f}")
```

---

### 7. Draw Custom Landmark Info

```python
# Show specific landmarks on frame

def draw_eye_centers(frame, landmarks):
    """Draw detected eye centers"""
    left_eye_idx = 33
    right_eye_idx = 263
    
    h, w, _ = frame.shape
    
    # Get landmarks
    left_eye = landmarks.landmark[left_eye_idx]
    right_eye = landmarks.landmark[right_eye_idx]
    
    # Convert to pixel coordinates
    left_x = int(left_eye.x * w)
    left_y = int(left_eye.y * h)
    right_x = int(right_eye.x * w)
    right_y = int(right_eye.y * h)
    
    # Draw circles at eye centers
    cv2.circle(frame, (left_x, left_y), 5, (255, 0, 0), -1)
    cv2.circle(frame, (right_x, right_y), 5, (255, 0, 0), -1)
    
    # Draw line between eyes
    cv2.line(frame, (left_x, left_y), (right_x, right_y), (255, 0, 0), 2)
    
    # Add labels
    cv2.putText(frame, "L", (left_x-15, left_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "R", (right_x-15, right_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame
```

---

### 8. Detect Specific Facial Expressions

```python
# Detect mouth opening

def detect_mouth_open(landmarks):
    """Check if mouth is open based on landmarks"""
    
    # Mouth landmarks
    mouth_top = landmarks.landmark[13]     # Upper lip center
    mouth_bottom = landmarks.landmark[14]  # Lower lip center
    
    # Calculate distance
    dy = abs(mouth_bottom.y - mouth_top.y)
    
    # Threshold for open mouth
    if dy > 0.05:  # Adjust threshold as needed
        return True, dy
    else:
        return False, dy

# Usage
if last_landmarks is not None:
    is_open, distance = detect_mouth_open(last_landmarks)
    if is_open:
        print(f"Mouth open (distance: {distance:.3f})")
    else:
        print("Mouth closed")
```

---

### 9. Confidence Filter

```python
# Only use highly confident landmarks

def get_high_confidence_landmarks(landmarks, threshold=0.7):
    """Extract only high-confidence landmarks"""
    
    confident = []
    for idx, lm in enumerate(landmarks.landmark):
        if lm.visibility >= threshold:
            confident.append({
                "index": idx,
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
            })
    
    return confident

# Usage
high_conf_lms = get_high_confidence_landmarks(last_landmarks)
print(f"High confidence landmarks: {len(high_conf_lms)}")
```

---

### 10. Export Landmarks to File

```python
import json
import numpy as np

def export_landmarks(landmarks, filename="landmarks.json"):
    """Export landmarks to JSON file"""
    
    data = {
        "total": 468,
        "timestamp": time.time(),
        "landmarks": []
    }
    
    for idx, lm in enumerate(landmarks.landmark):
        data["landmarks"].append({
            "index": idx,
            "x": float(lm.x),
            "y": float(lm.y),
            "z": float(lm.z),
            "visibility": float(lm.visibility),
        })
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported to {filename}")

# Usage
if last_landmarks is not None:
    export_landmarks(last_landmarks)
```

---

## 📊 Landmark Index Reference

### All 468 Landmarks Grouped

```python
# Quick reference for important landmark indices

LANDMARK_INDICES = {
    # Left Face Contour
    "left_face_contour": list(range(0, 9)),
    
    # Nose
    "nose_bridge": [1, 2, 3],
    "nose_tip": [4],
    "nose_bottom": [5, 6],
    "nostril_left": [97, 98],
    "nostril_right": [327, 328],
    
    # Left Eye
    "left_eye": list(range(33, 48)),
    "left_iris": list(range(468, 469)),
    
    # Right Eye
    "right_eye": list(range(362, 386)),
    "right_iris": list(range(469, 470)),
    
    # Left Eyebrow
    "left_eyebrow": list(range(51, 57)),
    
    # Right Eyebrow
    "right_eyebrow": list(range(281, 287)),
    
    # Mouth Upper
    "mouth_upper": list(range(61, 68)),
    
    # Mouth Lower
    "mouth_lower": list(range(82, 88)),
    
    # Mouth Inner
    "mouth_inner": list(range(78, 82)),
    
    # Chin
    "chin": [152],
    
    # Left Cheek
    "left_cheek": [50],
    
    # Right Cheek
    "right_cheek": [280],
}
```

---

## 🎨 Color Reference

```python
# Landmark region colors (BGR format)

COLOR_SCHEME = {
    "lips": (0, 0, 255),           # Red
    "eyes": (255, 0, 0),           # Blue
    "eyebrows": (0, 165, 255),     # Orange
    "nose": (0, 255, 0),           # Green
    "face_contour": (255, 255, 0), # Cyan
    "left_cheek": (255, 0, 255),   # Magenta
    "right_cheek": (128, 0, 128),  # Purple
    "unknown": (200, 200, 200),    # Gray
}

# How to use
region, color = get_landmark_region(idx)
print(f"Landmark {idx} is in {region} region")
print(f"Color to use: BGR{color}")
```

---

## 📈 Performance Metrics

```python
import time

def benchmark_detection(frame, num_iterations=10):
    """Benchmark face detection speed"""
    
    times = []
    for _ in range(num_iterations):
        t0 = time.time()
        roi = extract_face(frame)
        t1 = time.time()
        times.append((t1 - t0) * 1000)  # Convert to ms
    
    print(f"Detection: {np.mean(times):.2f}ms ± {np.std(times):.2f}ms")
    print(f"FPS: {1000/np.mean(times):.1f}")
```

---

## 🐛 Debugging Tips

### Check if Landmarks Exist

```python
if last_landmarks is None:
    print("ERROR: No landmarks detected")
    print("Reasons:")
    print("  1. Face not in frame")
    print("  2. Poor lighting")
    print("  3. Face at extreme angle")
    print("  4. MediaPipe not loaded")
else:
    print(f"SUCCESS: {len(last_landmarks.landmark)} landmarks detected")
```

### Visualize Specific Region

```python
def highlight_region(frame, landmarks, region_name):
    """Highlight specific region on frame"""
    
    h, w, _ = frame.shape
    
    # Get all points in region
    points = []
    for idx in range(468):
        lm = landmarks.landmark[idx]
        region, _ = get_landmark_region(idx)
        if region == region_name:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))
    
    # Draw points
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)
    
    # Draw convex hull
    if len(points) > 3:
        hull = cv2.convexHull(np.array(points))
        cv2.drawContours(frame, [hull], 0, (0, 255, 0), 2)
    
    return frame
```

---

## 📝 Example Output Log

```
[INFO] Starting webcam with model=advanced, fast_mode=False, fps_target=30...

[INFO] Loaded TorchScript model from ket_qua/advanced_torchscript.pt

[DEBUG] loop start frame_index=0 buffer=0 last_bbox=None
[DEBUG] cap.read ret=True frame_shape=(650, 800, 3)

[INFO] Mediapipe loaded successfully

Frame 1:
  ✓ Face detected at bbox=(100, 150, 500, 600)
  ✓ Landmarks: 468 detected, 456 confident (>0.5)
  ✓ Orientation: Yaw=5.2°, Pitch=-10.3°, Roll=2.1°
  ✓ FPS: 32

Frame 2:
  ✓ Using face tracker (skip detection)
  ✓ BPM: 75 ± 3 bpm
  ✓ Signal quality: ✓ GOOD
  ✓ FPS: 35

Frame 3:
  ✓ Using face tracker
  ✓ BPM: 74 ± 2 bpm
  ✓ FPS: 36

[Press Q to quit]
```

---

## 🎓 Integration Template

```python
# Template for integrating landmarks into custom application

from collections import deque
import numpy as np

class LandmarkProcessor:
    def __init__(self):
        self.landmark_history = deque(maxlen=30)  # Store last 30 frames
        
    def process_frame(self, frame):
        """Process single frame"""
        
        # Get face ROI
        roi = extract_face(frame)
        
        if roi is None:
            return None
        
        # Get landmarks
        if last_landmarks is not None:
            # Extract key information
            key_lms = get_key_landmarks(last_landmarks)
            orientation = last_orientation
            
            # Store in history
            self.landmark_history.append({
                "timestamp": time.time(),
                "key_landmarks": key_lms,
                "orientation": orientation,
                "confidence": self._compute_confidence(),
            })
            
            # Process accumulated data
            if len(self.landmark_history) >= 10:
                self.analyze_temporal_trends()
        
        return roi
    
    def _compute_confidence(self):
        """Calculate overall confidence"""
        if last_landmarks is None:
            return 0.0
        visibilities = [lm.visibility for lm in last_landmarks.landmark]
        return np.mean(visibilities)
    
    def analyze_temporal_trends(self):
        """Analyze changes over time"""
        # Your custom analysis here
        pass

# Usage
processor = LandmarkProcessor()
while True:
    ret, frame = cap.read()
    roi = processor.process_frame(frame)
    # Use roi for rPPG or other analysis
```

---

**Last Updated**: 2026-06-12  
**Version**: 1.0  
**Type**: Example Reference Guide  

