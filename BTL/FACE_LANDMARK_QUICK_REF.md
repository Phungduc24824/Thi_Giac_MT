# Face Detection & Landmark Tracking - Quick Reference

## Running the Webcam with Landmarks

```bash
# Default mode - full visualization with landmarks
python web_cam.py --model advanced

# Faster mode - minimal UI but still detects landmarks
python web_cam.py --model mobile

# Efficient model with higher FPS target
python web_cam.py --model efficient --fps 60

# Fast mode without display (headless)
python web_cam.py --fast
```

## Real-Time Display Features

### On-Screen Information (Default Mode)

```
┌─────────────────────────────────────────────────┐
│                                                  │
│  Green Bounding Box: Face region                │
│  ├─ 468 Colored Dots: Landmarks                 │
│  ├─ Lines: Face mesh connections                │
│  └─ Transparency: Visibility confidence          │
│                                                  │
│  Top-Right Corner:                              │
│  ├─ Key Landmarks (7 points with 3D coords)     │
│  └─ Nose, Eyes, Mouth, Chin, Forehead           │
│                                                  │
│  Bottom-Left:                                   │
│  ├─ Total Landmarks: 468                        │
│  ├─ Confident (>0.5): XXX/468                   │
│  └─ Avg Visibility: 0.XX                        │
│                                                  │
│  Top-Left:                                      │
│  ├─ Yaw: ±XX°   (horizontal rotation)           │
│  ├─ Pitch: ±XX° (vertical tilt)                 │
│  └─ Roll: ±XX°  (lateral tilt)                  │
│                                                  │
│  Center-Top: BPM: XXX  (heart rate)             │
│  FPS: XXX                                       │
└─────────────────────────────────────────────────┘
```

## Landmark Color Guide

```
🔴 RED = LIPS
  - Indices: 61-69, 78-96 (upper/lower lips)
  - Use: Smile detection, speech analysis

🔵 BLUE = EYES
  - Indices: 33-48, 362-385 (pupils, lids, corners)
  - Use: Gaze tracking, blink detection, eye closure

🟠 ORANGE = EYEBROWS
  - Indices: 51-75, 295-303 (left/right eyebrows)
  - Use: Expression recognition, emotion detection

🟢 GREEN = NOSE
  - Indices: 1-6, 195-212 (tip, bridge, nostril)
  - Use: Face orientation, 3D head pose

🟦 CYAN = FACE CONTOUR
  - Indices: 0-17, 227-232 (face outline)
  - Use: Face shape analysis, alignment check

🟪 MAGENTA = LEFT CHEEK
  - Indices: 48-65, 209-226
  - Use: rPPG signal (blood flow detection)

🟣 PURPLE = RIGHT CHEEK
  - Indices: 278-295, 426-443
  - Use: rPPG signal (blood flow detection)
```

## Key Functions in Code

### 1. Display Landmarks with Connections
```python
frame = draw_face_landmarks(frame, last_landmarks, 
                             draw_connections=True, 
                             landmark_size=2)
```

### 2. Show Bounding Box from Landmarks
```python
frame = draw_face_bounding_box(frame, last_landmarks,
                                bbox_color=(0, 255, 0),
                                thickness=2)
```

### 3. Extract Key Points
```python
key_lms = get_key_landmarks(last_landmarks)
# Returns: {"nose_tip": {x, y, z, visibility}, ...}
```

### 4. Display Key Points as Text
```python
draw_key_landmarks_text(frame, key_lms, start_x=400, start_y=50)
# Shows coordinates of 7 key points on screen
```

### 5. Show Detection Quality
```python
draw_landmark_statistics(frame, last_landmarks, 
                         start_x=10, start_y=350)
# Shows: Total, confident count, avg visibility
```

## Accessing Landmark Data Programmatically

### All 468 Landmarks
```python
if last_landmarks is not None:
    for idx, landmark in enumerate(last_landmarks.landmark):
        x = landmark.x      # 0-1 (normalized)
        y = landmark.y      # 0-1 (normalized)
        z = landmark.z      # 3D depth (-1 to 1)
        visibility = landmark.visibility  # 0-1 confidence
        
        # Convert to pixel coordinates
        px = int(x * frame_width)
        py = int(y * frame_height)
```

### Key Landmarks Dictionary
```python
key_lms = get_key_landmarks(last_landmarks)

nose_x = key_lms["nose_tip"]["x"]
nose_y = key_lms["nose_tip"]["y"]
nose_z = key_lms["nose_tip"]["z"]
nose_vis = key_lms["nose_tip"]["visibility"]

# Get region for a specific landmark
region, color = get_landmark_region(landmark_idx=4)  # nose_tip
print(f"Region: {region}, Color: {color}")
```

## 3D Face Orientation (Head Pose)

The system extracts head rotation angles:

```python
if last_orientation is not None:
    yaw = last_orientation["yaw"]      # ±90° (left/right)
    pitch = last_orientation["pitch"]  # ±90° (up/down)
    roll = last_orientation["roll"]    # ±45° (tilt)
    
    # Interpretation:
    # yaw > 30°: Turn right
    # yaw < -30°: Turn left
    # pitch > 20°: Look down
    # pitch < -20°: Look up
    # roll > 10°: Tilt right
    # roll < -10°: Tilt left
```

## Performance Tips

### 1. Skip Detection on Some Frames
```python
detection_skip = 6  # Default in code
# Haar cascade + Tracker on other frames = faster processing
```

### 2. Reduce Visualization Overhead
```python
# Disable landmark visualization in mobile mode
if not overlay_mobile:
    frame = draw_face_landmarks(frame, last_landmarks)
```

### 3. Use Mobile Model for Speed
```bash
python web_cam.py --model mobile  # ~40 FPS vs 30 FPS
```

### 4. Increase Frame Rate
```bash
python web_cam.py --fps 60  # Higher FPS target
```

## Troubleshooting

### Landmarks Not Showing
```python
# Check if MediaPipe loaded
if face_mesh is None:
    print("MediaPipe not available - using Haar cascade only")
    # No landmarks will be shown

# Check if detection is working
if last_landmarks is None:
    print("Face not detected")
```

### Performance Degradation
```python
# Monitor statistics display
# If "Confident (>0.5)" is very low (< 300/468):
# - Improve lighting
# - Decrease distance to camera
# - Ensure face is frontal

# If FPS drops:
# - Use mobile model
# - Increase detection_skip
# - Reduce window size
```

### Jittery Landmarks
```python
# This is normal - landmarks update every 6 frames
# Between frames, face tracker predicts position

# To smooth: Apply temporal filter
import scipy.ndimage
smoothed_coords = scipy.ndimage.gaussian_filter1d(coords, sigma=1)
```

## Signal Quality Indicators

The visualization shows signal quality:

```
✓ GOOD  - Bright green text
         → Standard deviation > 0.01
         → Ready for rPPG estimation

⚠ WEAK  - Orange/yellow text
         → Standard deviation < 0.01
         → Poor signal quality
         → Move closer, improve lighting
```

## Integration with rPPG Pipeline

The landmarks help in:

1. **ROI Extraction** - Multi-region sampling
   - Forehead (high blood flow)
   - Cheeks (secondary signal)
   - Eyes (sometimes useful)

2. **Face Quality Check** - Visibility score
   - Skip frames with < 80% confident landmarks

3. **Motion Correction** - 3D orientation
   - Track head movement
   - Adjust ROI accordingly

4. **Signal Validation**
   - High landmark confidence = good video quality
   - Can correlate with rPPG signal quality

## Example: Custom Landmark Processing

```python
# Extract forehead landmarks for rPPG
forehead_indices = [10, 67, 103, 109, 338]
forehead_lms = []

for idx in forehead_indices:
    lm = last_landmarks.landmark[idx]
    if lm.visibility > 0.5:  # Only confident ones
        forehead_lms.append((lm.x, lm.y, lm.z))

# Compute center of forehead
if forehead_lms:
    center_x = np.mean([lm[0] for lm in forehead_lms])
    center_y = np.mean([lm[1] for lm in forehead_lms])
    center_z = np.mean([lm[2] for lm in forehead_lms])
```

## References

- **Landmark Indices**: [Face Mesh Landmarks Map](https://github.com/google/mediapipe/wiki/MediaPipe-Face-Mesh)
- **Performance**: ~10-15ms on CPU, 30 FPS real-time
- **Accuracy**: 98%+ for frontal faces, 70-80% for profile views

---

**Last Updated**: 2026-06-12
**Tested On**: Windows, Python 3.8+, MediaPipe 0.8+
