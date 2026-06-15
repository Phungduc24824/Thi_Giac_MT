# Face Detection & Landmark Tracking - Comprehensive Guide

## Overview

This document describes the face detection and landmark tracking system used in the rPPG (remote Photoplethysmography) webcam pipeline. The system uses **MediaPipe Face Mesh** as the primary detector with fallback options for robust real-time performance.

---

## 1. Architecture

### 1.1 Detection Pipeline

```
Input Frame
    ↓
[Haar Cascade Detection] (Fast, on CPU)
    ↓ (if detected)
    ✓ Tracker Initialized
    ↓
[MediaPipe Face Mesh] (468 landmarks, 3D)
    ↓ (if landmarks detected)
    ✓ Extract ROI + Orientation
    ↓
[MTCNN Fallback] (if MediaPipe fails)
    ↓
Output: Face ROI + Landmarks + Orientation
```

### 1.2 Components

| Component | Purpose | Performance |
|-----------|---------|-------------|
| **Haar Cascade** | Fast face detection (mobile mode) | ~1-2ms |
| **MediaPipe Face Mesh** | 468 3D landmark detection | ~10-15ms on CPU |
| **MTCNN** | Fallback detector (if MTCNN installed) | ~20-30ms |
| **Face Tracker (MOSSE)** | Track face between detection frames | ~0.5-1ms |

---

## 2. MediaPipe Face Mesh Details

### 2.1 Key Features

- **468 Landmarks**: Complete 3D face mapping
- **Real-time Performance**: Works on CPU at ~30 FPS
- **3D Coordinates**: (x, y, z) for each landmark
- **Visibility Scores**: Confidence for each landmark (0-1)
- **Lightweight**: ~4.2 MB model size

### 2.2 Landmark Organization

Face landmarks are grouped into regions:

```
┌─────────────────────────────────────┐
│          FOREHEAD (0-10)            │
├─────────────────────────────────────┤
│  EYEBROWS (51-75)      EYES (33-48) │
│                        PUPILS(468)  │
│  CHEEKS (50-280)      PUPILS(468)   │
│                                      │
│      NOSE (1-6, 195-212)            │
│                                      │
│   LIPS (61-69, 78-96)               │
│   FACE CONTOUR (0-17, 227-232)     │
│   CHIN (152)                        │
└─────────────────────────────────────┘
```

### 2.3 Landmark Indices (Key Points)

| Landmark | Index | Region | 3D Position |
|----------|-------|--------|-------------|
| Nose Tip | 4 | Center | Forward |
| Left Eye | 33 | Left | Recessed |
| Right Eye | 263 | Right | Recessed |
| Left Mouth | 61 | Lower Left | Protruding |
| Right Mouth | 291 | Lower Right | Protruding |
| Chin | 152 | Bottom Center | Forward |
| Forehead | 10 | Top Center | Forward |

---

## 3. Face Detection Functions

### 3.1 Main Detection: `extract_face(frame, use_mobile=False)`

**Purpose**: Detect face and extract normalized ROI

**Parameters**:
- `frame`: Input BGR frame
- `use_mobile`: Use fast Haar cascade instead of MediaPipe

**Returns**: Normalized 32×32 ROI or None

**Process**:
1. Resize frame to 160×160 for faster processing
2. Convert BGR → RGB
3. Run MediaPipe Face Mesh detection
4. Extract multi-region ROI (eyes, nose, cheeks, forehead)
5. Extract 3D face orientation
6. Initialize face tracker
7. Normalize ROI (subtract ImageNet mean, divide by std)

### 3.2 Multi-Region ROI: `extract_multi_region_roi(frame, landmarks)`

**Why Multiple Regions?**
- Forehead alone may miss motion in eyes/cheeks
- Better signal capture for rPPG (blood flow visible in multiple areas)
- More robust to head rotation

**Regions Used**:
```python
{
    "forehead": [10, 67, 103, 109, 338],    # 5 points
    "eyes": [33, 133, 362, 263],             # 4 points
    "cheeks": [50, 280],                     # 2 points
    "nose": [1, 4, 195],                     # 3 points
}
```

### 3.3 3D Orientation: `compute_3d_orientation(landmarks)`

**Extracts**: Yaw, Pitch, Roll angles

**Method**:
- Forehead-to-chin vector → Pitch (vertical tilt)
- Left ear-to-right ear vector → Yaw (horizontal rotation)
- Eye symmetry → Roll (lateral tilt)

**Output**:
```python
{
    "pitch": float,  # -90 to 90 degrees
    "yaw": float,    # -90 to 90 degrees
    "roll": float,   # -45 to 45 degrees
}
```

---

## 4. Landmark Visualization Functions

### 4.1 `draw_face_landmarks(frame, landmarks, draw_connections=True, ...)`

**Features**:
- ✓ Color-coded landmarks by facial region
- ✓ Shows 468 individual landmarks
- ✓ Draws connections between adjacent landmarks
- ✓ Respects visibility/confidence scores
- ✓ Anti-aliased rendering

**Region Colors** (BGR):
| Region | Color | BGR |
|--------|-------|-----|
| Lips | Red | (0, 0, 255) |
| Eyes | Blue | (255, 0, 0) |
| Eyebrows | Orange | (0, 165, 255) |
| Nose | Green | (0, 255, 0) |
| Face Contour | Cyan | (255, 255, 0) |
| Left Cheek | Magenta | (255, 0, 255) |
| Right Cheek | Purple | (128, 0, 128) |

### 4.2 `draw_face_bounding_box(frame, landmarks, ...)`

**Purpose**: Draw tight bounding box around all landmarks

**Features**:
- ✓ Automatically computed from landmark bounds
- ✓ 10-pixel margin for stability
- ✓ Green rectangle (default)

### 4.3 `get_key_landmarks(landmarks)`

**Returns**: Dictionary of 7 key points

```python
{
    "nose_tip": {"x": 0.5, "y": 0.3, "z": -0.1, "visibility": 0.95},
    "left_eye": {"x": 0.35, "y": 0.2, ...},
    "right_eye": {"x": 0.65, "y": 0.2, ...},
    "left_mouth": {"x": 0.3, "y": 0.6, ...},
    "right_mouth": {"x": 0.7, "y": 0.6, ...},
    "chin": {"x": 0.5, "y": 0.85, ...},
    "forehead": {"x": 0.5, "y": 0.05, ...},
}
```

### 4.4 `draw_key_landmarks_text(frame, key_landmarks, ...)`

**Displays**: Precise 3D coordinates of key landmarks on frame

**Example Output**:
```
Key Landmarks:
nose_tip: (0.500, 0.300, -0.100)
left_eye: (0.350, 0.200, -0.050)
```

### 4.5 `draw_landmark_statistics(frame, landmarks, ...)`

**Displays**:
- Total landmarks detected (468)
- Number with high confidence (>0.5)
- Average visibility score

**Example Output**:
```
Total Landmarks: 468
Confident (>0.5): 456/468
Avg Visibility: 0.94
```

---

## 5. Usage in Main Loop

### 5.1 Running with Landmark Visualization

```bash
# Default mode (with full visualization)
python web_cam.py --model advanced

# Mobile mode (fast, minimal visualization)
python web_cam.py --model mobile --fast

# Custom FPS
python web_cam.py --fps 60
```

### 5.2 Real-Time Display

The webcam window shows:
1. **Green Bounding Box**: Face region
2. **Colored Landmarks**: 468 points by region
3. **Connections**: Face mesh structure
4. **Key Landmarks**: 7 important points (top-right)
5. **Statistics**: Detection quality (bottom-left)
6. **Orientation**: Head rotation angles (top-left)
7. **rPPG Signal**: Heart rate trend (bottom)
8. **BPM**: Estimated heart rate (top)

### 5.3 Global Variables

| Variable | Type | Purpose |
|----------|------|---------|
| `last_bbox` | tuple | Current face bounding box (x1, y1, x2, y2) |
| `last_landmarks` | MediaPipe object | 468 landmarks from last detection |
| `last_orientation` | dict | Current head orientation angles |
| `face_tracker` | cv2.Tracker | Tracker between detection frames |
| `tracker_initialized` | bool | Tracker status |

---

## 6. Performance Optimization Tips

### 6.1 Reduce Processing Load

```python
# Skip detection every N frames (use tracker instead)
if frame_index % 6 == 0:
    roi = extract_face(frame)  # Full detection
else:
    roi = extract_tracked_face(frame)  # Fast tracking
```

### 6.2 Downsampling Strategy

```python
# Downsample for faster landmark detection
small_size = 160  # Instead of full 800x650
small = cv2.resize(frame, (small_size, small_size))
results = face_mesh.process(small)
```

### 6.3 Mobile Mode

```python
# Use Haar cascade instead of MediaPipe
roi = extract_face(frame, use_mobile=True)
```

### 6.4 Threading for Inference

```python
# Inference happens in background thread
inference_worker(model, device, input_queue, output_queue)
# Main thread continues face detection/display
```

---

## 7. Troubleshooting

### Issue: Landmarks not appearing

**Cause**: MediaPipe not imported or face_mesh is None

**Solution**:
```python
# Check if MediaPipe loaded
if face_mesh is None:
    print("MediaPipe import failed")
    # Falls back to Haar cascade only
```

### Issue: Poor landmark detection in profile/side views

**Cause**: MediaPipe optimized for frontal views

**Solution**:
- Use multi-region ROI (extracts from cheeks, eyes, forehead)
- Check orientation angles (yaw > 45° = severe profile)
- Use lower confidence threshold if needed

### Issue: Flickering or jumpy landmarks

**Cause**: Detection lost/regained each frame

**Solution**:
- Use face tracker between detections
- Implement landmark smoothing (temporal filtering)
- Check lighting conditions

### Issue: High CPU usage

**Cause**: Full-frame MediaPipe processing

**Solution**:
```python
# Skip detection on some frames
detection_skip = 6  # Detect every 6 frames only
```

---

## 8. Advanced Features

### 8.1 Landmark-Based Gaze Estimation

Using eye landmarks (indices 33-48, 362-373) to estimate gaze direction:

```python
left_eye_center = np.mean([lm for idx in range(33, 42) 
                           for lm in [landmarks.landmark[idx]]], axis=0)
right_eye_center = np.mean([lm for idx in range(362, 371) 
                            for lm in [landmarks.landmark[idx]]], axis=0)
```

### 8.2 Mouth State Detection

Using mouth landmarks (indices 61-96) to detect:
- Open/closed mouth
- Smile intensity
- Speech

### 8.3 Eye Closure Detection

Compare eye region landmarks to detect:
- Blink events
- Eye closure duration
- Fatigue signs

---

## 9. References

### MediaPipe Documentation
- Official: https://mediapipe.dev/
- Face Mesh: https://mediapipe.dev/solutions/face_mesh
- 468 Landmarks: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

### Papers
- Face Detection: "BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs"
- Landmarks: "Real-time Joint Tracking of a Hand Manipulating an Object in an RGB-D Stream"

---

## 10. Summary Table

| Feature | Method | Speed | Accuracy | Notes |
|---------|--------|-------|----------|-------|
| Face Detection | Haar Cascade | ⚡⚡⚡ | ⭐⭐ | Fast, mobile-friendly |
| | MediaPipe | ⚡⚡ | ⭐⭐⭐⭐ | Recommended, CPU real-time |
| | MTCNN | ⚡ | ⭐⭐⭐⭐⭐ | Most accurate, slower |
| Landmark Tracking | MOSSE Tracker | ⚡⚡⚡ | ⭐⭐⭐ | Between-frame tracking |
| Landmark Visualization | Custom renderer | ⚡⚡ | N/A | 468 points, color-coded |
| 3D Orientation | Geometric | ⚡⚡⚡ | ⭐⭐⭐ | Real-time head angles |

---

**Last Updated**: 2026-06-12
**System**: MediaPipe Face Mesh v0.8+
**Python Version**: 3.8+
