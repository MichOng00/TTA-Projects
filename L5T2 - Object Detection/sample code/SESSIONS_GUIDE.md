# Mystic Shelf — 6-Session Learning Progression

A structured learning path from basic webcam capture to a complete hand-gesture-controlled game using MediaPipe and PyGame.

## Overview

This series teaches:
- **Hand detection** using Google's MediaPipe library
- **Gesture recognition** (pinch detection)
- **Physics simulation** (bouncing, gravity, collision)
- **Game development** (objects, state management, UI)
- **Python best practices** (OOP, event handling, real-time processing)

## Session Progression

### **Session 01: Webcam + Hand Detection** (`session_01_webcam_hand_detection.py`)
**Duration:** 1-2 hours | **Difficulty:** Beginner

**Concepts:**
- Opening and configuring a webcam with OpenCV
- Loading MediaPipe HandLandmarker model
- Converting between image formats (BGR ↔ RGB)
- Displaying video in PyGame
- Basic event handling

**Learning Goals:**
- Understand how to capture real-time video
- Learn MediaPipe API basics
- Display live camera feed with PyGame
- Count detected hands

**Activities:**
- Run the script and see hand count update
- Try with different lighting conditions
- Experiment with hand distance from camera

---

### **Session 02: Hand Pose Visualization** (`session_02_hand_visualization.py`)
**Duration:** 1-2 hours | **Difficulty:** Beginner

**Concepts:**
- Hand landmark coordinates and indices (21 points per hand)
- Skeleton connections (hand bones structure)
- Distinguishing important landmarks (fingertips, thumb, index)
- Normalized vs pixel coordinates
- Drawing geometric shapes on frames

**Learning Goals:**
- Understand hand landmark anatomy (21 keypoints)
- Draw hand skeleton (connections between landmarks)
- Visualize fingertips with different colors
- Build foundation for gesture recognition

**Activities:**
- Identify all 21 hand landmarks on your own hand
- Trace the hand skeleton with your fingers
- Notice how landmarks change with different hand poses

---

### **Session 03: Pinch Detection & Interaction** (`session_03_pinch_detection.py`)
**Duration:** 1.5-2 hours | **Difficulty:** Intermediate

**Concepts:**
- Distance calculation between two 2D points
- Gesture detection (distance thresholds)
- Pinch gesture (thumb ↔ index finger proximity)
- Real-time status feedback
- Visual cues for gesture state

**Learning Goals:**
- Calculate Euclidean distance between landmarks
- Detect pinch gesture when distance < threshold
- Provide visual feedback (circle color changes)
- Create interactive response to user gesture

**Key Code:**
```python
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

pinch_distance = dist(thumb_pos, index_pos)
is_pinching = pinch_distance < PINCH_DIST_PX
```

**Activities:**
- Practice smooth pinch gestures
- Experiment with different threshold values
- Notice how distance displayed in pixels changes
- Try pinching with both hands

---

### **Session 04: Orb Physics & Pedestals** (`session_04_orb_physics.py`)
**Duration:** 2-3 hours | **Difficulty:** Intermediate

**Concepts:**
- Object-oriented design (Orb, Pedestal classes)
- Physics simulation (velocity, acceleration, bouncing)
- Boundary collision detection
- Glow/lighting effects
- Color management and theming
- Smooth animation (wobble effect)

**Learning Goals:**
- Design reusable game object classes
- Implement simple physics with velocity and acceleration
- Handle wall collisions
- Create visually appealing effects (glow, core)
- Organize game state

**Key Code:**
```python
class Orb:
    def update(self, dt, w, h):
        self.pos[0] += self.vel[0] * dt
        # Bounce off walls
        if self.pos[0] < pad:
            self.vel[0] = abs(self.vel[0])
```

**Activities:**
- Adjust initial velocity ranges
- Modify glow layers for different visual styles
- Create custom colors (modify COLORS dict)
- Add gravity effect (uncomment + vy gravity)

---

### **Session 05: Orb Interaction & Snapping** (`session_05_orb_interaction.py`)
**Duration:** 2-3 hours | **Difficulty:** Intermediate

**Concepts:**
- Object grabbing (selecting objects near gesture)
- Smooth dragging (lerp interpolation)
- Snap-to-target detection
- Particle effects (visual feedback)
- State transitions (grabbed → placed)
- Score tracking

**Learning Goals:**
- Implement interactive object manipulation
- Use lerp for smooth motion
- Detect when objects reach targets
- Create particle burst effects
- Manage multiple object states
- Track game progress (score)

**Key Code:**
```python
grabbed_orb.pos[0] = lerp(grabbed_orb.pos[0], pinch_pos[0], 0.35)

if dist(grabbed_orb.pos, ped.pos) < SNAP_DIST_PX:
    grabbed_orb.placed = True
    particles.append(Particle(ped.pos, grabbed_orb.color))
```

**Activities:**
- Play the game and place all 5 orbs
- Adjust snap distance to make game easier/harder
- Modify lerp factor for faster/slower drag
- Increase particle count for more effects

---

### **Session 06: Complete Game with Timer & Win State** (`session_06_mystic_shelf.py`)
**Duration:** 2-3 hours | **Difficulty:** Advanced

**Concepts:**
- Game loop architecture
- Win condition and state management
- Timer system (elapsed time tracking)
- UI rendering (HUD)
- Game restart/reset
- Sound effects (optional enhancement)

**Learning Goals:**
- Build a complete, playable game
- Implement timer and win screen
- Create restart functionality
- Polish UI with clear instructions
- Integrate all previous concepts

**New Features:**
- Elapsed time tracking (timer on HUD)
- Win detection when all orbs placed
- Flashing victory message
- Restart with 'R' key
- Clean game state reset
- Professional UI layout

**Activities:**
- Play the complete game
- Try different hand positions/distances
- Customize the win screen
- Add a high-score system
- Create difficulty levels (fewer orbs, tighter snap distance)

---

## Installation & Setup

### Requirements
```bash
pip install opencv-python pygame "mediapipe>=0.10.33" numpy
```

### First Run
The first time you run any script, it will automatically download the MediaPipe hand landmarker model (~8 MB). This is cached next to the script as `hand_landmarker.task`.

### Running Each Session
```bash
# From the terminal in the sample code directory:
python session_01_webcam_hand_detection.py
python session_02_hand_visualization.py
python session_03_pinch_detection.py
python session_04_orb_physics.py
python session_05_orb_interaction.py
python session_06_mystic_shelf.py
```

## Teaching Tips

### For Session 1-2
- Focus on understanding the MediaPipe API
- Explore different hand poses and how landmarks move
- Discuss the 21-point skeleton structure

### For Session 3
- Emphasize gesture recognition as the bridge between tracking and interaction
- Experiment with different distance thresholds
- Show how real-world applications use similar techniques (sign language, VR control)

### For Session 4
- Discuss physics concepts (velocity, acceleration, time delta)
- Explain why dt (delta time) matters for frame-rate independence
- Experiment with different forces (gravity, friction)

### For Session 5
- Focus on user experience (feedback, snapping, particles)
- Discuss game feel and polish
- Show how small details make big differences

### For Session 6
- Talk about game architecture and main loop
- Discuss win conditions and game states
- Brainstorm enhancements (levels, power-ups, leaderboards)

## Common Modifications

### Change Game Difficulty
```python
# Session 05-06: Adjust snap distance
SNAP_DIST_PX = 50    # Harder (smaller target)
SNAP_DIST_PX = 100   # Easier (larger target)

# Adjust pinch threshold
PINCH_DIST_PX = 40   # Requires tighter pinch
PINCH_DIST_PX = 80   # More forgiving
```

### Add/Remove Orbs
```python
# In COLORS dict
COLORS = {
    "fire":   (45, 100, 255),
    "ice":    (230, 210, 80),
    "nature": (50, 200, 80),
    "shadow": (170, 50, 180),
    "light":  (50, 220, 230),
    "lightning": (255, 255, 100),  # Add new color
}
```

### Customize Colors
```python
# Modify color tuples (BGR format in OpenCV)
COLORS = {
    "fire":   (255, 0, 0),      # Pure blue in BGR
    "ice":    (255, 255, 0),    # Pure cyan in BGR
    ...
}
```

### Adjust Physics
```python
# Session 04: Initial velocity range
self.vel = [random.uniform(-100, 100), random.uniform(-100, 100)]  # Faster

# Add gravity
self.vy += 120 * dt  # In Particle.update()
```

## Troubleshooting

### Webcam Issues
- Try running with different camera indices: `cv2.VideoCapture(1)` instead of `cv2.VideoCapture(0)`
- Check camera permissions (macOS/Linux)
- Ensure good lighting for hand detection

### Hand Detection Problems
- Increase confidence thresholds if false positives occur:
  ```python
  min_hand_detection_confidence=0.8  # Default 0.6
  ```
- Decrease threshold if hands aren't detected:
  ```python
  min_hand_detection_confidence=0.4  # Default 0.6
  ```

### Performance Issues
- Reduce video resolution
- Lower FPS target (30 instead of 60)
- Disable hand visualization in later sessions

## Extension Ideas

1. **Multi-hand gestures** - Require both hands for certain actions
2. **Difficulty levels** - Progressive stages with more orbs
3. **Sound effects** - Add audio feedback for placement
4. **Animations** - Smoother transitions, orb trails
5. **Leaderboard** - Track best times
6. **AI opponent** - Computer-controlled orbs
7. **Network multiplayer** - Two players competing
8. **Custom gestures** - Rock-paper-scissors, thumbs up, etc.
9. **AR overlay** - Custom visual themes
10. **Mobile version** - Export to Android/iOS

## References

- **MediaPipe Hands**: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
- **PyGame Documentation**: https://www.pygame.org/docs/
- **OpenCV Camera Capture**: https://docs.opencv.org/master/d8/dfe/classcv_1_1VideoCapture.html
- **Hand Skeleton Reference**: See HAND_CONNECTIONS constant

## License & Attribution

Created for educational purposes. Uses:
- Google MediaPipe (Apache 2.0)
- PyGame (LGPL)
- OpenCV (Apache 2.0)

---

**Happy teaching! 🎮**
