# Mystic Shelf — 6-Session Learning Progression

A structured learning path from basic webcam capture to a complete hand-gesture-controlled game using MediaPipe and PyGame. **Each session builds directly on the previous one with minimal code deletion.**

## Overview

This series teaches:
- **Hand detection** using Google's MediaPipe library
- **Gesture recognition** (pinch detection)
- **Physics simulation** (bouncing, gravity, collision)
- **Game development** (objects, state management, UI)
- **Python best practices** (OOP, event handling, real-time processing)

## Cumulative Learning Structure

**Key Principle:** Each session adds new features to previous code WITHOUT deleting what came before.

```
Session 01: Basic hand detection
    ↓ ADD hand visualization
Session 02: Hand pose visualization  
    ↓ ADD pinch detection
Session 03: Pinch detection
    ↓ ADD game objects
Session 04: Orb physics & pedestals
    ↓ ADD interaction logic
Session 05: Orb interaction & snapping
    ↓ ADD timer & win state
Session 06: Complete game with scoring
```

This means you can copy Session 01's code, add to it to make Session 02, copy that and add to it for Session 03, etc. **No code is removed—only added to!**

## Session Progression

### **Session 01: Webcam + Hand Detection** (`session_01_webcam_hand_detection.py`)
**Duration:** 1-2 hours | **Difficulty:** Beginner | **Code Added:** ~120 lines

**What This Session Adds:**
- Camera capture setup with OpenCV
- MediaPipe HandLandmarker initialization
- PyGame window and basic display
- Hand detection loop (count hands)

**Code You'll Write:**
- `ensure_model()` - Download model if needed
- Basic `run()` main loop with hand detection
- Event handling (quit on Q key)

**What You Can Do:**
- Open a webcam and see hand count displayed
- Flip the image for selfie view
- Handle camera errors gracefully

---

### **Session 02: Hand Pose Visualization** (`session_02_hand_visualization.py`)
**Duration:** 1-2 hours | **Difficulty:** Beginner | **Code Added:** ~150 lines

**What This Session Adds (Keep All of Session 01):**
- Hand landmark indices (TIP_THUMB, TIP_INDEX, FINGERTIP_INDICES)
- Hand skeleton connections (21 points, 20 bones)
- Draw hand skeleton on video
- Highlight fingertips with different colors
- Visualize thumb and index separately

**Code You'll Write:**
- Add constants for hand landmarks and connections
- Draw hand skeleton using HAND_CONNECTIONS
- Color-code different fingertips
- Display hand visualization in main loop

**What You Can Do:**
- See the complete hand skeleton structure (21 keypoints)
- Understand which finger is which
- Trace the hand anatomy on video
- See how landmarks move with different hand poses

---

### **Session 03: Pinch Detection & Interaction** (`session_03_pinch_detection.py`)
**Duration:** 1.5-2 hours | **Difficulty:** Intermediate | **Code Added:** ~200 lines

**What This Session Adds (Keep All of Sessions 01-02):**
- `dist()` function to calculate distance between points
- Pinch detection logic (distance < threshold)
- Pinch point calculation (midpoint between thumb/index)
- Visual feedback for pinch state
- Distance display in pixels

**Code You'll Write:**
- Implement `dist(a, b)` function
- Calculate distance between thumb and index
- Determine if pinching (distance < PINCH_DIST_PX)
- Draw circles for pinch points (green = pinching, blue = not)
- Display distance values

**Key Insight:**
```python
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

pinch_distance = dist(thumb_pos, index_pos)
is_pinching = pinch_distance < PINCH_DIST_PX
```

**What You Can Do:**
- Detect when you pinch your fingers
- See the pinch gesture in real-time
- Understand gesture recognition basics
- Experiment with different distance thresholds

---

### **Session 04: Orb Physics & Pedestals** (`session_04_orb_physics.py`)
**Duration:** 2-3 hours | **Difficulty:** Intermediate | **Code Added:** ~450 lines

**What This Session Adds (Keep All of Sessions 01-03):**
- `Orb` class with physics (velocity, bouncing)
- `Pedestal` class as target platforms
- `draw_glow_circle()` for visual effects
- `draw_pedestal()` for target visualization
- `make_game()` to initialize game objects
- Color themes (fire, ice, nature, shadow, light)
- Physics updates in main loop (still shows hand visualization!)

**Code You'll Write:**
- Design Orb class with position, velocity, wobble
- Implement bouncing physics with wall collision
- Design Pedestal class with animation
- Add glow effects and core highlighting
- Initialize 5 colorful orbs and pedestals
- Update orbs in main loop

**Key Concepts:**
- Physics simulation (velocity × time)
- Boundary collision detection
- Object state (wobble animation)
- Visual polish (glow, shadows)

**What You Can Do:**
- See orbs bouncing around the screen
- Watch pedestals pulse with light
- Understand how to add game objects to an existing system
- See hand visualization while game objects update

---

### **Session 05: Orb Interaction & Snapping** (`session_05_orb_interaction.py`)
**Duration:** 2-3 hours | **Difficulty:** Intermediate | **Code Added:** ~300 lines

**What This Session Adds (Keep All of Sessions 01-04):**
- `Particle` class for visual effects
- `lerp()` function for smooth motion
- Orb grabbing logic (select nearest orb)
- Smooth dragging (interpolation toward pinch point)
- Snap detection (orb within threshold of matching pedestal)
- Score tracking
- State management (grabbed, placed, free)

**Code You'll Write:**
- Implement `lerp(a, b, t)` for smooth interpolation
- Create Particle class with velocity and fade
- Detect which orb to grab (nearest to pinch)
- Drag grabbed orb smoothly toward pinch point
- Detect snap to matching pedestal
- Create particle burst on successful placement
- Track score and game progress

**Key Insight:**
```python
def lerp(a, b, t):
    return a + (b - a) * t  # Smooth movement

grabbed_orb.pos[0] = lerp(grabbed_orb.pos[0], pinch_pos[0], 0.35)
```

**What You Can Do:**
- Grab and drag orbs with pinch gesture
- Watch orbs snap into place
- See particle effects on placement
- Play the matching game mechanics
- Try to match all 5 colors

---

### **Session 06: Complete Game with Timer & Win State** (`session_06_mystic_shelf.py`)
**Duration:** 2-3 hours | **Difficulty:** Advanced | **Code Added:** ~200 lines

**What This Session Adds (Keep All of Sessions 01-05):**
- Timer tracking with `start_time` and `win_time`
- Win condition detection (all orbs placed)
- Flashing victory screen with time display
- Game restart functionality (R key)
- Game state reset
- Professional HUD with timer and score
- Victory message with elapsed time

**Code You'll Write:**
- Track `start_time` when game starts
- Detect `win_time` when score == NUM_ORBS
- Display elapsed time in HUD
- Render victory screen with flashing text
- Implement restart logic (reset all variables)
- Update HUD to show timer

**What You Can Do:**
- Play the complete game from start to finish
- Race against the timer
- Restart and play again
- See your best times
- Challenge others to beat your score

---

## How to Progress Through Sessions

### Recommended Approach:

1. **Run each session's script** to see what you'll be building
2. **Read the code comments** to understand what's new
3. **Study the added code** - what functions/classes are new?
4. **Modify and experiment** - change colors, speeds, distances
5. **Build incrementally** - don't just copy-paste; type and understand

### Incremental Building (DIY):

If you want to learn by building from Session to Session:

1. Copy Session 01 → Session 02 (Keep all, add hand visualization)
2. Copy Session 02 → Session 03 (Keep all, add pinch detection)
3. Copy Session 03 → Session 04 (Keep all, add game objects)
4. Copy Session 04 → Session 05 (Keep all, add interaction)
5. Copy Session 05 → Session 06 (Keep all, add timer/win)

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

### Cumulative Teaching Approach

The sessions are designed so **nothing is removed—only added to**. This means:

- **Session 1 → 2:** Students keep their camera/detection code and ADD hand visualization
- **Session 2 → 3:** Students keep hand viz and ADD pinch detection
- **Session 3 → 4:** Students keep pinch detection and ADD game objects
- **Session 4 → 5:** Students keep objects and ADD interaction logic
- **Session 5 → 6:** Students keep interaction and ADD timer/win state

**Benefit:** Students see how features layer on top of each other. They understand integration deeply.

### For Session 1-2 (Foundations)
- Emphasize **reading data from camera** (real-time processing)
- Show how **MediaPipe provides normalized coordinates** (0-1 scale)
- Discuss **converting to pixel coordinates** (multiply by width/height)
- Have students trace their hand skeleton with their finger
- Show how 21 keypoints form the complete hand structure
- Let them experiment with different hand poses

### For Session 3 (Gesture Recognition)
- **Focus on distance math**: How does `hypot` work?
- **Threshold importance**: Why do we need PINCH_DIST_PX?
- **Real-world applications**: VR controllers, sign language, accessibility
- Let students **adjust thresholds** and see what happens
- Show how small distance changes affect gesture detection
- Discuss **feedback**: Why is visual feedback important?

### For Session 4 (Game Objects)
- **Discuss physics**: velocity, time delta, bounce
- **Why delta time (dt) matters**: Frame-rate independence
- **Object-oriented design**: Why classes help organize code
- **Visual polish**: How glow effects improve game feel
- Have students **customize colors** (COLORS dict)
- Show how game objects and hand tracking **coexist**

### For Session 5 (Interaction)
- **User experience matters**: Smooth dragging feels good
- **lerp function**: How interpolation smooths motion
- **State management**: Objects have multiple states (free, grabbed, placed)
- **Feedback loops**: Particles celebrate successful actions
- **Score systems**: How to motivate players
- Let students **experiment with lerp strength** (0.2 vs 0.5 vs 0.8)

### For Session 6 (Complete Game)
- **Game architecture**: Main loop structure
- **State machines**: Playing vs Won states
- **Timer systems**: Tracking start time, elapsed time, win time
- **Restart logic**: Why clean reset matters
- **Competitive play**: Times encourage replaying
- **Scope creep**: What could be added (levels, multiplayer, etc.)?

### Classroom Activities

**For small groups:**
1. Have students compare their Session 5 code with the provided version
2. Identify exactly what they ADD in Session 6
3. Discuss: "What would YOU add next?"

**For competition:**
1. Use Session 6 as a base game
2. Challenge students to beat each other's times
3. Modify difficulty (snap distance, number of orbs)
4. Add difficulty levels with timers

**For deeper learning:**
1. "What if we added more hands?"
2. "How would you add sound effects?"
3. "How would you save high scores?"
4. "What other gestures could you detect?"
5. "How would you make this multiplayer?"

## Common Modifications by Session

### Session 01-02: Camera Setup
```python
# Use different camera
cv2.VideoCapture(1)  # Try 1, 2, 3... for different cameras

# Change resolution
TARGET_W = 640   # Lower = faster
TARGET_H = 480
```

### Session 03: Pinch Sensitivity
```python
PINCH_DIST_PX = 40    # Tighter pinch required (harder)
PINCH_DIST_PX = 100   # Looser pinch (easier)
```

### Session 04: Game Visuals
```python
# Add/remove orbs - modify COLORS dict
COLORS = {
    "fire":   (45, 100, 255),
    "ice":    (230, 210, 80),
    "nature": (50, 200, 80),
    "shadow": (170, 50, 180),
    "light":  (50, 220, 230),
    "storm":  (100, 50, 200),      # Add new color!
}

# Adjust orb speeds
self.vel = [random.uniform(-80, 80), ...]  # Faster
self.vel = [random.uniform(-20, 20), ...]  # Slower

# Add gravity
self.vy += 60 * dt  # Gravity pulls down
```

### Session 05: Interaction Tweaks
```python
# Make dragging more/less responsive
grabbed_orb.pos[0] = lerp(..., 0.2)  # Slower, stickier
grabbed_orb.pos[0] = lerp(..., 0.8)  # Faster, snappier

# Change snap difficulty
SNAP_DIST_PX = 50    # Harder (smaller target)
SNAP_DIST_PX = 120   # Easier (larger target)

# More/fewer particles
for _ in range(50):    # More particles
    particles.append(Particle(ped.pos, grabbed_orb.color))
```

### Session 06: Game Difficulty
```python
# Fewer orbs = easier game
COLORS = {
    "fire":   (45, 100, 255),
    "ice":    (230, 210, 80),
    "nature": (50, 200, 80),
    # Remove "shadow" and "light" for 3 orbs
}

# Challenge mode
SNAP_DIST_PX = 40

# Easy mode
SNAP_DIST_PX = 150
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

## Session Summary: What Gets Added at Each Level

| Session | Lines | Adds | Keeps | Running Demo |
|---------|-------|------|-------|--------------|
| 01 | ~120 | Camera, MediaPipe, detection | - | Hand counter |
| 02 | +150 | Hand skeleton viz | ✓ Session 01 | Hand with skeleton |
| 03 | +200 | Pinch detection | ✓ Sessions 01-02 | Pinch circles |
| 04 | +450 | Orbs, pedestals, physics | ✓ Sessions 01-03 | Bouncing game objects |
| 05 | +300 | Interaction, particles | ✓ Sessions 01-04 | Playable game |
| 06 | +200 | Timer, win condition | ✓ Sessions 01-05 | Complete game |
| **Total** | **~1,420** | | | **Full game** |

## For Students

Follow this progression:

1. **Complete Session 01** - Get comfortable with camera and hand detection
2. **Copy Session 01 → Create Session 02** - Add hand visualization to your existing code
3. **Copy Session 02 → Create Session 03** - Add pinch detection logic
4. **Copy Session 03 → Create Session 04** - Add game objects while keeping hand tracking
5. **Copy Session 04 → Create Session 05** - Add interaction and see the game come alive
6. **Copy Session 05 → Create Session 06** - Add timer and victory screen

**Key Learning:** Notice how each session's new code integrates seamlessly with everything before it. Nothing breaks, nothing is removed—you're just adding layers of functionality.

## For Instructors

**Cumulative approach benefits:**

✅ **No cognitive reset** - Students keep their working code as a foundation  
✅ **Integration practice** - Students learn how to layer features on top of existing systems  
✅ **Confidence building** - Early successes (Sessions 01-02) build toward complex projects (Sessions 05-06)  
✅ **Debugging skills** - When something breaks in Session 05, students know it came from that session  
✅ **Code comprehension** - Students can easily see what's new vs. what's inherited  

---

**Happy teaching! 🎮**
