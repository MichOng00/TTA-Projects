# Hand Tracking Game Development Sessions

This series of Python scripts demonstrates progressive development of hand-tracking games using MediaPipe and OpenCV. Each session builds on the previous one, teaching fundamental concepts step by step.

## Session Overview

### Session 01: Basic Webcam Display (`session_01_webcam.py`)
**Goal**: Display webcam feed with proper cleanup
**Concepts**: OpenCV camera capture, window management, basic event loop
**New skills**: `cv2.VideoCapture()`, `cv2.imshow()`, `cv2.waitKey()`

### Session 02: Hand Detection (`session_02_hand_detection.py`)
**Goal**: Detect and display hand landmarks
**Concepts**: MediaPipe integration, landmark visualization
**New skills**: MediaPipe Hands, landmark coordinates, drawing utilities

### Session 03: Basic Interaction (`session_03_interaction.py`)
**Goal**: Detect when hand enters screen regions
**Concepts**: Coordinate conversion, region-based interaction
**New skills**: Landmark position access, collision detection with areas

### Session 04: Pinch Gestures (`session_04_pinch_gesture.py`)
**Goal**: Detect pinch gestures between thumb and index finger
**Concepts**: Gesture recognition, distance calculations
**New skills**: Multi-landmark calculations, gesture state management

### Session 05: Object Pickup (`session_05_object_pickup.py`)
**Goal**: Create interactive objects that can be grabbed with pinch
**Concepts**: Object-oriented design, grab/release mechanics
**New skills**: Object state management, point-in-circle collision

### Session 06: Physics and Gravity (`session_06_physics.py`)
**Goal**: Add realistic physics with gravity and bouncing
**Concepts**: Time-based updates, physics simulation
**New skills**: Velocity, acceleration, collision response, delta time

### Session 07: Multiple Objects (`session_07_multiple_objects.py`)
**Goal**: Handle multiple physics objects with collisions
**Concepts**: Object management, inter-object collision detection
**New skills**: Object-to-object collision, mouse input, dynamic object creation

### Session 08: Game Mechanics (`session_08_game_mechanics.py`)
**Goal**: Complete game with scoring, levels, and objectives
**Concepts**: Game state management, level progression, win/lose conditions
**New skills**: Game loops, scoring systems, UI design, time management

## Prerequisites

```bash
pip install opencv-python "mediapipe>=0.10.33" numpy
```

## How to Use

1. Start with Session 01 and work through each session in order
2. Read the comments and docstrings in each file
3. Run each script and observe the behavior
4. Modify the code to experiment with different values
5. Use the concepts from earlier sessions in later ones

## Learning Progression

Each session introduces 2-4 new concepts while building on previous knowledge:

- **Sessions 1-2**: Foundation (camera + hand detection)
- **Sessions 3-4**: Interaction (regions + gestures)
- **Sessions 5-6**: Objects + Physics (pickup + simulation)
- **Sessions 7-8**: Complexity (multiple objects + game design)

## Key Concepts Covered

- **Computer Vision**: Camera capture, image processing
- **Hand Tracking**: Landmark detection, gesture recognition
- **Physics Simulation**: Gravity, collision detection, time-based updates
- **Game Development**: State management, scoring, level design
- **User Interface**: Real-time feedback, visual design
- **Object-Oriented Programming**: Class design, object management

## Extension Ideas

After completing all sessions, try these modifications:

- Add sound effects or background music
- Implement different gesture types (peace sign, fist, etc.)
- Create power-ups or special objects
- Add multiplayer support
- Implement high score saving
- Create different game modes

## Troubleshooting

- **Camera not working**: Check camera permissions, try different camera index
- **Hand not detected**: Ensure good lighting, try adjusting detection confidence
- **Performance issues**: Reduce image resolution or processing frequency
- **Import errors**: Ensure all packages are installed with correct versions

## Files

- `session_01_webcam.py` - Basic camera display
- `session_02_hand_detection.py` - Hand landmark detection
- `session_03_interaction.py` - Region-based interaction
- `session_04_pinch_gesture.py` - Pinch gesture detection
- `session_05_object_pickup.py` - Simple object interaction
- `session_06_physics.py` - Physics simulation
- `session_07_multiple_objects.py` - Multiple object management
- `session_08_game_mechanics.py` - Complete game with scoring

Each file is self-contained and can be run independently, though later sessions build on concepts from earlier ones.