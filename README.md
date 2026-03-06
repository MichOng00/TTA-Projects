# TTA Projects

A repository of educational code for teaching AI and coding to school-aged students, mostly in Python. This collection spans foundational Python concepts through advanced machine learning applications in robotics. 

The curriculum is designed on the belief that interest drives motivation, which drives learning. Hence projects cover a broad range of AI applications to allow more students who may not think of themselves as "coders" or "interested in AI" to see how AI is applicable to the technologies they encounter every day. Students also complete concrete projects rather than just learning theory.

Disclaimer: some parts of the code, such as helper functions, comments, or docstrings, were generated using LLMs. LLM output is always checked and tested before use.

## 📚 Curriculum Overview

### Level 3: Python Fundamentals & Introduction to AI
- **L3T3 - Python Basics & UGOT**: Introduction to Python programming with beginner robotics control
- **L3T4 - Lists & UGOT CV**: More Python data structures and using built-in pre-trained CV models for robotics

### Level 4: Intermediate AI & Game Development
- **L4T1 - Chatbot**: Building basic conversational systems
- **L4T2 - Pygame**: Game development fundamentals with Python
- **L4T3 - Pygame with NEAT**: Introduction to neural networks via evolutionary AI for game-playing agents
- **L4T4 - Pytorch for CNNs**: Deep learning for image classification with convolutional neural networks

### Level 5: More AI Applications
- **L5T1 - OpenCV Basics**: Applied computer vision for robots using OpenCV and Mediapipe

## 🎯 Highlighted Projects

### L4T4 - Pytorch for CNNs

An introduction to deep learning with PyTorch, focusing on convolutional neural networks. This module teaches:
- Neural network architecture design
- Image classification with CNNs
- Training loops and optimization
- Model evaluation and validation

Students learn to build neural networks in Pytorch and understand the fundamentals of deep learning. They apply their knowledge to the classic MNIST and FashionMNIST datasets, as well as the Google QuickDraw dataset. They also integrate each trained model into a simple GUI desktop application using Tkinter, allowing them to qualitatively test model accuracy with their own drawings or uploaded images.

### L5T1 - OpenCV

Students learn how to use OpenCV and MediaPipe for real-time image processing and object detection. Content includes:
- Real-time video capture and frame processing
- Face / object detection, recognition, and tracking
- Drawing on images for visualization and debugging
- Interfacing computer vision with robotics hardware

This module bridges traditional computer vision with practical robotics applications. Students connect pre-trained models to hardware such as the UGOT robot or a drone, implementing face tracking and automated camera movement to follow a target.

### L5T2 - Object Detection

This code was initially developed as proof of concept while designing a [national competition](https://www.imda.gov.sg/activities/activities-catalogue/national-youth-tech-championship) for secondary school students (aged 13-17). Teams were to perform an end-to-end object detection pipeline to complete a robot navigation challenge. The [Jupyter notebook](https://github.com/MichOng00/TTA-Projects/blob/main/IMDA/NYTC_2026_demo_noVOC.ipynb) demonstrates and explains a simple workflow that students could use.

Subsequently, the competition format was modified and the code was adapted for use in the classroom.

**Pipeline Stages:**

1. **Data Collection**: Gathering images for dataset
2. **Annotation**: Using Roboflow to annotate images 
3. **Model Fine-tuning**: Fine-tuning YOLO base model on custom dataset
4. **Live Deployment**: Real-time object detection with robot control integration

**Learning Outcomes:**
- Understanding full ML workflow from raw data to deployed system
- Practical experience with data annotation and preparation
- Transfer learning and model fine-tuning
- Integrating computer vision with robotics systems

## 📝 Notes
- Archive folder contains experimental and legacy projects, and may be attributable to others
- Robot integration examples use UGOT robots and Tello drones
