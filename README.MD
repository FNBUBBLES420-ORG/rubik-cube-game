# `IN DEVELOPMENT`



---

# Rubik's Cube Game (Python + PyOpenGL)

This project is a **Rubik's Cube Game** built using **Python** with `PyOpenGL` / `ModernGL` for rendering, and `numpy` for cube logic. The goal is to create an interactive Rubik's Cube simulator where developers can contribute to different parts of the project.

---

## 🎯 Project Goals
- Build a 3D Rubik’s Cube that can be scrambled, rotated, and solved.
- Provide smooth animations and intuitive controls.
- Lay the foundation for adding solving algorithms and possibly a web/Unity port later.

---

## 🛠️ Installation

Make sure you have Python 3.11+ installed. Then install the dependencies:

```bash
pip install PyOpenGL PyOpenGL_accelerate numpy pygame PyOpenGL.GLUT
```

---

## 👥 Roles & Responsibilities

- `Let Me Know Which Role You Would Like`

### **Game Logic Developer**
- Represents cube state in code (arrays or matrices).
- Implements functions for rotations (U, D, L, R, F, B).
- Keeps track of solved/unsolved state.

### **3D Rendering Developer**
- Uses PyOpenGL/ModernGL to draw cubelets.
- Handles rotation animations (not just instant state change).
- Manages camera angles, zoom, and lighting.

### **Controls / Input Developer**
- Handles keyboard input (arrow keys for rotation, space to scramble).
- Implements mouse dragging to rotate the camera view.
- Adds shortcuts for cube moves (`R`, `U'`, etc.).

### **Scramble & Solver Developer (Optional)**
- Writes an algorithm to scramble cube randomly.
- Adds solving algorithms (layer-by-layer, CFOP, or others).

---

## 🚀 Milestones

### **Milestone 1**
1. Open a window with one cubelet (single block).
2. Expand to full **3×3 Rubik’s Cube**.
3. Enable keyboard input to rotate one face.

### **Milestone 2**
1. Implement scramble function for random shuffling.
2. Add smooth rotation animations for all faces.
3. Camera controls: allow dragging and zooming.

### **Milestone 3**
1. Add solving algorithm support (basic layer-by-layer).
2. Display move history and statistics.
3. Add visual indicators for solved/unsolved state.

### **Milestone 4**
1. Implement advanced solving algorithms (CFOP, beginner’s method, etc.).
2. Build a simple GUI menu for start/reset/scramble.
3. Optimize rendering and input handling for performance.


### **Milestone 5 (Future Expansion)**
1. Add multiplayer cube races or timed challenges.
2. Port to JavaScript (Three.js).
3. Add sound effects, themes, and accessibility options.

---

## 📌 Future Ideas
- Add cube scrambling.
- Implement solving algorithms.
- Multiplayer cube races.
- Port to JavaScript (Three.js) or Unity for cross-platform play.

---

## 🤝 Contributing
- Assign yourself a role above.
- Follow Python best practices and keep code modular.
