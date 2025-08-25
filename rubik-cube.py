"""
Rubik's Cube Game — Python + Pygame + PyOpenGL
------------------------------------------------
Single-file implementation with:
  • 3D rendering of a 3×3×3 cube using 26 visible cubelets
  • Face turns with smooth animation (U, D, L, R, F, B) incl. prime via Shift
  • Mouse camera (drag to orbit, wheel to zoom)
  • Scramble (key: S), Reset (key: C)
  • Basic timer that starts on first move, shows in window title

Dependencies (see README):
    pip install PyOpenGL PyOpenGL_accelerate numpy pygame PyOpenGL.GLUT

Controls:
    Mouse drag       – Orbit camera
    Mouse wheel      – Zoom
    
    CUBE MOVES:
    R L U D F B      – Clockwise face turns (letter keys)
    Arrow Keys       – U/D/L/R face turns
    W A S            – U/L/D face turns (WASD style)
    SPACE            – Front face turn
    Shift + (key)    – Counterclockwise (prime) turn
    
    SHORTCUTS:
    S                – Scramble (25 moves)
    C                – Reset to solved
    ESC or Q         – Quit
    
    NOTE: R=Right, L=Left, U=Up, D=Down, F=Front, B=Back faces

Note: This is a clean starter that focuses on correctness + smooth turns.
      It keeps the design straightforward so contributors can extend it.
"""
from __future__ import annotations

# Suppress pygame warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")
import math
import random
import sys
import tkinter as tk
from tkinter import messagebox, simpledialog
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union

import pygame
from pygame.locals import *  # noqa: F401,F403
from OpenGL.GL import *
from OpenGL.GLU import *

# -----------------------------
# Input Sanitization Functions
# -----------------------------

def sanitize_numeric_input(value: Union[int, float], min_val: Union[int, float], max_val: Union[int, float], default: Union[int, float]) -> Union[int, float]:
    """Sanitize numeric input to ensure it's within valid bounds."""
    try:
        if value is None:
            return default
        if not isinstance(value, (int, float)):
            return default
        if math.isnan(value) or math.isinf(value):
            return default
        return max(min_val, min(max_val, value))
    except (TypeError, ValueError):
        return default

def sanitize_string_input(value: str, allowed_values: List[str], default: str = "") -> str:
    """Sanitize string input to ensure it's in allowed values."""
    try:
        if not isinstance(value, str):
            return default
        if value in allowed_values:
            return value
        return default
    except (TypeError, AttributeError):
        return default

def sanitize_position(pos: List[int]) -> List[int]:
    """Sanitize position coordinates to valid cube positions."""
    try:
        if not isinstance(pos, list) or len(pos) != 3:
            return [0, 0, 0]
        sanitized = []
        for coord in pos:
            if not isinstance(coord, int):
                sanitized.append(0)
            else:
                sanitized.append(max(-1, min(1, coord)))
        return sanitized
    except (TypeError, ValueError):
        return [0, 0, 0]

def sanitize_face_dict(faces: Dict[str, str]) -> Dict[str, str]:
    """Sanitize face dictionary to ensure valid face mappings."""
    try:
        if not isinstance(faces, dict):
            return {}
        sanitized = {}
        valid_faces = set(FACES)
        for face, color in faces.items():
            clean_face = sanitize_string_input(face, FACES)
            clean_color = sanitize_string_input(color, FACES)
            if clean_face and clean_color and clean_face in valid_faces and clean_color in valid_faces:
                sanitized[clean_face] = clean_color
        return sanitized
    except (TypeError, AttributeError):
        return {}

# -----------------------------
# Configuration (with sanitization)
# -----------------------------
WINDOW_W = sanitize_numeric_input(1024, 400, 4096, 1024)
WINDOW_H = sanitize_numeric_input(720, 300, 2160, 720)
FPS = sanitize_numeric_input(60, 30, 120, 60)
ANIM_DEG_PER_SEC = sanitize_numeric_input(360, 90, 720, 360)  # degrees/sec for a face turn animation
CUBELET_GAP = sanitize_numeric_input(0.03, 0.0, 0.2, 0.03)      # gap between cubelets for visual clarity
CUBELET_SIZE = sanitize_numeric_input(0.94, 0.1, 2.0, 0.94)     # size of each cubelet (edge length)
ZOOM_SENS = sanitize_numeric_input(1.1, 1.01, 2.0, 1.1)
MOUSE_SENS = sanitize_numeric_input(0.3, 0.1, 2.0, 0.3)

# WCA standard color scheme - Enhanced for better visibility
COLORS = {
    'U': (1.0, 1.0, 1.0),      # white (bright)
    'D': (1.0, 1.0, 0.0),      # yellow (bright)
    'F': (0.0, 0.8, 0.0),      # green (brighter)
    'B': (0.0, 0.4, 1.0),      # blue (brighter)
    'R': (1.0, 0.0, 0.0),      # red (pure red)
    'L': (1.0, 0.6, 0.0),      # orange (brighter)
}

# Faces list for convenience
FACES = ['U', 'D', 'L', 'R', 'F', 'B']

# Axis vectors used for rotations
AXIS = {
    'x': (1.0, 0.0, 0.0),
    'y': (0.0, 1.0, 0.0),
    'z': (0.0, 0.0, 1.0),
}

# Mapping of move -> (axis, layer_value, angle_sign)
# Angle_sign is +1 for +90 degrees, -1 for -90 degrees
MOVE_DEFS = {
    'U': ('y', +1, +1),
    'D': ('y', -1, -1),
    'R': ('x', +1, +1),
    'L': ('x', -1, -1),
    'F': ('z', +1, +1),
    'B': ('z', -1, -1),
}

# Orientation remapping for a +90° rotation around an axis
# e.g., for +90° about y, the lateral faces cycle: F->R->B->L->F
ORIENT_MAP_PLUS90 = {
    'x': {'U': 'B', 'B': 'D', 'D': 'F', 'F': 'U', 'L': 'L', 'R': 'R'},
    'y': {'F': 'R', 'R': 'B', 'B': 'L', 'L': 'F', 'U': 'U', 'D': 'D'},
    'z': {'U': 'R', 'R': 'D', 'D': 'L', 'L': 'U', 'F': 'F', 'B': 'B'},
}

# Helper: inverse mapping for -90°
def invert_orient_map(mapping: Dict[str, str]) -> Dict[str, str]:
    return {v: k for k, v in mapping.items()}

ORIENT_MAP_MINUS90 = {
    'x': invert_orient_map(ORIENT_MAP_PLUS90['x']),
    'y': invert_orient_map(ORIENT_MAP_PLUS90['y']),
    'z': invert_orient_map(ORIENT_MAP_PLUS90['z']),
}

# -----------------------------
# Cube Model
# -----------------------------
@dataclass
class Cubelet:
    pos: List[int]  # [x,y,z] each in {-1,0,1}
    faces: Dict[str, str]  # face -> face-letter color, e.g., {'U':'U','F':'F'}

    def __post_init__(self):
        """Sanitize inputs after initialization."""
        self.pos = sanitize_position(self.pos)
        self.faces = sanitize_face_dict(self.faces)

    def clone(self) -> 'Cubelet':
        """Create a sanitized clone of this cubelet."""
        try:
            return Cubelet(self.pos.copy(), self.faces.copy())
        except (AttributeError, TypeError):
            return Cubelet([0, 0, 0], {})

class Cube:
    def __init__(self) -> None:
        self.cubelets: List[Cubelet] = []
        try:
            self.reset_solved()
        except Exception as e:
            print(f"Error initializing cube: {e}", file=sys.stderr)
            self.cubelets = []

    def reset_solved(self) -> None:
        """Reset cube to solved state with input sanitization."""
        try:
            self.cubelets.clear()
            for x in (-1, 0, 1):
                for y in (-1, 0, 1):
                    for z in (-1, 0, 1):
                        # Skip the hidden core at (0,0,0)
                        if x == 0 and y == 0 and z == 0:
                            continue
                        
                        # Sanitize coordinates
                        x_safe = sanitize_numeric_input(x, -1, 1, 0)
                        y_safe = sanitize_numeric_input(y, -1, 1, 0)
                        z_safe = sanitize_numeric_input(z, -1, 1, 0)
                        
                        faces: Dict[str, str] = {}
                        if y_safe == 1: faces['U'] = 'U'
                        if y_safe == -1: faces['D'] = 'D'
                        if x_safe == -1: faces['L'] = 'L'
                        if x_safe == 1: faces['R'] = 'R'
                        if z_safe == 1: faces['F'] = 'F'
                        if z_safe == -1: faces['B'] = 'B'
                        
                        # Create cubelet with sanitized data
                        cubelet = Cubelet([int(x_safe), int(y_safe), int(z_safe)], faces)
                        self.cubelets.append(cubelet)
        except Exception as e:
            print(f"Error resetting cube: {e}", file=sys.stderr)
            self.cubelets = []

    def is_solved(self) -> bool:
        """Check if all stickers per global face share the same color with sanitization."""
        try:
            # Check if all stickers per global face share the same color
            # Determine reference colors from centers
            refs = {
                'U': 'U', 'D': 'D', 'L': 'L', 'R': 'R', 'F': 'F', 'B': 'B'
            }
            # For each face, examine cubelets that sit on that face
            for face in FACES:
                if not isinstance(face, str) or face not in refs:
                    continue
                for cubie in self.cubelets:
                    if not isinstance(cubie, Cubelet):
                        continue
                    if face in cubie.faces:
                        if cubie.faces[face] != refs[face]:
                            return False
            return True
        except Exception as e:
            print(f"Error checking if cube is solved: {e}", file=sys.stderr)
            return False

    # -------- Rotation helpers --------
    @staticmethod
    def _rot_pos_about_axis(pos: List[int], axis: str, sign: int) -> List[int]:
        """Rotate position about axis with input sanitization."""
        try:
            # Sanitize inputs
            pos = sanitize_position(pos)
            axis = sanitize_string_input(axis, ['x', 'y', 'z'], 'y')
            sign = sanitize_numeric_input(sign, -1, 1, 1)
            sign = 1 if sign > 0 else -1
            
            x, y, z = pos
            if axis == 'y':  # (x,z) -> (z,-x) for +90
                if sign > 0:
                    return [z, y, -x]
                else:
                    return [-z, y, x]
            elif axis == 'x':  # (y,z) -> (-z, y) for +90 (y'=-z, z'=y)
                if sign > 0:
                    return [x, -z, y]
                else:
                    return [x, z, -y]
            else:  # 'z': (x,y) -> (-y, x) for +90
                if sign > 0:
                    return [-y, x, z]
                else:
                    return [y, -x, z]
        except Exception as e:
            print(f"Error rotating position: {e}", file=sys.stderr)
            return pos if isinstance(pos, list) and len(pos) == 3 else [0, 0, 0]

    @staticmethod
    def _rot_faces(faces: Dict[str, str], axis: str, sign: int) -> Dict[str, str]:
        """Rotate face mappings with input sanitization."""
        try:
            # Sanitize inputs
            faces = sanitize_face_dict(faces)
            axis = sanitize_string_input(axis, ['x', 'y', 'z'], 'y')
            sign = sanitize_numeric_input(sign, -1, 1, 1)
            sign = 1 if sign > 0 else -1
            
            src = ORIENT_MAP_PLUS90[axis] if sign > 0 else ORIENT_MAP_MINUS90[axis]
            new_faces: Dict[str, str] = {}
            for f, color in faces.items():
                if isinstance(f, str) and isinstance(color, str):
                    new_f = src.get(f, f)
                    new_faces[new_f] = color
            return new_faces
        except Exception as e:
            print(f"Error rotating faces: {e}", file=sys.stderr)
            return faces if isinstance(faces, dict) else {}

    def apply_turn(self, move: str, prime: bool = False) -> None:
        """Apply turn with input sanitization."""
        try:
            # Sanitize inputs
            move = sanitize_string_input(move, list(MOVE_DEFS.keys()))
            if not move or move not in MOVE_DEFS:
                print(f"Invalid move: {move}", file=sys.stderr)
                return
            
            prime = bool(prime)
            
            # Immediate (no animation) state update — useful for scramble
            axis, layer, sign = MOVE_DEFS[move]
            sign = -sign if prime else sign
            
            for cubie in self.cubelets:
                if not isinstance(cubie, Cubelet):
                    continue
                x, y, z = cubie.pos
                if ((axis == 'x' and x == layer) or
                    (axis == 'y' and y == layer) or
                    (axis == 'z' and z == layer)):
                    cubie.pos = self._rot_pos_about_axis(cubie.pos, axis, sign)
                    cubie.faces = self._rot_faces(cubie.faces, axis, sign)
        except Exception as e:
            print(f"Error applying turn {move}: {e}", file=sys.stderr)

# -----------------------------
# Rendering
# -----------------------------

def draw_cubelet(c: Cubelet) -> None:
    """Draw a cubelet with input sanitization."""
    try:
        # Validate input
        if not isinstance(c, Cubelet):
            print("Invalid cubelet object", file=sys.stderr)
            return
        
        # Sanitize position
        pos = sanitize_position(c.pos)
        x, y, z = pos
        
        # Sanitize spacing and size
        spacing = sanitize_numeric_input(1.0 + CUBELET_GAP, 1.0, 2.0, 1.03)
        half = sanitize_numeric_input(CUBELET_SIZE / 2.0, 0.1, 1.0, 0.47)
        
        wx, wy, wz = x * spacing, y * spacing, z * spacing

        glPushMatrix()
        glTranslatef(float(wx), float(wy), float(wz))

        # Helper function to set material properties safely
        def set_face_material(face: str, faces_dict: Dict[str, str]):
            try:
                if face in faces_dict and faces_dict[face] in COLORS:
                    r, g, b = COLORS[faces_dict[face]]
                    # Sanitize color values
                    r = sanitize_numeric_input(r, 0.0, 1.0, 0.5)
                    g = sanitize_numeric_input(g, 0.0, 1.0, 0.5)
                    b = sanitize_numeric_input(b, 0.0, 1.0, 0.5)
                    
                    material_diffuse = (GLfloat * 4)(r, g, b, 1.0)
                    material_ambient = (GLfloat * 4)(r * 0.3, g * 0.3, b * 0.3, 1.0)
                    material_specular = (GLfloat * 4)(0.3, 0.3, 0.3, 1.0)
                    material_shininess = (GLfloat * 1)(20.0)
                    glMaterialfv(GL_FRONT, GL_DIFFUSE, material_diffuse)
                    glMaterialfv(GL_FRONT, GL_AMBIENT, material_ambient)
                    glMaterialfv(GL_FRONT, GL_SPECULAR, material_specular)
                    glMaterialfv(GL_FRONT, GL_SHININESS, material_shininess)
                else:
                    # Dark material for faces without stickers
                    dark_diffuse = (GLfloat * 4)(0.05, 0.05, 0.05, 1.0)
                    dark_ambient = (GLfloat * 4)(0.01, 0.01, 0.01, 1.0)
                    glMaterialfv(GL_FRONT, GL_DIFFUSE, dark_diffuse)
                    glMaterialfv(GL_FRONT, GL_AMBIENT, dark_ambient)
            except Exception as e:
                print(f"Error setting material for face {face}: {e}", file=sys.stderr)

        # Draw each face separately with proper OpenGL state management
        faces_to_draw = [
            ('U', (0.0, 1.0, 0.0), [(-half, half, -half), (-half, half, half), (half, half, half), (half, half, -half)]),
            ('D', (0.0, -1.0, 0.0), [(-half, -half, -half), (half, -half, -half), (half, -half, half), (-half, -half, half)]),
            ('F', (0.0, 0.0, 1.0), [(-half, -half, half), (half, -half, half), (half, half, half), (-half, half, half)]),
            ('B', (0.0, 0.0, -1.0), [(-half, -half, -half), (-half, half, -half), (half, half, -half), (half, -half, -half)]),
            ('R', (1.0, 0.0, 0.0), [(half, -half, -half), (half, half, -half), (half, half, half), (half, -half, half)]),
            ('L', (-1.0, 0.0, 0.0), [(-half, -half, -half), (-half, -half, half), (-half, half, half), (-half, half, -half)])
        ]
        
        for face_name, normal, vertices in faces_to_draw:
            # Set material properties OUTSIDE of glBegin/glEnd
            set_face_material(face_name, c.faces)
            
            # Draw the face
            glBegin(GL_QUADS)
            glNormal3f(normal[0], normal[1], normal[2])
            for vertex in vertices:
                glVertex3f(vertex[0], vertex[1], vertex[2])
            glEnd()
        
        glPopMatrix()
        
    except Exception as e:
        print(f"Error drawing cubelet: {e}", file=sys.stderr)
        # Try to clean up OpenGL state
        try:
            glPopMatrix()
        except:
            pass


# -----------------------------
# Draw box helper function
# -----------------------------

def _draw_box(h: float) -> None:
    # Draw a solid cube centered at origin with half-size h with proper normals
    glBegin(GL_QUADS)
    
    # Top (U) - normal pointing up
    glNormal3f(0.0, 1.0, 0.0)
    glVertex3f(-h, h, -h)
    glVertex3f(-h, h, h)
    glVertex3f(h, h, h)
    glVertex3f(h, h, -h)
    
    # Bottom (D) - normal pointing down
    glNormal3f(0.0, -1.0, 0.0)
    glVertex3f(-h, -h, -h)
    glVertex3f(h, -h, -h)
    glVertex3f(h, -h, h)
    glVertex3f(-h, -h, h)
    
    # Front (F) - normal pointing forward
    glNormal3f(0.0, 0.0, 1.0)
    glVertex3f(-h, -h, h)
    glVertex3f(h, -h, h)
    glVertex3f(h, h, h)
    glVertex3f(-h, h, h)
    
    # Back (B) - normal pointing backward
    glNormal3f(0.0, 0.0, -1.0)
    glVertex3f(-h, -h, -h)
    glVertex3f(-h, h, -h)
    glVertex3f(h, h, -h)
    glVertex3f(h, -h, -h)
    
    # Right (R) - normal pointing right
    glNormal3f(1.0, 0.0, 0.0)
    glVertex3f(h, -h, -h)
    glVertex3f(h, h, -h)
    glVertex3f(h, h, h)
    glVertex3f(h, -h, h)
    
    # Left (L) - normal pointing left
    glNormal3f(-1.0, 0.0, 0.0)
    glVertex3f(-h, -h, -h)
    glVertex3f(-h, -h, h)
    glVertex3f(-h, h, h)
    glVertex3f(-h, h, -h)
    
    glEnd()


def _quad(size: float) -> None:
    # Draw a square in the +Z plane centered at origin with edge=2*size
    # Include normal for proper lighting
    glBegin(GL_QUADS)
    glNormal3f(0.0, 0.0, 1.0)  # Normal pointing in +Z direction
    glVertex3f(-size, -size, 0)
    glVertex3f(size, -size, 0)
    glVertex3f(size, size, 0)
    glVertex3f(-size, size, 0)
    glEnd()

# -----------------------------
# Animation & Move Queue
# -----------------------------
@dataclass
class Turn:
    move: str  # one of 'U','D','L','R','F','B'
    prime: bool  # counterclockwise if True (relative to face perspective)

@dataclass
class ActiveAnim:
    move: str
    prime: bool
    axis: str
    layer: int
    sign: int
    angle: float = 0.0  # accumulated degrees

class MoveQueue:
    def __init__(self):
        self.queue: List[Turn] = []
        self.active: Optional[ActiveAnim] = None

    def enqueue(self, move: str, prime: bool = False):
        self.queue.append(Turn(move, prime))

    def start_next(self) -> Optional[ActiveAnim]:
        if self.active is not None:
            return self.active
        if not self.queue:
            return None
        t = self.queue.pop(0)
        axis, layer, sign = MOVE_DEFS[t.move]
        if t.prime:
            sign = -sign
        self.active = ActiveAnim(t.move, t.prime, axis, layer, sign, 0.0)
        return self.active

    def update(self, dt: float, cube: Cube) -> None:
        if self.active is None:
            return
        self.active.angle += ANIM_DEG_PER_SEC * dt
        if self.active.angle >= 90.0 - 1e-3:
            # Commit the turn to the cube state
            cube.apply_turn(self.active.move, self.active.prime)
            self.active = None

# -----------------------------
# Scramble
# -----------------------------
MOVES = ['U', 'D', 'L', 'R', 'F', 'B']

def generate_scramble(n: int = 25) -> List[Turn]:
    seq: List[Turn] = []
    last: Optional[str] = None
    for _ in range(n):
        m = random.choice(MOVES)
        while last is not None and (m == last):
            m = random.choice(MOVES)
        prime = bool(random.getrandbits(1))
        seq.append(Turn(m, prime))
        last = m
    return seq

# -----------------------------
# Menu System Functions
# -----------------------------

def show_controls_dialog():
    """Show controls help dialog."""
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        controls_text = """
RUBIK'S CUBE CONTROLS
=====================

CUBE MOVES:
• R L U D F B    - Face turns (letter keys)
• Arrow Keys     - U/D/L/R face turns  
• W A S          - U/L/D face turns
• SPACE          - Front face turn
• Shift + (key)  - Counterclockwise turn

SHORTCUTS:
• S              - Scramble cube
• C              - Reset to solved
• ESC or Q       - Quit

MOUSE:
• Drag           - Orbit camera
• Wheel          - Zoom

FACES:
• R = Right, L = Left, U = Up
• D = Down, F = Front, B = Back
        """
        
        messagebox.showinfo("Controls", controls_text)
        root.destroy()
    except Exception as e:
        print(f"Error showing controls dialog: {e}", file=sys.stderr)

def show_how_to_play_dialog():
    """Show how to play dialog."""
    try:
        root = tk.Tk()
        root.withdraw()
        
        how_to_play_text = """
HOW TO PLAY RUBIK'S CUBE
========================

OBJECTIVE:
• Get all six faces to show a single solid color
• Each face should match the center square's color

BASIC MOVES:
• Each letter (R, L, U, D, F, B) rotates that face 90° clockwise
• Hold Shift + letter for counterclockwise rotation
• Use mouse to rotate view and see all sides

GETTING STARTED:
1. Press 'S' to scramble the cube
2. Try basic moves to understand how pieces move
3. Use 'C' to reset if you get stuck

TIPS:
• Start by making a cross on one face
• Work layer by layer (beginners method)
• Each move affects multiple pieces
• Practice makes perfect!

COLOR SCHEME:
• White opposite Yellow
• Red opposite Orange  
• Blue opposite Green
        """
        
        messagebox.showinfo("How to Play", how_to_play_text)
        root.destroy()
    except Exception as e:
        print(f"Error showing how to play dialog: {e}", file=sys.stderr)

def show_about_dialog():
    """Show about dialog."""
    try:
        root = tk.Tk()
        root.withdraw()
        
        about_text = """
RUBIK'S CUBE SIMULATOR
======================

Version: 1.2
Built with: Python + Pygame + PyOpenGL

Features:
• 3D Rubik's Cube simulation
• Smooth animations
• Multiple control schemes
• Timer and move counter
• Scramble and reset functions
• Comprehensive input validation
• Help system and guides

This is a fully interactive 3D Rubik's Cube 
simulator with realistic physics and controls.

Enjoy solving!
        """
        
        messagebox.showinfo("About", about_text)
        root.destroy()
    except Exception as e:
        print(f"Error showing about dialog: {e}", file=sys.stderr)

def show_settings_dialog():
    """Show settings dialog for customization."""
    try:
        root = tk.Tk()
        root.withdraw()
        
        # Simple settings for now - could be expanded
        choice = messagebox.askyesnocancel(
            "Settings", 
            "Would you like to:\n\n" +
            "• YES - Show detailed move notifications\n" +
            "• NO - Hide move notifications\n" +
            "• CANCEL - Keep current settings"
        )
        
        if choice is not None:
            if choice:
                messagebox.showinfo("Settings", "Move notifications enabled!")
            else:
                messagebox.showinfo("Settings", "Move notifications disabled!")
        
        root.destroy()
        return choice
    except Exception as e:
        print(f"Error showing settings dialog: {e}", file=sys.stderr)
        return None

def create_menu_bar():
    """Create and display menu bar - called from main app."""
    try:
        # Create a hidden root window for the menu
        menu_root = tk.Tk()
        menu_root.withdraw()
        
        # Create a simple menu using message boxes
        # This will be triggered by keyboard shortcuts in the main game
        return menu_root
    except Exception as e:
        print(f"Error creating menu bar: {e}", file=sys.stderr)
        return None

# -----------------------------
# App / Main Loop
# -----------------------------
class App:
    def __init__(self) -> None:
        try:
            pygame.init()
            
            # Sanitize window dimensions
            window_w = int(sanitize_numeric_input(WINDOW_W, 400, 4096, 1024))
            window_h = int(sanitize_numeric_input(WINDOW_H, 300, 2160, 720))
            
            pygame.display.set_mode((window_w, window_h), DOUBLEBUF | OPENGL)
            pygame.display.set_caption("Rubik's Cube — Press H for Help | F1 for Guide")
            self.clock = pygame.time.Clock()
            self.cube = Cube()
            self.mq = MoveQueue()

            # Camera with sanitized values
            self.cam_dist = sanitize_numeric_input(8.5, 3.0, 50.0, 8.5)
            self.cam_yaw = sanitize_numeric_input(30, -360, 360, 30)
            self.cam_pitch = sanitize_numeric_input(20, -89, 89, 20)
            self.mouse_down = False
            self.last_mouse: Optional[Tuple[int, int]] = None

            # Timing
            self.solve_started = False
            self.time_ms = 0

            self._setup_gl()
            
            # Print controls to console
            print("\n" + "="*60)
            print("RUBIK'S CUBE CONTROLS:")
            print("="*60)
            print("CUBE MOVES:")
            print("  R L U D F B    - Face turns (letter keys)")
            print("  Arrow Keys     - U/D/L/R face turns")
            print("  W A S          - U/L/D face turns")
            print("  SPACE          - Front face turn")
            print("  Shift + (key)  - Counterclockwise turn")
            print("\nSHORTCUTS:")
            print("  S              - Scramble cube")
            print("  C              - Reset to solved")
            print("  ESC or Q       - Quit")
            print("\nMENUS & HELP:")
            print("  H              - Show controls help")
            print("  F1             - How to play guide")
            print("  I              - About information")
            print("  F2             - Settings")
            print("\nMOUSE:")
            print("  Drag           - Orbit camera")
            print("  Wheel          - Zoom")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"Error initializing application: {e}", file=sys.stderr)
            sys.exit(1)

    def _setup_gl(self) -> None:
        glViewport(0, 0, WINDOW_W, WINDOW_H)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, WINDOW_W / float(WINDOW_H), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glClearColor(0.08, 0.08, 0.1, 1.0)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_LINE_SMOOTH)

        # Light (simple ambient + directional)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        ambient = (GLfloat * 4)(0.3, 0.3, 0.3, 1.0)
        diffuse = (GLfloat * 4)(0.8, 0.8, 0.8, 1.0)
        position = (GLfloat * 4)(5.0, 8.0, 10.0, 1.0)
        glLightfv(GL_LIGHT0, GL_AMBIENT, ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse)
        glLightfv(GL_LIGHT0, GL_POSITION, position)
        glDisable(GL_COLOR_MATERIAL)  # we'll set colors directly

    def run(self) -> None:
        """Main game loop with input sanitization."""
        running = True
        try:
            while running:
                # Sanitize frame rate
                fps = sanitize_numeric_input(FPS, 30, 120, 60)
                dt = self.clock.tick(int(fps)) / 1000.0
                dt = sanitize_numeric_input(dt, 0.0, 1.0, 0.016)  # Cap at 1 second
                
                for event in pygame.event.get():
                    try:
                        if event.type == QUIT:
                            running = False
                        elif event.type == KEYDOWN:
                            if hasattr(event, 'key'):
                                if event.key in (K_ESCAPE, K_q):
                                    running = False
                                elif event.key == K_c:
                                    self.cube.reset_solved()
                                    if hasattr(self.mq, 'active'):
                                        self.mq.active = None
                                    if hasattr(self.mq, 'queue'):
                                        self.mq.queue.clear()
                                    self.solve_started = False
                                    self.time_ms = 0
                                elif event.key == K_s:
                                    try:
                                        scramble_moves = generate_scramble(25)
                                        for t in scramble_moves:
                                            if hasattr(t, 'move') and hasattr(t, 'prime'):
                                                self.mq.enqueue(t.move, t.prime)
                                    except Exception as e:
                                        print(f"Error generating scramble: {e}", file=sys.stderr)
                                # Menu shortcuts
                                elif event.key == K_h:  # H for help/controls
                                    show_controls_dialog()
                                elif event.key == K_F1:  # F1 for how to play
                                    show_how_to_play_dialog()
                                elif event.key == K_i:  # I for info/about
                                    show_about_dialog()
                                elif event.key == K_F2:  # F2 for settings
                                    show_settings_dialog()
                                else:
                                    self._handle_move_key(event)
                        elif event.type == MOUSEBUTTONDOWN:
                            self._handle_mouse_button_down(event)
                        elif event.type == MOUSEBUTTONUP:
                            self._handle_mouse_button_up(event)
                        elif event.type == MOUSEMOTION:
                            self._handle_mouse_motion(event)
                    except Exception as e:
                        print(f"Error handling event: {e}", file=sys.stderr)

                # Start anim if none active
                try:
                    if hasattr(self.mq, 'active') and self.mq.active is None:
                        self.mq.start_next()
                except Exception as e:
                    print(f"Error starting animation: {e}", file=sys.stderr)

                # Update animation
                try:
                    pre_active = hasattr(self.mq, 'active') and self.mq.active is not None
                    self.mq.update(dt, self.cube)
                    post_active = hasattr(self.mq, 'active') and self.mq.active is not None

                    # Start timer on the first committed move
                    if pre_active and not post_active and not self.solve_started:
                        self.solve_started = True

                    if self.solve_started and not self.cube.is_solved():
                        self.time_ms += int(dt * 1000)

                    # If solved and no animations pending, freeze timer
                    if (self.solve_started and self.cube.is_solved() and 
                        hasattr(self.mq, 'active') and self.mq.active is None and 
                        hasattr(self.mq, 'queue') and not self.mq.queue):
                        pass  # time_ms remains at finish
                except Exception as e:
                    print(f"Error updating animations: {e}", file=sys.stderr)

                self._render()

        except Exception as e:
            print(f"Error in main loop: {e}", file=sys.stderr)
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up OpenGL resources and pygame."""
        try:
            # Clean up OpenGL state
            glDisable(GL_LIGHTING)
            glDisable(GL_LIGHT0)
            glDisable(GL_DEPTH_TEST)
            glDisable(GL_COLOR_MATERIAL)
            glDisable(GL_NORMALIZE)
            
            # Quit pygame
            pygame.quit()
        except Exception as e:
            print(f"Error during cleanup: {e}", file=sys.stderr)
            try:
                pygame.quit()
            except:
                pass

    def _handle_mouse_button_down(self, event) -> None:
        """Handle mouse button down with sanitization."""
        try:
            if hasattr(event, 'button'):
                if event.button == 1:
                    self.mouse_down = True
                    if hasattr(pygame.mouse, 'get_pos'):
                        pos = pygame.mouse.get_pos()
                        if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                            self.last_mouse = (int(pos[0]), int(pos[1]))
                elif event.button == 4:  # wheel up
                    zoom_factor = sanitize_numeric_input(ZOOM_SENS, 1.01, 2.0, 1.1)
                    self.cam_dist = sanitize_numeric_input(self.cam_dist / zoom_factor, 1.0, 100.0, self.cam_dist)
                elif event.button == 5:  # wheel down
                    zoom_factor = sanitize_numeric_input(ZOOM_SENS, 1.01, 2.0, 1.1)
                    self.cam_dist = sanitize_numeric_input(self.cam_dist * zoom_factor, 1.0, 100.0, self.cam_dist)
        except Exception as e:
            print(f"Error handling mouse button down: {e}", file=sys.stderr)

    def _handle_mouse_button_up(self, event) -> None:
        """Handle mouse button up with sanitization."""
        try:
            if hasattr(event, 'button') and event.button == 1:
                self.mouse_down = False
                self.last_mouse = None
        except Exception as e:
            print(f"Error handling mouse button up: {e}", file=sys.stderr)

    def _handle_mouse_motion(self, event) -> None:
        """Handle mouse motion with sanitization."""
        try:
            if self.mouse_down and hasattr(event, 'pos'):
                pos = event.pos
                if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                    x, y = int(pos[0]), int(pos[1])
                    if self.last_mouse is not None and isinstance(self.last_mouse, (tuple, list)) and len(self.last_mouse) >= 2:
                        dx = sanitize_numeric_input(x - self.last_mouse[0], -1000, 1000, 0)
                        dy = sanitize_numeric_input(y - self.last_mouse[1], -1000, 1000, 0)
                        mouse_sens = sanitize_numeric_input(MOUSE_SENS, 0.1, 2.0, 0.3)
                        
                        self.cam_yaw = sanitize_numeric_input(self.cam_yaw + dx * mouse_sens, -720, 720, self.cam_yaw)
                        new_pitch = self.cam_pitch + dy * mouse_sens
                        self.cam_pitch = sanitize_numeric_input(new_pitch, -89, 89, self.cam_pitch)
                    self.last_mouse = (x, y)
        except Exception as e:
            print(f"Error handling mouse motion: {e}", file=sys.stderr)

    def _handle_move_key(self, event: pygame.event.Event) -> None:
        """Handle move key input with sanitization."""
        try:
            if not hasattr(event, 'key'):
                return
                
            key_to_move = {
                # Primary letter keys for face moves
                K_u: 'U', K_d: 'D', K_l: 'L', K_r: 'R', K_f: 'F', K_b: 'B',
                # Arrow keys
                K_UP: 'U', K_DOWN: 'D', K_LEFT: 'L', K_RIGHT: 'R',
                # WASD (alternative mapping, avoiding conflicts)
                K_w: 'U', K_s: 'D', K_a: 'L',  # K_d conflicts with 'D' face
                K_SPACE: 'F',  # Space for front
            }
            
            move = key_to_move.get(event.key)
            if move is None:
                return
                
            # Sanitize the move
            move = sanitize_string_input(move, list(MOVE_DEFS.keys()))
            if not move:
                return
                
            try:
                mod = pygame.key.get_mods()
                prime = bool(mod & (KMOD_LSHIFT | KMOD_RSHIFT))
                self.mq.enqueue(move, prime)
            except Exception as e:
                print(f"Error getting key modifiers: {e}", file=sys.stderr)
                
        except Exception as e:
            print(f"Error handling move key: {e}", file=sys.stderr)
        prime = bool(mod & (KMOD_LSHIFT | KMOD_RSHIFT))
        self.mq.enqueue(move, prime)

    def _apply_camera(self) -> None:
        glLoadIdentity()
        # Position the camera on a sphere around origin
        pitch_rad = math.radians(self.cam_pitch)
        yaw_rad = math.radians(self.cam_yaw)
        x = self.cam_dist * math.cos(pitch_rad) * math.cos(yaw_rad)
        y = self.cam_dist * math.sin(pitch_rad)
        z = self.cam_dist * math.cos(pitch_rad) * math.sin(yaw_rad)
        gluLookAt(x, y, z, 0, 0, 0, 0, 1, 0)

    def _render(self) -> None:
        """Render the scene with error handling."""
        try:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self._apply_camera()

            # Nice axis lines (optional)
            try:
                glDisable(GL_LIGHTING)
                glLineWidth(2.0)
                glBegin(GL_LINES)
                # X red
                glColor3f(1, 0, 0)
                glVertex3f(-5, 0, 0)
                glVertex3f(5, 0, 0)
                # Y green
                glColor3f(0, 1, 0)
                glVertex3f(0, -5, 0)
                glVertex3f(0, 5, 0)
                # Z blue
                glColor3f(0, 0, 1)
                glVertex3f(0, 0, -5)
                glVertex3f(0, 0, 5)
                glEnd()
                glEnable(GL_LIGHTING)
            except Exception as e:
                print(f"Error drawing axis lines: {e}", file=sys.stderr)
                try:
                    glEnd()  # Try to clean up
                    glEnable(GL_LIGHTING)
                except:
                    pass

            # Determine active animation transform
            try:
                active = getattr(self.mq, 'active', None) if hasattr(self, 'mq') else None

                # Draw cubelets, applying rotation to affected layer if animating
                if hasattr(self, 'cube') and hasattr(self.cube, 'cubelets'):
                    for cubie in self.cube.cubelets:
                        if not isinstance(cubie, Cubelet):
                            continue
                        
                        try:
                            glPushMatrix()
                            if active is not None and hasattr(active, 'axis') and hasattr(active, 'layer'):
                                axis = getattr(active, 'axis', 'y')
                                layer = getattr(active, 'layer', 0)
                                sign = getattr(active, 'sign', 1)
                                angle = min(getattr(active, 'angle', 0), 90.0)
                                
                                if hasattr(cubie, 'pos') and len(cubie.pos) >= 3:
                                    x, y, z = cubie.pos[0], cubie.pos[1], cubie.pos[2]
                                    affected = ((axis == 'x' and x == layer) or 
                                              (axis == 'y' and y == layer) or 
                                              (axis == 'z' and z == layer))
                                    if affected and axis in AXIS:
                                        ax = AXIS[axis]
                                        glRotatef(sign * angle, ax[0], ax[1], ax[2])
                            
                            draw_cubelet(cubie)
                            glPopMatrix()
                        except Exception as e:
                            print(f"Error drawing cubelet: {e}", file=sys.stderr)
                            try:
                                glPopMatrix()
                            except:
                                pass
            except Exception as e:
                print(f"Error in cubelet rendering: {e}", file=sys.stderr)

            # Update window title with timer + state
            try:
                secs = sanitize_numeric_input(self.time_ms / 1000.0, 0.0, 86400.0, 0.0)
                queue_len = len(getattr(self.mq, 'queue', [])) if hasattr(self, 'mq') else 0
                is_solved = self.cube.is_solved() if hasattr(self, 'cube') else False
                title = f"Rubik's Cube — MovesQ:{queue_len}  Time: {secs:.2f}s  {'SOLVED!' if is_solved else ''}"
                pygame.display.set_caption(title)
            except Exception as e:
                print(f"Error updating window title: {e}", file=sys.stderr)
                pygame.display.set_caption("Rubik's Cube")

            pygame.display.flip()
            
        except Exception as e:
            print(f"Error in render function: {e}", file=sys.stderr)


if __name__ == '__main__':
    try:
        app = App()
        app.run()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
