# 🧩 Rubik's Cube 3D Game

![Version](https://img.shields.io/badge/version-1.2-blue.svg)
![Python](https://img.shields.io/badge/python-3.11.9-blue.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)

🎮 **Play a virtual 3D Rubik's Cube on your computer!** 

This game gives you a realistic 3D Rubik's Cube that you can rotate, scramble, and solve using your keyboard and mouse. Perfect for practicing your solving skills or learning how to solve a Rubik's Cube!

## ✨ What You Can Do

- 🎯 **Practice Solving**: Scramble and solve the cube as many times as you want
- 🖱️ **3D View**: Rotate the camera to see all sides of the cube
- ⏱️ **Timer**: Track how long it takes you to solve
- 🎮 **Easy Controls**: Use simple keyboard keys to turn the cube faces
- 📚 **Built-in Help**: Press H for controls, F1 for solving tips
- 🔄 **Instant Reset**: Press C to reset to a solved state anytime

## 🚀 Quick Start (Easy Way!)

### Step 1: Install Python
**You need Python 3.11.9 first** - Choose your version:

**Windows Users:**
- 💻 [Download for 64-bit Windows](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe) (Most common)
- 💻 [Download for 32-bit Windows](https://www.python.org/ftp/python/3.11.9/python-3.11.9.exe)

**Mac Users:**
- 🍎 [Download for Mac](https://www.python.org/ftp/python/3.11.9/python-3.11.9-macos11.pkg)

**Important:** When installing Python, make sure to check "Add Python to PATH"!

### Step 2: Get the Game
1. Click the green **"Code"** button at the top of this page
2. Select **"Download ZIP"**
3. Save it to your Desktop
4. Right-click the ZIP file and select "Extract All"
5. Double-click the extracted folder to open it

### Step 3: Install Game Requirements
1. Open Command Prompt (Windows) or Terminal (Mac)
2. Type this command and press Enter:
   ```
   pip install pygame==2.6.1 PyOpenGL==3.1.10 PyOpenGL_accelerate==3.1.10 numpy==2.2.6
   ```

### Step 4: Play!
Double-click the file called **`rubix-cube.py`** and start playing!

## 🎮 How to Play

### Basic Controls
- **R L U D F B** = Turn cube faces (Right, Left, Up, Down, Front, Back)
- **Arrow Keys** = Also turn faces (easier for beginners)
- **SPACE** = Turn the front face
- **Hold Shift + any key** = Turn the opposite direction

### Quick Actions
- **S** = Scramble the cube (mix it up randomly)
- **C** = Reset to solved (if you get stuck)
- **H** = Show help
- **F1** = Learn how to solve
- **ESC** = Quit game

### Mouse Controls
- **Drag** = Rotate your view around the cube
- **Mouse Wheel** = Zoom in and out

## 🔧 System Requirements

**Your Computer Should Have:**
- **Windows 10/11** (64-bit or 32-bit), **macOS 12.0+**, or **Linux**
- **Python 3.11.9** (Required - see installation steps above)
- **At least 2GB RAM** (4GB recommended for smooth performance)
- **OpenGL 2.1+ compatible graphics card** (Most computers from 2010+ work fine)
- **200MB free disk space** (including Python dependencies)
- **Internet connection** (for initial setup and Python package downloads)

## 🆘 Having Problems?

### Game Won't Start?
1. Make sure you installed Python 3.11.9 with "Add to PATH" checked
2. Try running this in Command Prompt: `pip install pygame PyOpenGL numpy`
3. Make sure you extracted the ZIP file completely

### Game Runs Slowly?
- Update your graphics drivers
- Close other programs to free up memory
- Install PyOpenGL_accelerate for better performance

### Need Help Learning to Solve?
- Press **F1** in the game for solving tips
- Start with making a white cross on one face
- Work one layer at a time
- Use online tutorials for detailed steps

## 📁 What's Included

```
📁 rubix-cube/
├── 🐍 rubix-cube.py      # The game (double-click to play!)
├── 📖 README.md          # This guide
├── 🔒 SECURITY.md        # Security information
├── 📄 LICENSE            # Legal stuff
└── 📋 requirements.txt   # Technical requirements
```

## 🤝 Want to Help Improve the Game?

We'd love your help! You can:
- 🐛 Report bugs or problems
- 💡 Suggest new features  
- 🔧 Fix issues if you know programming
- 📝 Improve the documentation

Just open an "Issue" on this GitHub page to get started!

## 📈 Version History

- **v1.2** ✨ Added help system, better error handling, improved stability
- **v1.1** 🎨 Enhanced colors, fixed display issues  
- **v1.0** 🚀 First version with basic 3D cube

## 📜 Legal Stuff

This game is free and open source under the Apache 2.0 License. 
You can use it, modify it, and share it freely!

Full license details: http://www.apache.org/licenses/LICENSE-2.0

## 💝 Credits

Made with ❤️ for cube enthusiasts everywhere!

Special thanks to:
- The Pygame community for graphics support
- The PyOpenGL team for 3D rendering
- Rubik's Cube solvers worldwide for inspiration

---

🧩 **Happy Cubing!** 🧩
