Robotic Chess Player – User Guide

---

> **Host OS tested:** Ubuntu 22.04.5 LTS (x86‑64)

This repository contains everything you need to reproduce a tabletop **robotic chess player** built around a **Niryo Ned 2** arm and an **Intel RealSense D455** camera.  The pipeline recognises an ArUCo‑tagged 8 × 8 chessboard, translates natural‑language or UCI moves into safe trajectories via an LLM, and executes them on the robot.

---

## 1 Hardware & physical assumptions

| Component   | Required spec                                                          |
| ----------- | ---------------------------------------------------------------------- |
| Robotic arm | Niryo Ned 2 (Ethernet control)                                         |
| Camera      | Intel RealSense D455 (RGB 1280 × 720 @ 30 fps)                         |
| Chessboard  | 8 × 8 board, **40 mm square width** (≈ 320 mm × 320 mm)                |
| Markers     | ArUCo IDs 0‑3 at the four corners (TL, TR, BR, BL); ID 4 on robot base. Type DICT_4X4_50; Side length 2.8cm. |

Piece heights used in path planning live in `robot_executor.py` and can be tuned for different chess sets.

---

## 2 Repository layout

```
realsense_aruco_pose.py   ← camera calibration & board mapping (vision)
main_controller.py        ← match loop orchestrator (robot‑side)
motion_planner.py         ← LLM → robot‑command translator
robot_executor.py         ← low‑level motion execution on Ned 2
llm_player.py             ← optional LLM chess opponent
chessboard_table.py       ← generated XY board positions
requirements_cal.txt      ← pinned deps for venv **cal**
requirements_chess.txt    ← pinned deps for venv **chess**
commands/                 ← auto‑saved motion scripts
```

---

## 3 Why two virtual environments?

OpenCV + RealSense often conflict with packages required by the Niryo SDK and OpenAI client.  To keep everything stable we isolate them:

| venv            | Purpose                   | Activates which scripts?                                                        |
| --------------- | ------------------------- | ------------------------------------------------------------------------------- |
| **venv\_cal**   | Vision & calibration      | `realsense_aruco_pose.py`                                                       |
| **venv\_chess** | LLM, Stockfish, Niryo SDK | `main_controller.py`, `motion_planner.py`, `robot_executor.py`, `llm_player.py` |

### 3.1 Create & install

```bash
# Vision
python3 -m venv venv_cal
source venv_cal/bin/activate
pip install -r requirements_cal.txt   # uses pinned versions

# Robotics / LLM
python3 -m venv venv_chess
source venv_chess/bin/activate
pip install -r requirements_chess.txt
```


---

## 4 Additional system requirements

* **OpenAI API key** – export before running any LLM‑driven script:

  ```bash
  export OPENAI_API_KEY="sk‑..."
  ```
* **Stockfish** engine installed at `/usr/games/stockfish` (edit path in `main_controller.py` if different):

  ```bash
  sudo apt install stockfish
  ```
* **udev rules** for RealSense cameras (if not already present):

  ```bash
  sudo cp librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/
  sudo udevadm control --reload-rules
  ```

---

## 5 Step‑by‑step workflow

### 5.1 Calibrate camera → board mapping (one‑off per setup)

1. Mount D455 so it sees the chessboard and robot base marker.
2. Activate **venv\_cal**:

   ```bash
   source venv_cal/bin/activate
   ```
3. Run calibration (default dictionary `DICT_4X4_50`, marker side = 28 mm):

   ```bash
   python realsense_aruco_pose.py --marker_len 0.028
   ```
4. When all five markers are detected and the red grid looks centred, press **`c`**.  The script will:

   * Print calibrated XY positions (metres),
   * Save `grid_robot_xy.npy` & `grid_robot_xy_homog.npy`,
   * Store an annotated screenshot in `captures/` (if recording enabled).
5. Press **`q`** to quit.
6. Commit the two `.npy` files — they are loaded by `chessboard_table.py` at runtime.

### 5.2 Play a game

1. Home the Niryo arm and ensure the gripper is attached.
2. Activate **venv\_chess** and export your OpenAI key.

   ```bash
   source venv_chess/bin/activate
   export OPENAI_API_KEY="sk-..."
   ```
3. Launch the controller:

   ```bash
   python main_controller.py
   ```

   *The script randomly decides whether the LLM plays White.*
4. Watch the terminal for latency metrics and prompts in case of unreachable poses (you can manually move pieces when asked).
5. Stop with **Ctrl‑C** or let the game finish naturally.

### 5.3 Manual intervention & recovery

* If a motion fails, the executor pauses and asks you to move the piece by hand; press **Enter** to continue skipped rest.
* After a collision, clear the fault in Niryo Studio and rerun.

---

## 6 Tuning

| Parameter                                | Location                  | When to change                  |
| ---------------------------------------- | ------------------------- | ------------------------------- |
| `piece_height_lookup`                    | `robot_executor.py`       | Different chess pieces          |
| `MIN_Z` / ±0.12 “high” offset            | `robot_executor.py`       | Safer clearance                 |
| `MARKER2ORIGIN_OFF`, homography `H_GRID` | `realsense_aruco_pose.py` | Not running with Niryo Ned2, or a differnet setup from what is specified is used |

---

## 7 Troubleshooting

| Symptom                                 | Likely cause                  | Remedy                                                      |
| --------------------------------------- | ----------------------------- | ----------------------------------------------------------- |
| `estimatePoseSingleMarkers` **missing** | OpenCV < 4.7 in **venv\_cal** | `pip install --upgrade opencv-contrib-python`               |
| **No RealSense frames**                 | Missing udev rules            | See §4 above                                                |
| Niryo **IK/planner error**              | Target XY unreachable         | Adjust board position or piece heights; use manual fallback |
| LLM returns **invalid move**            | API latency or key issue      | Check `OPENAI_API_KEY`, internet; retry                     |

---

## 8 Licence & acknowledgements

All original source files are © 2025 Chenglin Sun (Imperial College London) and released under the MIT Licence unless stated otherwise.  This project builds on open‑source libraries released under MIT, BSD or Apache 2.0; see individual files for notices.
