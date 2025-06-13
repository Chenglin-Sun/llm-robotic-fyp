# robot_executor.py

from pyniryo import *
import time
import os
import ast
from chessboard_table import chessboard_positions_list  # Must be defined elsewhere

# helpers_motion.py  (or inside robot_executor.py – your choice)
import math
from pyniryo import NiryoRobot, PoseObject, PoseMetadata
from pyniryo.api.exceptions import TcpCommandException

# Constants
ROBOT_IP = "169.254.200.200"
MIN_Z = 0.14

DISPOSAL_OFFSET = 0.10  # meters left of A1 along A1H1 direction


# Piece height lookup in meters
piece_height_lookup = {
    'P': 0.02,  # White Pawn
    'N': 0.04,  # White Knight
    'B': 0.04,  # White Bishop
    'R': 0.035,  # White Rook
    'Q': 0.05,  # White Queen
    'K': 0.06,  # White King
    'p': 0.02,  # Black Pawn
    'n': 0.04,
    'b': 0.04,
    'r': 0.035,
    'q': 0.05,
    'k': 0.06,
}

EPS_LIN = 0.01      # 1 cm – below that we say “didn’t move”

def _distance(p1, p2):
    return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)

def ensure_gripper_open(robot: NiryoRobot) -> None:
    """Idempotently open the tool gripper.

    *If the tool is already open the command is ignored by firmware, so
    this call is always safe.*  We swallow any TCP error to keep startup
    bullet‑proof when no tool is mounted.
    """
    try:
        robot.release_with_tool()
    except TcpCommandException:
        pass
    except Exception as exc:
        print(f"[WARN] could not open gripper at startup: {exc}")

def move_with_user_fallback(robot: NiryoRobot,
                            pose_xyzrpy: list[float] | tuple[float, ...],
                            *,
                            linear: bool = False) -> bool:
    """
    Return True  → pose reached
           False → failure (unreachable / collision / operator abort)
    """
    pose   = PoseObject(*pose_xyzrpy, metadata=PoseMetadata.v2())
    before = robot.get_pose()

    try:
        robot.move_pose(pose)
    except TcpCommandException as exc:
        print("⚠️  IK/planner error:", exc)
    else:
        after = robot.get_pose()
        if _distance(before, after) > EPS_LIN and not robot.collision_detected:
            return True
        if robot.collision_detected:
            robot.clear_collision_detected()

    # ── fallback ────────────────────────────────────────────────────────────
    print("\n🚧  Arm could not reach that square.")
    print("     Move the piece by hand, then press <Enter> to CONTINUE, or")
    print("     simply press <Enter> again to ABORT the rest of the script.")
    user_input = input()
    return bool(user_input.strip())        # empty → abort; anything → continue


def fen_to_square_map(fen: str) -> dict[str, str]:
    """
    Convert FEN to a dictionary mapping square names to piece characters.
    """
    rows = fen.split()[0].split('/')
    square_map = {}
    files = "abcdefgh"

    for rank_idx, row in enumerate(rows):
        file_idx = 0
        for char in row:
            if char.isdigit():
                file_idx += int(char)
            else:
                square = files[file_idx] + str(8 - rank_idx)
                square_map[square.upper()] = char
                file_idx += 1
    return square_map




def adjust_z(z):
    return max(z, MIN_Z)

def read_command_file(file_path):
    task_list = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                task = ast.literal_eval(line.strip())
                task_list.append(task)
            except Exception:
                print(f"Skipping invalid line: {line}")
    return task_list

def get_exact_position(position_name):
    if position_name == "disposal":
        x, y, _ = get_exact_position("A1")
        return x - DISPOSAL_OFFSET, y, 0
    for position in chessboard_positions_list:
        if position['name'].upper() == position_name.upper():
            return position['position'][0], position['position'][1], 0
    raise ValueError(f"Position '{position_name}' not found")

def get_target_pose(task, resolved_position, roll=0.0, pitch=1.57, yaw=0.0):
    x, y, _ = resolved_position
    z = MIN_Z

    if task[0] == "move":
        piece_or_height = task[2]
        if piece_or_height.lower() == "high":
            z += 0.12
        elif piece_or_height.lower() in piece_height_lookup:
            z += piece_height_lookup[piece_or_height.lower()]
        else:
            z += 0.10  # fallback
    z = adjust_z(z)
    return [x, y, z, roll, pitch, yaw]



def resolve_position(task):
    if task[1] == "initial position":
        return (0.23, 0.0, 0.34)
    return get_exact_position(task[1])


def execute_command(robot, task) -> bool:
    if task[0] == "grab":
        robot.grasp_with_tool()
        moved = True
    elif task[0] == "release":
        robot.release_with_tool()
        moved = True
    elif task[0] == "move":
        if task[1] == "initial position":
            x,y,z = resolve_position(task)
            pose = [x, y, z, 0.0, 1.57, 0.0]
        else:
            resolved_position = resolve_position(task)
            pose = get_target_pose(task, resolved_position)
        moved = move_with_user_fallback(robot, pose, linear=False)
        if moved:
            print("🤖  Robot moved successfully.")
        else:
            print("👤  Operator handled manually.")
    else:
        raise ValueError(f"Unknown task: {task}")
    time.sleep(1)   
    return moved


def execute_motion(command_file_path, fen_string=None):
    square_map = fen_to_square_map(fen_string) if fen_string else {}

    print("[DEBUG] FEN square map:", square_map)  # ← Add this line

    task_list = read_command_file(command_file_path)

    held_piece = [None]  # Mutable list to simulate reference


    robot = NiryoRobot(ROBOT_IP)
    try:
        robot.calibrate_auto()
        robot.update_tool()
        ensure_gripper_open(robot)
        robot.clear_collision_detected() 

        for task in task_list:
            print(f"Executing: {task}")
            if not execute_command(robot, task):
                print("⛔  Task aborted by operator / collision. "
                    "Skipping the remaining commands.")
                break

        print("✅ All tasks completed.")
    except Exception as e:  
        print(f"❌ Execution error: {e}")
    finally:
        robot.close_connection()

# === Test Mode ===
if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        command_path = sys.argv[1]
        print("⚠️  No FEN provided — using default fallback FEN.")
        default_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        execute_motion(command_path, fen_string=default_fen)
    elif len(sys.argv) == 3:
        command_path = sys.argv[1]  
        fen_str = sys.argv[2]
        execute_motion(command_path, fen_string=fen_str)
    else:
        print("Usage: python3 robot_executor.py <command_file.txt> [optional_fen_string]")

