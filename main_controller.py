# main_controller.py

from stockfish import Stockfish
from llm_player import get_llm_move
from motion_planner import plan_motion
from robot_executor import execute_motion

import random
import time


# === Configuration ===
STOCKFISH_PATH = "/usr/games/stockfish" 
LLM_IS_WHITE = random.choice([True, False])

# === Init Stockfish ===
stockfish = Stockfish(path=STOCKFISH_PATH, depth=15, parameters={"Threads": 2})
stockfish.set_fen_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

# === Memory model (optional) ===
move_history = []

# === Game loop ===
turn = "white"
while True:
    print("\n--- Turn:", turn.upper(), "---")

    fen_before_move = stockfish.get_fen_position()
    best_move = None

    if (turn == "white" and LLM_IS_WHITE) or (turn == "black" and not LLM_IS_WHITE):
        # === LLM Plays ===
        llm_start = time.time()
        while True:
            candidate_move = get_llm_move(fen_before_move, move_history)
            if stockfish.is_move_correct(candidate_move):
                best_move = candidate_move
                break
            print("LLM move invalid:", candidate_move)
        llm_end = time.time()
        print(f"[Latency] LLM decision time: {llm_end - llm_start:.3f} seconds")
    else:
        # === Stockfish Plays ===
        stockfish_start = time.time()
        best_move = stockfish.get_best_move()
        if best_move is None:
            print("Game Over: No legal moves")
            break
        stockfish_end = time.time()
        print(f"[Latency] stockfish decision time: {stockfish_end - stockfish_start:.3f} seconds")

    print(f"Move played: {best_move}")

    # === Motion planning + execution (before applying move in engine) ===
    try:
        plan_start = time.time()
        plan = plan_motion(move_instruction=best_move, fen=fen_before_move)
        plan_end = time.time()
        print(f"[Latency] LLM Motion planning time: {plan_end - plan_start:.3f} seconds")


        exec_start = time.time()
        execute_motion(plan, fen_string=fen_before_move)
        exec_end = time.time()
        print(f"[Latency] Motion execution time: {exec_end - exec_start:.3f} seconds")
    except Exception as e:
        print("Motion execution error:", e)

    # === Apply move in Stockfish after physical execution ===
    stockfish.make_moves_from_current_position([best_move])
    move_history.append(best_move)

    # === Optional: Check if game is over ===
    if stockfish.get_best_move() is None:
        print("Game Over: No legal next move")
        break

    # === Switch turn ===
    turn = "black" if turn == "white" else "white"
    time.sleep(1)
