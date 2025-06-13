# motion_planner.py

from openai import OpenAI
from datetime import datetime
import os
import sys


client = OpenAI()

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


def plan_motion(move_instruction: str, fen: str) -> str:
    """
    Passes a move instruction (either UCI like 'e2e4' or natural language) to the LLM.
    The LLM is responsible for interpreting the move and generating robot motion commands.
    """

    square_map = fen_to_square_map(fen) if fen else {}

    messages = [
        {
            "role": "system",
            "content": (
                "You are the brain of a robotic arm that plays chess. Your job is to generate robotic movement commands "
                "You will be given:"
                "- A square map string that describes the current board layout."
                "- A move instruction."
                "based on a given instruction, which can be in standard UCI format (e.g. 'e2e4') or in natural language (e.g. 'move the pawn to e4'). "
                "Use the square map to identify piece types and any potential captures. If a square contains an opponent piece,"
                " include a step to move it to a disposal position, before moving the chess in place."
                "Use the following command syntax only:"
                "['move', '<square, initial position or disposal position>', '<height>']"
                "['grab']"
                "['release']"
                "- The <square> refers to a board position like 'E2'."
                "- You may also use 'initial position' to return to initial position after each command is finished, and use 'disposal' to discard any captured chess pieces."
                "If initial postion is used, do not specify a height."
                "- <height> can be:"
                "    • A piece type: 'P', 'N', 'B', 'R', 'Q', 'K', etc."
                "    • Or a symbolic label 'high' for generic high-altitude moves."
                "You will need to infer based on the piece present at sqaure of interest, if any, and the chess type the robotic arm is currently holding, is going to hold, or is going to drop off, in determine the height used in the command."                "- Do not include any explanations or comments."
                "Other than the piece types, you can only use 'high', which moves the arm high above to avoid collisions, this height is also used when disposing captured chess pieces."
                "A rule of thumb: to get into a grabbing or releasing position and avoid collision, we go to that position, start and end on height 'high'."
                "Example 1: Normal move (pawn from E2 to E4)"
                "FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  
                "Instruction: e2e4"

                "['move', 'E2', 'high']"  
                "['move', 'E2', 'P']"  
                "['grab']"  
                "['move', 'E2', 'high']"  
                "['move', 'E4', 'high']"  
                "['move', 'E4', 'P']"  
                "['release']"  
                "['move', 'initial position']"
                "Example 2: Capture scenario (white knight captures black pawn on D5)"
                "FEN: rnbqkbnr/pppp1ppp/8/3p4/8/5N2/PPPPPPPP/RNBQKB1R w KQkq d6 0 3" 
                "Instruction: f3d5"

                "['move', 'D5', 'high']"
                "['move', 'D5', 'P']"  
                "['grab']"  
                "['move', 'D5', 'high']"  
                "['move', 'disposal', 'high']"  
                "['release']"  
                "['move', 'F3', 'high']"  
                "['move', 'F3', 'N']"  
                "['grab']"  
                "['move', 'F3', 'high']" 
                "['move', 'D5', 'high']"  
                "['move', 'D5', 'N']"  
                "['release']" 
                "['move', 'initial position']"
                "- Output only one command per line, no extra text."

            )
        },
        {
            "role": "user",
            "content": f"Current square map: {square_map}\nInstruction: {move_instruction}"
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2
    )

    commands = response.choices[0].message.content.strip()

    # Save to file
    folder_path = "commands"
    os.makedirs(folder_path, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(folder_path, f"commands_{current_time}.txt")

    with open(file_path, "w") as file:
        for line in commands.splitlines():
            file.write(line.strip() + "\n")

    print(f"[motion_planner] Saved command script: {file_path}")
    return file_path

# === Test mode ===
if __name__ == "__main__":
    if len(sys.argv) >= 3:
        fen = sys.argv[1]
        instruction = " ".join(sys.argv[2:])
    else:
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        instruction = "e2e4"
        print("⚠️ Using default FEN and move instruction")

    try:
        plan_motion(instruction, fen)
    except Exception as e:
        print("❌ Error:", str(e))
