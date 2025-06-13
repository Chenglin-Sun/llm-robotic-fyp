# llm_player.py

from openai import OpenAI
client = OpenAI()
import chess

def _legal_uci_moves_from_fen(fen: str) -> list[str]:
    """
    Uses python-chess to enumerate every legal move in the given FEN.
    """
    board = chess.Board(fen)
    return [m.uci() for m in board.legal_moves]

def get_llm_move(fen: str, history: list[str]) -> str:
    """
    Given the current FEN and move history, ask the LLM to return a legal move in UCI.
    """

    # 1. Figure out which side is to move
    parts = fen.split()
    if len(parts) < 2:
        raise ValueError(f"Invalid FEN: {fen}")
    side = "White" if parts[1] == "w" else "Black"

    legal_moves = _legal_uci_moves_from_fen(fen)

    # 2. Build messages, including side info
    messages = [
        {
            "role": "system",
            "content": (
                f"You are a chess-playing AI controlling {side}. "
                "Your job is to decide one legal move from the given board state (in FEN). "
                "Choose exactly ONE move from the list that follows. "
                "Return *only* the move in UCI format (e.g. e2e4). "
                "Return the move in exact 4- or 5-character UCI."
                "Do not explain or add anything else. "
                "Return ‘None’ only if there are zero legal moves”"
            )
        },
        {
            "role": "user",
            "content": (
                f"Position (FEN): {fen}\n\n"
                f"Legal moves: {', '.join(legal_moves)}"
            )
        }
    ]

    # 3. Query OpenAI
    response = client.chat.completions.create(
        model="gpt-4o",  # or "gpt-4"
        messages=messages,
        temperature=0.2
    )
    move = response.choices[0].message.content.strip().lower()

    return move


if __name__ == "__main__":
    # Quick sanity check
    for test_fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "6k1/5ppp/8/8/8/8/6PP/5qK1 w - - 0 1",
        "6k1/5ppp/8/8/8/8/6PP/5qK1 b - - 0 1",
    ]:
        print(test_fen, "→", get_llm_move(test_fen, []))
