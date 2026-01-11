"""
Chess LLM Client
Connects to model_server to get LLM-generated chess moves.
"""

import sys
import os
import re
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.local_client import LocalLLMClient
from config import MODEL_SERVER_URL


class ChessLLMClient:
    """Client that uses LLM to analyze chess positions and suggest moves."""
    
    def __init__(self):
        self.llm = LocalLLMClient(MODEL_SERVER_URL)
    
    def get_board_text(self, game) -> str:
        """Convert board state to text representation for LLM."""
        state = game.get_game_state()
        
        # Piece symbols
        symbols = {
            'white': {'king': 'K', 'queen': 'Q', 'rook': 'R', 'bishop': 'B', 'knight': 'N', 'pawn': 'P'},
            'black': {'king': 'k', 'queen': 'q', 'rook': 'r', 'bishop': 'b', 'knight': 'n', 'pawn': 'p'}
        }
        
        lines = []
        lines.append("  a b c d e f g h")
        lines.append("  ─────────────────")
        
        for row in range(8):
            row_str = f"{8-row}│"
            for col in range(8):
                piece = state["board"][row][col]
                if piece:
                    symbol = symbols[piece["color"]][piece["type"]]
                    row_str += f"{symbol} "
                else:
                    row_str += ". "
            row_str += f"│{8-row}"
            lines.append(row_str)
        
        lines.append("  ─────────────────")
        lines.append("  a b c d e f g h")
        
        return "\n".join(lines)
    
    def get_valid_moves_text(self, game) -> str:
        """Get all valid moves in algebraic notation."""
        state = game.get_game_state()
        current_color = state["turn"]
        
        moves = []
        for row in range(8):
            for col in range(8):
                piece = game.get_piece(row, col)
                if piece and piece.color.value == current_color:
                    valid = game.get_valid_moves(row, col)
                    for to_row, to_col in valid:
                        from_sq = self._to_algebraic(row, col)
                        to_sq = self._to_algebraic(to_row, to_col)
                        piece_name = piece.type.value[0].upper()
                        if piece.type.value == "knight":
                            piece_name = "N"
                        elif piece.type.value == "pawn":
                            piece_name = ""
                        moves.append(f"{piece_name}{from_sq}-{to_sq}")
        
        return ", ".join(moves)
    
    def _to_algebraic(self, row: int, col: int) -> str:
        """Convert row, col to algebraic notation (e.g., e4)."""
        files = "abcdefgh"
        ranks = "87654321"
        return f"{files[col]}{ranks[row]}"
    
    def _from_algebraic(self, notation: str) -> tuple:
        """Convert algebraic notation to row, col."""
        files = "abcdefgh"
        ranks = "87654321"
        col = files.index(notation[0].lower())
        row = ranks.index(notation[1])
        return row, col
    
    def get_llm_move(self, game) -> dict:
        """
        Ask the LLM to choose a move for the current position.
        Returns: {"from": [row, col], "to": [row, col]} or None on failure.
        """
        state = game.get_game_state()
        
        if state["gameOver"]:
            return None
        
        board_text = self.get_board_text(game)
        valid_moves = self.get_valid_moves_text(game)
        current_color = state["turn"]
        
        # Build prompt for LLM
        prompt = f"""You are a chess engine playing as {current_color.upper()}.

Current board position:
{board_text}

It is {current_color}'s turn to move.
{"The king is in CHECK!" if state["check"] else ""}

Valid moves available: {valid_moves}

Analyze the position and choose the BEST move. Consider:
1. Material advantage (captures)
2. King safety
3. Piece development
4. Control of center
5. Tactical opportunities

Respond with ONLY the move in this exact format: FROM-TO
For example: e2-e4 or Ng1-f3

Your move:"""

        try:
            # Get LLM response
            response = self.llm.complete(prompt, max_tokens=50, temperature=0.3)
            
            # Parse the move from response
            move = self._parse_move(response, game)
            
            if move:
                return move
            else:
                # Fallback: return first valid move
                return self._get_fallback_move(game)
                
        except Exception as e:
            print(f"LLM error: {e}")
            return self._get_fallback_move(game)
    
    def _parse_move(self, response: str, game) -> dict:
        """Parse LLM response to extract the move."""
        response = response.strip()
        
        # Try to find move pattern like "e2-e4" or "Nf3-e5"
        patterns = [
            r'([a-h][1-8])-([a-h][1-8])',  # e2-e4
            r'[KQRBN]?([a-h][1-8])\s*[-x]\s*([a-h][1-8])',  # Ke1-e2 or Nxf3
            r'([a-h][1-8])\s*to\s*([a-h][1-8])',  # e2 to e4
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                from_sq = match.group(1)
                to_sq = match.group(2)
                
                try:
                    from_row, from_col = self._from_algebraic(from_sq)
                    to_row, to_col = self._from_algebraic(to_sq)
                    
                    # Validate the move
                    valid_moves = game.get_valid_moves(from_row, from_col)
                    if (to_row, to_col) in valid_moves:
                        return {
                            "from": [from_row, from_col],
                            "to": [to_row, to_col]
                        }
                except:
                    continue
        
        return None
    
    def _get_fallback_move(self, game) -> dict:
        """Get a valid move when LLM parsing fails."""
        state = game.get_game_state()
        current_color = state["turn"]
        
        # Priority: captures > center control > any move
        best_move = None
        
        for row in range(8):
            for col in range(8):
                piece = game.get_piece(row, col)
                if piece and piece.color.value == current_color:
                    valid = game.get_valid_moves(row, col)
                    for to_row, to_col in valid:
                        target = game.get_piece(to_row, to_col)
                        
                        move = {"from": [row, col], "to": [to_row, to_col]}
                        
                        # Prefer captures
                        if target:
                            return move
                        
                        # Prefer center squares
                        if 2 <= to_row <= 5 and 2 <= to_col <= 5:
                            best_move = move
                        
                        # Store as fallback
                        if best_move is None:
                            best_move = move
        
        return best_move


# Singleton instance
_client = None

def get_chess_client() -> ChessLLMClient:
    """Get or create the chess LLM client instance."""
    global _client
    if _client is None:
        _client = ChessLLMClient()
    return _client
