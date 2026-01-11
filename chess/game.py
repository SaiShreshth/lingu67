"""
Chess Game Logic
Full piece movement rules, validation, and game state management.
"""

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import copy


class PieceType(Enum):
    PAWN = "pawn"
    ROOK = "rook"
    KNIGHT = "knight"
    BISHOP = "bishop"
    QUEEN = "queen"
    KING = "king"


class Color(Enum):
    WHITE = "white"
    BLACK = "black"


@dataclass
class Piece:
    type: PieceType
    color: Color
    has_moved: bool = False
    
    def to_dict(self):
        return {
            "type": self.type.value,
            "color": self.color.value
        }


class ChessGame:
    """Full chess game with move validation and game state."""
    
    def __init__(self):
        self.board: List[List[Optional[Piece]]] = [[None] * 8 for _ in range(8)]
        self.current_turn: Color = Color.WHITE
        self.move_history: List[Dict] = []
        self.captured_pieces: Dict[str, List[str]] = {"white": [], "black": []}
        self.game_over: bool = False
        self.winner: Optional[Color] = None
        self._setup_board()
    
    def _setup_board(self):
        """Set up initial board position."""
        # Pawns
        for col in range(8):
            self.board[1][col] = Piece(PieceType.PAWN, Color.BLACK)
            self.board[6][col] = Piece(PieceType.PAWN, Color.WHITE)
        
        # Back row pieces
        back_row = [
            PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP, PieceType.QUEEN,
            PieceType.KING, PieceType.BISHOP, PieceType.KNIGHT, PieceType.ROOK
        ]
        
        for col, piece_type in enumerate(back_row):
            self.board[0][col] = Piece(piece_type, Color.BLACK)
            self.board[7][col] = Piece(piece_type, Color.WHITE)
    
    def get_board_state(self) -> List[List[Optional[Dict]]]:
        """Return board state as serializable format."""
        result = []
        for row in self.board:
            row_data = []
            for cell in row:
                row_data.append(cell.to_dict() if cell else None)
            result.append(row_data)
        return result
    
    def get_piece(self, row: int, col: int) -> Optional[Piece]:
        """Get piece at position."""
        if 0 <= row < 8 and 0 <= col < 8:
            return self.board[row][col]
        return None
    
    def get_valid_moves(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get all valid moves for a piece at given position."""
        piece = self.get_piece(row, col)
        if not piece or piece.color != self.current_turn:
            return []
        
        moves = self._get_raw_moves(row, col, piece)
        
        # Filter moves that would leave king in check
        valid_moves = []
        for move in moves:
            if not self._would_be_in_check(row, col, move[0], move[1]):
                valid_moves.append(move)
        
        return valid_moves
    
    def _get_raw_moves(self, row: int, col: int, piece: Piece) -> List[Tuple[int, int]]:
        """Get raw moves without check validation."""
        if piece.type == PieceType.PAWN:
            return self._pawn_moves(row, col, piece)
        elif piece.type == PieceType.ROOK:
            return self._rook_moves(row, col, piece)
        elif piece.type == PieceType.KNIGHT:
            return self._knight_moves(row, col, piece)
        elif piece.type == PieceType.BISHOP:
            return self._bishop_moves(row, col, piece)
        elif piece.type == PieceType.QUEEN:
            return self._queen_moves(row, col, piece)
        elif piece.type == PieceType.KING:
            return self._king_moves(row, col, piece)
        return []
    
    def _pawn_moves(self, row: int, col: int, piece: Piece) -> List[Tuple[int, int]]:
        """Calculate pawn moves."""
        moves = []
        direction = -1 if piece.color == Color.WHITE else 1
        start_row = 6 if piece.color == Color.WHITE else 1
        
        # Forward one
        new_row = row + direction
        if 0 <= new_row < 8 and self.board[new_row][col] is None:
            moves.append((new_row, col))
            
            # Forward two from start
            if row == start_row:
                new_row2 = row + 2 * direction
                if self.board[new_row2][col] is None:
                    moves.append((new_row2, col))
        
        # Diagonal captures
        for dc in [-1, 1]:
            new_col = col + dc
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                target = self.board[new_row][new_col]
                if target and target.color != piece.color:
                    moves.append((new_row, new_col))
        
        return moves
    
    def _rook_moves(self, row: int, col: int, piece: Piece) -> List[Tuple[int, int]]:
        """Calculate rook moves."""
        return self._slide_moves(row, col, piece, [(0, 1), (0, -1), (1, 0), (-1, 0)])
    
    def _bishop_moves(self, row: int, col: int, piece: Piece) -> List[Tuple[int, int]]:
        """Calculate bishop moves."""
        return self._slide_moves(row, col, piece, [(1, 1), (1, -1), (-1, 1), (-1, -1)])
    
    def _queen_moves(self, row: int, col: int, piece: Piece) -> List[Tuple[int, int]]:
        """Calculate queen moves."""
        return self._slide_moves(row, col, piece, [
            (0, 1), (0, -1), (1, 0), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ])
    
    def _slide_moves(self, row: int, col: int, piece: Piece, 
                     directions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Calculate sliding piece moves (rook, bishop, queen)."""
        moves = []
        for dr, dc in directions:
            r, c = row + dr, col + dc
            while 0 <= r < 8 and 0 <= c < 8:
                target = self.board[r][c]
                if target is None:
                    moves.append((r, c))
                elif target.color != piece.color:
                    moves.append((r, c))
                    break
                else:
                    break
                r, c = r + dr, c + dc
        return moves
    
    def _knight_moves(self, row: int, col: int, piece: Piece) -> List[Tuple[int, int]]:
        """Calculate knight moves."""
        moves = []
        offsets = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        for dr, dc in offsets:
            r, c = row + dr, col + dc
            if 0 <= r < 8 and 0 <= c < 8:
                target = self.board[r][c]
                if target is None or target.color != piece.color:
                    moves.append((r, c))
        return moves
    
    def _king_moves(self, row: int, col: int, piece: Piece) -> List[Tuple[int, int]]:
        """Calculate king moves."""
        moves = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if 0 <= r < 8 and 0 <= c < 8:
                    target = self.board[r][c]
                    if target is None or target.color != piece.color:
                        moves.append((r, c))
        return moves
    
    def _find_king(self, color: Color) -> Tuple[int, int]:
        """Find the king's position."""
        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if piece and piece.type == PieceType.KING and piece.color == color:
                    return (r, c)
        return (-1, -1)
    
    def _is_in_check(self, color: Color) -> bool:
        """Check if the given color's king is in check."""
        king_pos = self._find_king(color)
        opponent = Color.BLACK if color == Color.WHITE else Color.WHITE
        
        # Check if any opponent piece can attack the king
        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if piece and piece.color == opponent:
                    moves = self._get_raw_moves(r, c, piece)
                    if king_pos in moves:
                        return True
        return False
    
    def _would_be_in_check(self, from_row: int, from_col: int, 
                           to_row: int, to_col: int) -> bool:
        """Check if a move would leave the current player in check."""
        # Make temporary move
        piece = self.board[from_row][from_col]
        captured = self.board[to_row][to_col]
        
        self.board[to_row][to_col] = piece
        self.board[from_row][from_col] = None
        
        in_check = self._is_in_check(self.current_turn)
        
        # Undo move
        self.board[from_row][from_col] = piece
        self.board[to_row][to_col] = captured
        
        return in_check
    
    def make_move(self, from_row: int, from_col: int, 
                  to_row: int, to_col: int) -> Dict:
        """
        Attempt to make a move. Returns result dict.
        """
        piece = self.get_piece(from_row, from_col)
        
        if not piece:
            return {"success": False, "error": "No piece at source"}
        
        if piece.color != self.current_turn:
            return {"success": False, "error": "Not your turn"}
        
        valid_moves = self.get_valid_moves(from_row, from_col)
        
        if (to_row, to_col) not in valid_moves:
            return {"success": False, "error": "Invalid move"}
        
        # Execute move
        captured = self.board[to_row][to_col]
        if captured:
            self.captured_pieces[self.current_turn.value].append(captured.type.value)
        
        self.board[to_row][to_col] = piece
        self.board[from_row][from_col] = None
        piece.has_moved = True
        
        # Pawn promotion (auto-queen)
        if piece.type == PieceType.PAWN:
            if (piece.color == Color.WHITE and to_row == 0) or \
               (piece.color == Color.BLACK and to_row == 7):
                self.board[to_row][to_col] = Piece(PieceType.QUEEN, piece.color, True)
        
        # Record move
        self.move_history.append({
            "from": [from_row, from_col],
            "to": [to_row, to_col],
            "piece": piece.type.value,
            "color": piece.color.value,
            "captured": captured.type.value if captured else None
        })
        
        # Switch turn
        self.current_turn = Color.BLACK if self.current_turn == Color.WHITE else Color.WHITE
        
        # Check for checkmate/stalemate
        if self._is_checkmate():
            self.game_over = True
            self.winner = Color.WHITE if self.current_turn == Color.BLACK else Color.BLACK
            return {"success": True, "checkmate": True, "winner": self.winner.value}
        
        in_check = self._is_in_check(self.current_turn)
        
        return {
            "success": True,
            "captured": captured.type.value if captured else None,
            "check": in_check
        }
    
    def _is_checkmate(self) -> bool:
        """Check if current player is in checkmate."""
        if not self._is_in_check(self.current_turn):
            return False
        
        # Check if any move can get out of check
        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if piece and piece.color == self.current_turn:
                    if self.get_valid_moves(r, c):
                        return False
        return True
    
    def reset(self):
        """Reset the game to initial state."""
        self.__init__()
    
    def get_game_state(self) -> Dict:
        """Get full game state for frontend."""
        return {
            "board": self.get_board_state(),
            "turn": self.current_turn.value,
            "captured": self.captured_pieces,
            "history": self.move_history,
            "gameOver": self.game_over,
            "winner": self.winner.value if self.winner else None,
            "check": self._is_in_check(self.current_turn)
        }
