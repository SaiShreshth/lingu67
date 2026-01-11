"""
Stockfish API Client
Connects to chess-api.com for Stockfish 17 analysis and moves.
"""

import httpx
import asyncio
from typing import Optional, Dict


class StockfishClient:
    """Client for chess-api.com Stockfish 17 API."""
    
    API_URL = "https://chess-api.com/v1"
    
    def __init__(self, depth: int = 12, max_thinking_time: int = 50):
        self.depth = depth
        self.max_thinking_time = max_thinking_time
    
    def get_fen(self, game) -> str:
        """Convert game state to FEN notation."""
        state = game.get_game_state()
        board = state["board"]
        
        fen_rows = []
        for row in range(8):
            fen_row = ""
            empty_count = 0
            
            for col in range(8):
                piece = board[row][col]
                if piece is None:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    
                    # Piece symbol
                    symbols = {
                        'king': 'k', 'queen': 'q', 'rook': 'r',
                        'bishop': 'b', 'knight': 'n', 'pawn': 'p'
                    }
                    symbol = symbols[piece["type"]]
                    if piece["color"] == "white":
                        symbol = symbol.upper()
                    fen_row += symbol
            
            if empty_count > 0:
                fen_row += str(empty_count)
            fen_rows.append(fen_row)
        
        # Build FEN string
        position = "/".join(fen_rows)
        turn = "w" if state["turn"] == "white" else "b"
        
        # Simplified castling and en passant (full implementation would track these)
        castling = "KQkq"  # Simplified
        en_passant = "-"
        halfmove = "0"
        fullmove = str(len(state["history"]) // 2 + 1)
        
        return f"{position} {turn} {castling} {en_passant} {halfmove} {fullmove}"
    
    def get_best_move(self, game, variants: int = 1) -> Dict:
        """
        Get the best move from Stockfish API.
        Returns dict with move info, eval, commentary text.
        """
        fen = self.get_fen(game)
        
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    self.API_URL,
                    json={
                        "fen": fen,
                        "depth": self.depth,
                        "maxThinkingTime": self.max_thinking_time,
                        "variants": variants
                    },
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return self._parse_response(data, game)
                else:
                    return {"error": f"API error: {response.status_code}"}
                    
        except Exception as e:
            return {"error": str(e)}
    
    def _parse_response(self, data: Dict, game) -> Dict:
        """Parse Stockfish API response into move format."""
        if "error" in data:
            return {"error": data.get("error", "Unknown error")}
        
        # Extract move coordinates
        move_lan = data.get("lan", data.get("move", ""))
        
        if len(move_lan) >= 4:
            from_sq = move_lan[:2]
            to_sq = move_lan[2:4]
            
            from_coords = self._algebraic_to_coords(from_sq)
            to_coords = self._algebraic_to_coords(to_sq)
            
            if from_coords and to_coords:
                return {
                    "success": True,
                    "from": list(from_coords),
                    "to": list(to_coords),
                    "eval": data.get("eval", 0),
                    "depth": data.get("depth", 12),
                    "text": data.get("text", ""),
                    "san": data.get("san", ""),
                    "winChance": data.get("winChance", 50),
                    "mate": data.get("mate"),
                    "continuation": data.get("continuationArr", [])
                }
        
        return {"error": "Could not parse move"}
    
    def _algebraic_to_coords(self, notation: str) -> Optional[tuple]:
        """Convert algebraic notation (e.g., 'e4') to row, col."""
        if len(notation) < 2:
            return None
        
        files = "abcdefgh"
        ranks = "87654321"
        
        file_char = notation[0].lower()
        rank_char = notation[1]
        
        if file_char in files and rank_char in ranks:
            col = files.index(file_char)
            row = ranks.index(rank_char)
            return (row, col)
        
        return None
    
    def get_position_eval(self, game) -> Dict:
        """Get position evaluation without making a move."""
        fen = self.get_fen(game)
        
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    self.API_URL,
                    json={
                        "fen": fen,
                        "depth": self.depth,
                        "maxThinkingTime": self.max_thinking_time
                    },
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "eval": data.get("eval", 0),
                        "winChance": data.get("winChance", 50),
                        "text": data.get("text", ""),
                        "depth": data.get("depth", 12),
                        "mate": data.get("mate"),
                        "bestMove": data.get("san", "")
                    }
                    
        except Exception as e:
            return {"error": str(e)}
        
        return {"error": "API request failed"}


# Singleton
_stockfish_client = None

def get_stockfish_client(depth: int = 12) -> StockfishClient:
    """Get or create Stockfish client instance."""
    global _stockfish_client
    if _stockfish_client is None:
        _stockfish_client = StockfishClient(depth=depth)
    return _stockfish_client
