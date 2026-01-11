"""
Chess - Flask Web UI
Interactive chess board with dual player selection: Player, LLM, Stockfish 17
Includes live commentary for bot games.
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from game import ChessGame, Color
from chess_client import get_chess_client
from stockfish_client import get_stockfish_client

app = Flask(__name__)
CORS(app)

# Global game state
game = ChessGame()
game_settings = {
    "white_player": "player",  # "player", "llm", "stockfish"
    "black_player": "player",
    "commentary": []  # Live commentary for bot games
}


# ==================== HTML/CSS/JS UI ====================
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Chess - Lingu67</title>
<style>
:root {
    --bg-main: #1e272e;
    --bg-panel: #2d3436;
    --light-square: #f0d9b5;
    --dark-square: #b58863;
    --highlight: rgba(255, 255, 0, 0.5);
    --valid-move: rgba(0, 0, 0, 0.15);
    --selected: rgba(20, 85, 30, 0.5);
    --check: rgba(255, 0, 0, 0.6);
    --text: #dfe6e9;
    --accent: #0984e3;
    --stockfish: #e17055;
    --llm: #6c5ce7;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: var(--bg-main);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    justify-content: center;
    align-items: flex-start;
    padding: 20px;
    gap: 20px;
}

/* Main Layout */
.main-container {
    display: flex;
    gap: 20px;
    align-items: flex-start;
}

/* Board */
.board-area {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.board {
    display: grid;
    grid-template-columns: repeat(8, 60px);
    grid-template-rows: repeat(8, 60px);
    border: 3px solid #1a1a1a;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}

.square {
    width: 60px;
    height: 60px;
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 44px;
    cursor: pointer;
    user-select: none;
    position: relative;
}

.square.light { background: var(--light-square); }
.square.dark { background: var(--dark-square); }
.square.selected { background: var(--selected) !important; }
.square.valid-move::after {
    content: '';
    position: absolute;
    width: 16px;
    height: 16px;
    background: var(--valid-move);
    border-radius: 50%;
}
.square.valid-capture { box-shadow: inset 0 0 0 4px rgba(255, 0, 0, 0.5); }
.square.check { background: var(--check) !important; }
.square.last-move { background: var(--highlight) !important; }

.piece { line-height: 1; }
.piece.white { color: #fff; text-shadow: 0 0 3px #000, 0 0 3px #000; }
.piece.black { color: #000; }

/* Eval Bar */
.eval-bar {
    width: 100%;
    height: 24px;
    background: #333;
    border-radius: 4px;
    overflow: hidden;
    position: relative;
}

.eval-fill {
    height: 100%;
    background: linear-gradient(90deg, #fff 0%, #fff 100%);
    transition: width 0.3s ease;
    width: 50%;
}

.eval-text {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 12px;
    font-weight: bold;
    color: #888;
}

/* Control Panel */
.control-panel {
    background: var(--bg-panel);
    border-radius: 10px;
    padding: 16px;
    min-width: 280px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

.control-panel h2 {
    margin-bottom: 16px;
    font-size: 1.3rem;
    text-align: center;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    padding-bottom: 10px;
}

/* Player Selection */
.player-select {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 16px;
}

.player-box {
    padding: 10px;
    background: rgba(0,0,0,0.2);
    border-radius: 8px;
}

.player-box label {
    display: block;
    font-size: 0.8rem;
    color: #95a5a6;
    margin-bottom: 4px;
}

.player-box select {
    width: 100%;
    padding: 8px;
    border: none;
    border-radius: 6px;
    background: #1a1a1a;
    color: white;
    font-size: 0.85rem;
    cursor: pointer;
}

.player-box select option[value="stockfish"] { color: var(--stockfish); }
.player-box select option[value="llm"] { color: var(--llm); }

/* Turn & Status */
.turn-indicator {
    text-align: center;
    padding: 10px;
    border-radius: 6px;
    margin-bottom: 10px;
    font-weight: 600;
}
.turn-indicator.white { background: #ecf0f1; color: #2c3e50; }
.turn-indicator.black { background: #2c3e50; color: #ecf0f1; border: 1px solid #7f8c8d; }

.status {
    text-align: center;
    padding: 8px;
    margin-bottom: 10px;
    border-radius: 6px;
    font-weight: 500;
    font-size: 0.9rem;
}
.status.check { background: rgba(230, 126, 34, 0.3); color: #e67e22; }
.status.checkmate { background: rgba(231, 76, 60, 0.3); color: #e74c3c; }
.status.thinking { background: rgba(52, 152, 219, 0.3); color: #3498db; }
.status.stockfish { background: rgba(225, 112, 85, 0.2); color: var(--stockfish); }

/* Captured */
.captured { margin-top: 10px; }
.captured h4 { font-size: 0.75rem; color: #95a5a6; margin-bottom: 2px; }
.captured-pieces { font-size: 18px; min-height: 20px; }

/* Commentary */
.commentary {
    margin-top: 14px;
    padding: 10px;
    background: rgba(0,0,0,0.3);
    border-radius: 8px;
    max-height: 150px;
    overflow-y: auto;
}

.commentary h4 {
    font-size: 0.8rem;
    color: #74b9ff;
    margin-bottom: 6px;
}

.commentary-text {
    font-size: 0.8rem;
    color: #b2bec3;
    line-height: 1.4;
}

.commentary-item {
    padding: 4px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}

/* Buttons */
.btn {
    width: 100%;
    padding: 10px;
    margin-top: 10px;
    background: var(--accent);
    border: none;
    border-radius: 6px;
    color: white;
    font-size: 0.9rem;
    cursor: pointer;
    transition: all 0.2s;
}
.btn:hover { background: #0773c7; transform: translateY(-1px); }
.btn:disabled { background: #636e72; cursor: not-allowed; transform: none; }
.btn.stockfish-btn { background: var(--stockfish); }
.btn.stockfish-btn:hover { background: #d35400; }

/* Mobile */
@media (max-width: 800px) {
    body { flex-direction: column; align-items: center; }
    .main-container { flex-direction: column; }
    .board { grid-template-columns: repeat(8, 45px); grid-template-rows: repeat(8, 45px); }
    .square { width: 45px; height: 45px; font-size: 32px; }
}
</style>
</head>
<body>

<div class="main-container">
    <div class="board-area">
        <div class="board" id="board"></div>
        <div class="eval-bar">
            <div class="eval-fill" id="evalFill"></div>
            <div class="eval-text" id="evalText">0.0</div>
        </div>
    </div>
    
    <div class="control-panel">
        <h2>♔ Chess Arena</h2>
        
        <div class="player-select">
            <div class="player-box">
                <label>⬜ White</label>
                <select id="whitePlayer">
                    <option value="player">👤 Player</option>
                    <option value="stockfish">🐟 Stockfish 17</option>
                    <option value="llm">🤖 LLM</option>
                </select>
            </div>
            <div class="player-box">
                <label>⬛ Black</label>
                <select id="blackPlayer">
                    <option value="player">👤 Player</option>
                    <option value="stockfish" selected>🐟 Stockfish 17</option>
                    <option value="llm">🤖 LLM</option>
                </select>
            </div>
        </div>
        
        <div class="turn-indicator" id="turnIndicator">White's Turn</div>
        <div class="status" id="status"></div>
        
        <div class="captured">
            <h4>⬜ captured:</h4>
            <div class="captured-pieces" id="whiteCaptured"></div>
        </div>
        <div class="captured">
            <h4>⬛ captured:</h4>
            <div class="captured-pieces" id="blackCaptured"></div>
        </div>
        
        <div class="commentary" id="commentaryBox" style="display:none">
            <h4>📢 Commentary</h4>
            <div id="commentaryContent"></div>
        </div>
        
        <button class="btn" onclick="startGame()">▶ Start Game</button>
        <button class="btn stockfish-btn" id="autoPlayBtn" style="display:none" onclick="toggleAutoPlay()">⏸ Pause</button>
    </div>
</div>

<script>
const PIECES = {
    'pawn': '♟', 'rook': '♜', 'knight': '♞',
    'bishop': '♝', 'queen': '♛', 'king': '♚'
};

let selectedSquare = null;
let validMoves = [];
let gameState = null;
let lastMove = null;
let whitePlayer = 'player';
let blackPlayer = 'stockfish';
let autoPlayInterval = null;
let isAutoPlaying = false;
let isBotThinking = false;  // Mutex lock

async function fetchState() {
    try {
        const res = await fetch('/api/state');
        gameState = await res.json();
        renderBoard();
        updateInfo();
    } catch (e) {
        console.error('fetchState error:', e);
    }
}

async function getValidMoves(row, col) {
    const res = await fetch(`/api/moves?row=${row}&col=${col}`);
    return await res.json();
}

async function makeMove(fromRow, fromCol, toRow, toCol) {
    const res = await fetch('/api/move', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({from_row: fromRow, from_col: fromCol, to_row: toRow, to_col: toCol})
    });
    const result = await res.json();
    if (result.success) {
        lastMove = {from: [fromRow, fromCol], to: [toRow, toCol]};
        await fetchState();
        scheduleNextBotMove();
    }
    return result;
}

async function startGame() {
    // Stop any existing game
    isAutoPlaying = false;
    isBotThinking = false;
    
    whitePlayer = document.getElementById('whitePlayer').value;
    blackPlayer = document.getElementById('blackPlayer').value;
    
    await fetch('/api/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({white_player: whitePlayer, black_player: blackPlayer})
    });
    
    selectedSquare = null;
    validMoves = [];
    lastMove = null;
    document.getElementById('commentaryContent').innerHTML = '';
    
    await fetchState();
    
    // Show commentary for bot games
    const isBotGame = whitePlayer !== 'player' && blackPlayer !== 'player';
    document.getElementById('commentaryBox').style.display = isBotGame ? 'block' : 'none';
    document.getElementById('autoPlayBtn').style.display = isBotGame ? 'block' : 'none';
    
    if (isBotGame) {
        isAutoPlaying = true;
        document.getElementById('autoPlayBtn').textContent = '⏸ Pause';
    }
    
    // Trigger first bot move if needed
    setTimeout(scheduleNextBotMove, 1000);
}

function scheduleNextBotMove() {
    if (gameState.gameOver || isBotThinking || !isAutoPlaying) {
        return;
    }
    
    const currentPlayer = gameState.turn === 'white' ? whitePlayer : blackPlayer;
    
    if (currentPlayer !== 'player') {
        // Delay between moves for readability
        setTimeout(triggerBotMove, 1500);
    }
}

async function triggerBotMove() {
    // Check all conditions again
    if (gameState.gameOver || !isAutoPlaying || isBotThinking) {
        return;
    }
    
    const currentPlayer = gameState.turn === 'white' ? whitePlayer : blackPlayer;
    if (currentPlayer === 'player') {
        return;
    }
    
    // Set lock
    isBotThinking = true;
    
    const statusEl = document.getElementById('status');
    
    if (currentPlayer === 'stockfish') {
        statusEl.textContent = '🐟 Stockfish thinking...';
        statusEl.className = 'status stockfish';
    } else if (currentPlayer === 'llm') {
        statusEl.textContent = '🤖 LLM thinking...';
        statusEl.className = 'status thinking';
    }
    
    try {
        const res = await fetch(`/api/bot_move?type=${currentPlayer}`, {method: 'POST'});
        const result = await res.json();
        
        if (result.success && result.move) {
            lastMove = {from: result.move.from, to: result.move.to};
            
            // Add commentary
            if (result.text) {
                addCommentary(result.text, currentPlayer);
            }
        } else {
            console.error('Bot move failed:', result.error);
            addCommentary('Error: ' + (result.error || 'Unknown error'), currentPlayer);
        }
        
        await fetchState();
        
    } catch (e) {
        console.error('triggerBotMove error:', e);
        addCommentary('Error: ' + e.message, currentPlayer);
    }
    
    // Release lock
    isBotThinking = false;
    
    // Schedule next move
    if (!gameState.gameOver && isAutoPlaying) {
        scheduleNextBotMove();
    }
}

function toggleAutoPlay() {
    isAutoPlaying = !isAutoPlaying;
    const btn = document.getElementById('autoPlayBtn');
    btn.textContent = isAutoPlaying ? '⏸ Pause' : '▶ Resume';
    
    if (isAutoPlaying && !isBotThinking) {
        scheduleNextBotMove();
    }
}

function addCommentary(text, player) {
    const container = document.getElementById('commentaryContent');
    const icon = player === 'stockfish' ? '🐟' : '🤖';
    const item = document.createElement('div');
    item.className = 'commentary-item';
    item.innerHTML = `<span>${icon}</span> ${text}`;
    container.insertBefore(item, container.firstChild);
    
    // Keep only last 10
    while (container.children.length > 10) {
        container.removeChild(container.lastChild);
    }
}

async function updateEvalBar() {
    try {
        const res = await fetch('/api/eval');
        const data = await res.json();
        
        if (data.winChance !== undefined) {
            const fill = document.getElementById('evalFill');
            const text = document.getElementById('evalText');
            
            fill.style.width = `${data.winChance}%`;
            
            const evalVal = data.eval || 0;
            const sign = evalVal >= 0 ? '+' : '';
            text.textContent = data.mate ? `M${data.mate}` : `${sign}${evalVal.toFixed(1)}`;
        }
    } catch (e) {}
}

function renderBoard() {
    const board = document.getElementById('board');
    board.innerHTML = '';
    
    for (let row = 0; row < 8; row++) {
        for (let col = 0; col < 8; col++) {
            const square = document.createElement('div');
            const isLight = (row + col) % 2 === 0;
            square.className = `square ${isLight ? 'light' : 'dark'}`;
            
            if (lastMove) {
                if ((row === lastMove.from[0] && col === lastMove.from[1]) ||
                    (row === lastMove.to[0] && col === lastMove.to[1])) {
                    square.classList.add('last-move');
                }
            }
            if (selectedSquare && selectedSquare[0] === row && selectedSquare[1] === col) {
                square.classList.add('selected');
            }
            const isValidMove = validMoves.some(m => m[0] === row && m[1] === col);
            if (isValidMove) {
                const piece = gameState.board[row][col];
                square.classList.add(piece ? 'valid-capture' : 'valid-move');
            }
            
            const piece = gameState.board[row][col];
            if (piece) {
                if (piece.type === 'king' && piece.color === gameState.turn && gameState.check) {
                    square.classList.add('check');
                }
                const pieceSpan = document.createElement('span');
                pieceSpan.className = `piece ${piece.color}`;
                pieceSpan.textContent = PIECES[piece.type];
                square.appendChild(pieceSpan);
            }
            
            square.onclick = () => handleClick(row, col);
            board.appendChild(square);
        }
    }
}

async function handleClick(row, col) {
    if (gameState.gameOver) return;
    
    // Check if current player is human
    const currentPlayer = gameState.turn === 'white' ? whitePlayer : blackPlayer;
    if (currentPlayer !== 'player') return;
    
    const piece = gameState.board[row][col];
    
    if (selectedSquare && validMoves.some(m => m[0] === row && m[1] === col)) {
        await makeMove(selectedSquare[0], selectedSquare[1], row, col);
        selectedSquare = null;
        validMoves = [];
        return;
    }
    
    if (piece && piece.color === gameState.turn) {
        selectedSquare = [row, col];
        const moves = await getValidMoves(row, col);
        validMoves = moves.moves || [];
        renderBoard();
        return;
    }
    
    selectedSquare = null;
    validMoves = [];
    renderBoard();
}

function updateInfo() {
    const turnEl = document.getElementById('turnIndicator');
    turnEl.textContent = gameState.turn === 'white' ? "White's Turn" : "Black's Turn";
    turnEl.className = `turn-indicator ${gameState.turn}`;
    
    const statusEl = document.getElementById('status');
    if (gameState.gameOver) {
        statusEl.textContent = `🏆 Checkmate! ${gameState.winner} wins!`;
        statusEl.className = 'status checkmate';
        isAutoPlaying = false;
    } else if (gameState.check) {
        statusEl.textContent = '⚠️ Check!';
        statusEl.className = 'status check';
    } else if (!statusEl.className.includes('thinking') && !statusEl.className.includes('stockfish')) {
        statusEl.textContent = '';
        statusEl.className = 'status';
    }
    
    document.getElementById('whiteCaptured').textContent = 
        gameState.captured.white.map(p => PIECES[p]).join(' ');
    document.getElementById('blackCaptured').textContent = 
        gameState.captured.black.map(p => PIECES[p]).join(' ');
}

fetchState();
</script>

</body>
</html>
"""


# ==================== API Routes ====================

@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/api/state')
def get_state():
    state = game.get_game_state()
    state["settings"] = game_settings
    return jsonify(state)


@app.route('/api/moves')
def get_moves():
    row = int(request.args.get('row', 0))
    col = int(request.args.get('col', 0))
    moves = game.get_valid_moves(row, col)
    return jsonify({"moves": moves})


@app.route('/api/move', methods=['POST'])
def make_move():
    data = request.json
    result = game.make_move(
        data['from_row'], data['from_col'],
        data['to_row'], data['to_col']
    )
    return jsonify(result)


@app.route('/api/start', methods=['POST'])
def start_game():
    """Start a new game with specified player types."""
    global game, game_settings
    data = request.json or {}
    
    game_settings["white_player"] = data.get("white_player", "player")
    game_settings["black_player"] = data.get("black_player", "stockfish")
    game_settings["commentary"] = []
    
    game = ChessGame()
    return jsonify({"success": True, "settings": game_settings})


@app.route('/api/reset', methods=['POST'])
def reset():
    global game
    game = ChessGame()
    return jsonify({"success": True})


@app.route('/api/bot_move', methods=['POST'])
def bot_move():
    """
    Unified endpoint for bot moves (Stockfish or LLM).
    """
    bot_type = request.args.get('type', 'stockfish')
    state = game.get_game_state()
    
    if state["gameOver"]:
        return jsonify({"success": False, "error": "Game is over"})
    
    try:
        if bot_type == "stockfish":
            # Use Stockfish API
            sf_client = get_stockfish_client()
            result = sf_client.get_best_move(game)
            
            if result.get("success") and result.get("from") and result.get("to"):
                move_result = game.make_move(
                    result["from"][0], result["from"][1],
                    result["to"][0], result["to"][1]
                )
                move_result["move"] = {"from": result["from"], "to": result["to"]}
                move_result["text"] = result.get("text", "")
                move_result["eval"] = result.get("eval", 0)
                return jsonify(move_result)
            else:
                return jsonify({"success": False, "error": result.get("error", "No move found")})
        
        elif bot_type == "llm":
            # Use LLM
            llm_client = get_chess_client()
            selected = llm_client.get_llm_move(game)
            
            if selected:
                move_result = game.make_move(
                    selected["from"][0], selected["from"][1],
                    selected["to"][0], selected["to"][1]
                )
                move_result["move"] = selected
                move_result["text"] = f"LLM plays {selected.get('san', '')}"
                return jsonify(move_result)
            else:
                return jsonify({"success": False, "error": "LLM failed to find move"})
        
        else:
            return jsonify({"success": False, "error": f"Unknown bot type: {bot_type}"})
            
    except Exception as e:
        print(f"Bot move error: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/eval')
def get_eval():
    """Get position evaluation from Stockfish."""
    try:
        sf_client = get_stockfish_client()
        result = sf_client.get_position_eval(game)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/api/settings')
def get_settings():
    """Get current game settings."""
    return jsonify(game_settings)


if __name__ == '__main__':
    print("=" * 50)
    print("  CHESS ARENA - Lingu67")
    print("=" * 50)
    print("Players: Human, Stockfish 17, LLM")
    print("Server: http://localhost:7861")
    print("=" * 50)
    app.run(host='0.0.0.0', port=7861, debug=False)
