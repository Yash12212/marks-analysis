from flask import Flask, request, render_template_string, redirect, url_for, jsonify
import math
import random
import copy
import time
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)
app.jinja_env.globals.update(enumerate=enumerate)

# ----- Global Configurable Board Settings -----
board_m = 15         # Number of rows (m)
board_n = 15         # Number of columns (n)
win_condition = 5    # Number in a row needed to win

# ----- Device Setup & Optimizer -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Global Training Progress Variables & Lock -----
training_progress = {
    'running': False,
    'progress': 0,       # number of steps completed (self-play games + training batches)
    'total': 0,          # total steps (self-play games + training batches)
    'estimated_remaining': 'N/A',
    'phase': 'idle'      # can be "self-play", "training", or "idle"
}
training_message = ""
progress_lock = threading.Lock()

# ----- Global Training Statistics -----
training_stats = {
    "total_games": 0,
    "player_1_wins": 0,
    "player_2_wins": 0,
    "draws": 0,
    "average_moves": 0.0
}

# ----- Simple Evaluation Cache -----
evaluation_cache = {}

# ----- Game Logic Classes (Optimized for Large Boards) -----
class MNKGame:
    """
    Represents an m,n,k-game.
    Board cells: 0 = empty, 1 = player 1, -1 = player 2.
    Uses a NumPy array for faster operations.
    """
    def __init__(self, m, n, k):
        self.m = m  # rows
        self.n = n  # columns
        self.k = k  # number in a row needed to win
        self.board = np.zeros((m, n), dtype=int)
        self.current_player = 1  # player 1 starts

    def clone(self):
        new_game = MNKGame(self.m, self.n, self.k)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        return new_game

    def get_possible_moves(self):
        # Use a neighborhood heuristic: if the board isn’t empty, consider only cells near existing pieces.
        if not np.any(self.board != 0):
            # For an empty board, start from the center.
            return [(self.m // 2, self.n // 2)]
        moves = set()
        indices = np.argwhere(self.board != 0)
        for (i, j) in indices:
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ni = i + di
                    nj = j + dj
                    if 0 <= ni < self.m and 0 <= nj < self.n and self.board[ni, nj] == 0:
                        moves.add((ni, nj))
        if not moves:  # Fallback in the unlikely event
            moves = {(i, j) for i in range(self.m) for j in range(self.n) if self.board[i, j] == 0}
        return list(moves)

    def make_move(self, move):
        i, j = move
        if self.board[i, j] != 0:
            raise ValueError("Invalid move: Cell is not empty.")
        self.board[i, j] = self.current_player
        win = self.check_win(i, j, self.current_player)
        self.current_player *= -1  # switch player
        return win

    def check_win(self, row, col, player):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            # Check in the (dr, dc) direction.
            r, c = row + dr, col + dc
            while 0 <= r < self.m and 0 <= c < self.n and self.board[r, c] == player:
                count += 1
                r += dr
                c += dc
            # Check in the opposite (-dr, -dc) direction.
            r, c = row - dr, col - dc
            while 0 <= r < self.m and 0 <= c < self.n and self.board[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            if count >= self.k:
                return True
        return False

    def is_draw(self):
        return not np.any(self.board == 0)

    def get_winner(self):
        for i in range(self.m):
            for j in range(self.n):
                if self.board[i, j] != 0:
                    if self.check_win(i, j, self.board[i, j]):
                        return self.board[i, j]
        if self.is_draw():
            return 0
        return None

# ----- Neural Network for Board Evaluation (Flexible to Board Size) -----
class MNKNet(nn.Module):
    def __init__(self, board_size):
        super(MNKNet, self).__init__()
        self.m, self.n = board_size
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Policy head
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * self.m * self.n, self.m * self.n)
        # Value head
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(self.m * self.n, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Policy head
        p = F.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        p = F.softmax(p, dim=1)
        # Value head
        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        return p, v

# Instantiate network with current board size.
net = MNKNet((board_m, board_n)).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

def evaluate_board(board_state):
    """
    Evaluate the board state using the neural network.
    Uses caching to speed up repeated evaluations.
    board_state: NumPy array representing the board.
    Returns: (policy_vector, value_estimate)
    """
    board_key = tuple(board_state.flatten().tolist())
    if board_key in evaluation_cache:
        return evaluation_cache[board_key]
    board_tensor = torch.tensor(board_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    net.eval()
    with torch.no_grad():
        policy, value = net(board_tensor)
    policy = policy.cpu().numpy()[0]  # shape: (m*n,)
    value = value.item()
    evaluation_cache[board_key] = (policy, value)
    return policy, value

# ----- Neural Network Integrated MCTS -----
class NN_MCTSNode:
    def __init__(self, game_state, parent=None, move=None, prior=1.0):
        self.game_state = game_state  # instance of MNKGame
        self.parent = parent
        self.move = move  # move that led to this node (None for root)
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.untried_moves = game_state.get_possible_moves()
        self.prior = prior  # prior probability from the NN

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c_param=1.4):
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            Q = child.wins / child.visits if child.visits > 0 else 0
            U = c_param * child.prior * math.sqrt(self.visits) / (1 + child.visits)
            score = Q + U
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

def nn_mcts(root, iterations):
    for _ in range(iterations):
        node = root
        state = root.game_state.clone()

        # --- Selection: traverse tree using best_child ---
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
            state.make_move(node.move)

        # --- Expansion: expand one untried move ---
        if node.untried_moves:
            move = random.choice(node.untried_moves)
            state.make_move(move)
            policy_vector, _ = evaluate_board(state.board)
            move_index = move[0] * state.n + move[1]
            child_prior = policy_vector[move_index] if move_index < len(policy_vector) else 1.0
            child_node = NN_MCTSNode(state.clone(), parent=node, move=move, prior=child_prior)
            node.children.append(child_node)
            node.untried_moves.remove(move)
            node = child_node

        # --- Simulation: terminal check or value network evaluation ---
        terminal = state.get_winner()
        if terminal is not None:
            if terminal == 0:
                result = 0.0
            else:
                result = 1.0 if terminal == -state.current_player else -1.0
        else:
            _, value_estimate = evaluate_board(state.board)
            result = value_estimate

        # --- Backpropagation ---
        reward = result
        while node is not None:
            node.visits += 1
            node.wins += reward
            reward = -reward  # invert reward for opponent
            node = node.parent
    return root

def best_move_nn(game, iterations=1000):
    root = NN_MCTSNode(game.clone(), prior=1.0)
    nn_mcts(root, iterations)
    if not root.children:
        return None
    best_child_node = max(root.children, key=lambda n: n.visits)
    return best_child_node.move

def get_move_and_policy(game, iterations, temperature=1.0):
    """
    Run MCTS from the current game state and return:
      - move: selected move (tuple)
      - policy: an (m*n)-element vector derived from visit counts.
    """
    root = NN_MCTSNode(game.clone(), prior=1.0)
    nn_mcts(root, iterations)
    policy = [0] * (game.m * game.n)
    total_visits = 0
    for child in root.children:
        index = child.move[0] * game.n + child.move[1]
        policy[index] = child.visits
        total_visits += child.visits
    if total_visits > 0:
        policy = [x / total_visits for x in policy]
    else:
        policy = [1 / (game.m * game.n)] * (game.m * game.n)
    policy_np = np.array(policy)
    if temperature == 0:
        move_index = int(np.argmax(policy_np))
    else:
        adjusted = policy_np ** (1 / temperature)
        if np.sum(adjusted) == 0:
            adjusted = np.ones_like(adjusted)
        adjusted = adjusted / np.sum(adjusted)
        move_index = int(np.random.choice(len(adjusted), p=adjusted))
    move = (move_index // game.n, move_index % game.n)
    return move, policy

# ----- Self-Play Training Functions -----
def self_play_game(mcts_iterations, temperature=1.0):
    """
    Simulate one self-play game using NN-MCTS.
    Returns training examples as a list of (board, policy, outcome) tuples.
    Also updates training statistics.
    """
    global training_stats
    game_instance = MNKGame(board_m, board_n, win_condition)
    examples = []
    move_count = 0

    while True:
        move_count += 1
        board_copy = game_instance.board.copy()
        current_player = game_instance.current_player

        if not game_instance.get_possible_moves():
            examples.append((board_copy.copy(), [1 / (game_instance.m * game_instance.n)] * (game_instance.m * game_instance.n), 0))
            break

        move, pi = get_move_and_policy(game_instance, mcts_iterations, temperature)
        examples.append((board_copy.copy(), pi, current_player))
        if game_instance.make_move(move):
            break
        if game_instance.is_draw():
            break

    # Determine winner and update stats
    winner = game_instance.get_winner()
    training_stats["total_games"] += 1
    training_stats["average_moves"] = ((training_stats["average_moves"] * (training_stats["total_games"] - 1)) + move_count) / training_stats["total_games"]

    if winner == 1:
        training_stats["player_1_wins"] += 1
    elif winner == -1:
        training_stats["player_2_wins"] += 1
    else:
        training_stats["draws"] += 1

    updated_examples = []
    for board, pi, player in examples:
        if winner is None or winner == 0:
            z = 0
        else:
            z = 1 if player == winner else -1
        updated_examples.append((board, pi, z))
    return updated_examples

def train_network(training_data, epochs, batch_size, start_time):
    """
    Train the network on the collected training_data.
    Each example is (board, policy, outcome).
    """
    global training_progress
    net.train()
    num_examples = len(training_data)
    num_batches = (num_examples + batch_size - 1) // batch_size
    total_training_steps = epochs * num_batches
    with progress_lock:
        training_progress['total'] += total_training_steps
    for epoch in range(epochs):
        random.shuffle(training_data)
        for i in range(0, num_examples, batch_size):
            batch = training_data[i:i+batch_size]
            boards = [x[0] for x in batch]
            target_pis = [x[1] for x in batch]
            target_vs = [x[2] for x in batch]
            board_tensor = torch.tensor(np.array(boards), dtype=torch.float32).unsqueeze(1).to(device)
            target_pi_tensor = torch.tensor(target_pis, dtype=torch.float32).to(device)
            target_v_tensor = torch.tensor(target_vs, dtype=torch.float32).to(device)
            pred_pi, pred_v = net(board_tensor)
            loss_pi = -torch.mean(torch.sum(target_pi_tensor * torch.log(pred_pi + 1e-8), dim=1))
            loss_v = torch.mean((pred_v.squeeze() - target_v_tensor) ** 2)
            loss = loss_pi + loss_v
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with progress_lock:
                training_progress['progress'] += 1
                prog = training_progress['progress']
                tot = training_progress['total']
            elapsed = time.time() - start_time
            avg_time = elapsed / prog if prog > 0 else 0
            remaining = (tot - prog) * avg_time
            with progress_lock:
                training_progress['estimated_remaining'] = f"{remaining:.1f} sec"
    net.eval()

def run_selfplay_training(num_games, mcts_iterations_sp, training_epochs, batch_size):
    """
    Run self-play training by generating training examples and then training the network.
    """
    global training_progress, training_message
    start_time = time.time()
    training_examples = []
    with progress_lock:
        training_progress['phase'] = 'self-play'
        training_progress['total'] = num_games
        training_progress['progress'] = 0
    for game_index in range(num_games):
        examples = self_play_game(mcts_iterations_sp, temperature=1.0)
        training_examples.extend(examples)
        with progress_lock:
            training_progress['progress'] = game_index + 1
        elapsed = time.time() - start_time
        remaining = (num_games - (game_index + 1)) * (elapsed / (game_index + 1)) if game_index > 0 else 0
        with progress_lock:
            training_progress['estimated_remaining'] = f"{remaining:.1f} sec"
    with progress_lock:
        training_progress['phase'] = 'training'
    num_examples = len(training_examples)
    num_batches = (num_examples + batch_size - 1) // batch_size
    total_training_steps = training_epochs * num_batches
    with progress_lock:
        training_progress['total'] += total_training_steps
    train_network(training_examples, training_epochs, batch_size, start_time)
    with progress_lock:
        training_progress['progress'] = training_progress['total']
        training_progress['estimated_remaining'] = "0.0 sec"
        training_progress['running'] = False
        training_progress['phase'] = 'idle'
    training_message = f"Self-play training complete! Generated {len(training_examples)} examples."

# ----- Global Game State & Settings for Flask -----
game = MNKGame(board_m, board_n, win_condition)
game_over = False
winner_message = ""
ai_iterations = 2000  # Default AI MCTS iterations for human play

# ----- Flask Routes & Template -----
TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <title>AlphaGo-like m,n,k Game</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <!-- Bootstrap CSS via CDN -->
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    <style>
      body { background-color: #f8f9fa; }
      .card { margin-bottom: 20px; }
      table { margin: 0 auto; }
      td { width: 40px; height: 40px; vertical-align: middle; font-size: 18px; }
    </style>
  </head>
  <body>
    <div class="container mt-4">
      <h1 class="text-center mb-4">AlphaGo-like m,n,k Game</h1>
      {% if training_message %}
        <div class="alert alert-success text-center">{{ training_message }}</div>
      {% endif %}
      <div class="row">
        <!-- Left Panel: Settings, Training Stats & Self-Play Training -->
        <div class="col-md-3">
          <div class="card">
            <div class="card-header">Game Settings</div>
            <div class="card-body">
              <form method="post" action="{{ url_for('reset') }}">
                <button type="submit" class="btn btn-danger btn-block">Reset Game</button>
              </form>
              <hr>
              <form method="post" action="{{ url_for('update_settings') }}">
                <div class="form-group">
                  <label for="ai_iterations">AI MCTS Iterations</label>
                  <input type="number" class="form-control" id="ai_iterations" name="ai_iterations" value="{{ ai_iterations }}" min="100" step="100">
                </div>
                <div class="form-group">
                  <label for="board_m">Board Rows (m)</label>
                  <input type="number" class="form-control" id="board_m" name="board_m" value="{{ board_m }}" min="3" step="1">
                </div>
                <div class="form-group">
                  <label for="board_n">Board Columns (n)</label>
                  <input type="number" class="form-control" id="board_n" name="board_n" value="{{ board_n }}" min="3" step="1">
                </div>
                <div class="form-group">
                  <label for="win_condition">Win Condition (k)</label>
                  <input type="number" class="form-control" id="win_condition" name="win_condition" value="{{ win_condition }}" min="3" step="1">
                </div>
                <button type="submit" class="btn btn-primary btn-block">Update Settings</button>
              </form>
            </div>
          </div>
          <div class="card">
            <div class="card-header">Training Statistics</div>
            <div class="card-body">
              <ul class="list-group">
                <li class="list-group-item">Total Games: <span id="total_games">0</span></li>
                <li class="list-group-item">Player 1 Wins: <span id="player_1_wins">0</span></li>
                <li class="list-group-item">Player 2 Wins: <span id="player_2_wins">0</span></li>
                <li class="list-group-item">Draws: <span id="draws">0</span></li>
                <li class="list-group-item">Avg. Moves/Game: <span id="average_moves">0</span></li>
              </ul>
            </div>
          </div>
          <div class="card">
            <div class="card-header">Self-Play Training</div>
            <div class="card-body">
              <form method="post" action="{{ url_for('selfplay_train') }}">
                <div class="form-group">
                  <label for="num_games">Number of Self-Play Games</label>
                  <input type="number" class="form-control" id="num_games" name="num_games" value="10" min="1" step="1">
                </div>
                <div class="form-group">
                  <label for="mcts_iterations_sp">MCTS Iterations (Self-Play)</label>
                  <input type="number" class="form-control" id="mcts_iterations_sp" name="mcts_iterations_sp" value="100" min="10" step="10">
                </div>
                <div class="form-group">
                  <label for="training_epochs">Training Epochs</label>
                  <input type="number" class="form-control" id="training_epochs" name="training_epochs" value="10" min="1" step="1">
                </div>
                <div class="form-group">
                  <label for="batch_size">Batch Size</label>
                  <input type="number" class="form-control" id="batch_size" name="batch_size" value="32" min="1" step="1">
                </div>
                <button type="submit" class="btn btn-success btn-block">Start Self-Play Training</button>
              </form>
            </div>
          </div>
          <!-- Training Progress Card -->
          <div class="card" id="training-progress-card" style="display: none;">
            <div class="card-header">Training Progress</div>
            <div class="card-body">
              <div class="progress">
                <div id="progress-bar" class="progress-bar" role="progressbar" style="width: 0%">0%</div>
              </div>
              <p id="progress-text" class="mt-2">Estimated time remaining: N/A</p>
              <p id="phase-text" class="mt-1">Phase: idle</p>
            </div>
          </div>
        </div>
        <!-- Middle Panel: Game Board -->
        <div class="col-md-6">
          <div class="card">
            <div class="card-header text-center">Game Board</div>
            <div class="card-body">
              <table class="table table-bordered text-center">
                {% for i, row in enumerate(board) %}
                  <tr>
                  {% for j, cell in enumerate(row) %}
                    <td>
                      {% if cell == 0 and not game_over %}
                        <form method="post" action="{{ url_for('move') }}">
                          <input type="hidden" name="row" value="{{ i }}">
                          <input type="hidden" name="col" value="{{ j }}">
                          <button type="submit" class="btn btn-link" style="font-size: 18px;">&nbsp;</button>
                        </form>
                      {% else %}
                        {% if cell == 1 %} X {% elif cell == -1 %} O {% endif %}
                      {% endif %}
                    </td>
                  {% endfor %}
                  </tr>
                {% endfor %}
              </table>
            </div>
          </div>
        </div>
        <!-- Right Panel: Game Info -->
        <div class="col-md-3">
          <div class="card">
            <div class="card-header">Game Info</div>
            <div class="card-body">
              {% if message %}
                <div class="alert alert-info">{{ message }}</div>
              {% endif %}
              <p><strong>Current Turn:</strong> {% if game.current_player == 1 %}X{% else %}O{% endif %}</p>
              <p><strong>AI Iterations:</strong> {{ ai_iterations }}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
    <!-- JavaScript to Poll Training Progress & Stats -->
    <script>
      function updateProgress() {
          fetch("/progress")
            .then(response => response.json())
            .then(data => {
               if(data.total > 0) {
                 document.getElementById("training-progress-card").style.display = "block";
                 let prog = data.progress;
                 let tot = data.total;
                 let percent = tot > 0 ? Math.round((prog / tot) * 100) : 0;
                 document.getElementById("progress-bar").style.width = percent + "%";
                 document.getElementById("progress-bar").innerText = percent + "%";
                 document.getElementById("progress-text").innerText = "Estimated time remaining: " + data.estimated_remaining;
                 document.getElementById("phase-text").innerText = "Phase: " + data.phase;
               } else {
                 document.getElementById("training-progress-card").style.display = "none";
               }
            });
      }
      function updateStats() {
          fetch('/get_training_stats')
              .then(response => response.json())
              .then(data => {
                  document.getElementById('total_games').textContent = data.total_games;
                  document.getElementById('player_1_wins').textContent = data.player_1_wins;
                  document.getElementById('player_2_wins').textContent = data.player_2_wins;
                  document.getElementById('draws').textContent = data.draws;
                  document.getElementById('average_moves').textContent = data.average_moves.toFixed(1);
              })
              .catch(error => console.error('Error fetching stats:', error));
      }
      setInterval(updateProgress, 1000);
      setInterval(updateStats, 5000);
      updateProgress();
      updateStats();
    </script>
  </body>
</html>
"""

@app.route("/")
def index():
    global game, game_over, winner_message, ai_iterations, training_message, board_m, board_n, win_condition
    message = winner_message if game_over else ("Your turn (X)" if game.current_player == 1 else "AI is thinking...")
    return render_template_string(
        TEMPLATE, 
        board=game.board.tolist(), 
        game_over=game_over, 
        message=message, 
        ai_iterations=ai_iterations, 
        training_message=training_message,
        board_m=board_m, 
        board_n=board_n, 
        win_condition=win_condition,
        game=game
    )

@app.route("/move", methods=["POST"])
def move():
    global game, game_over, winner_message, ai_iterations
    if game_over:
        return redirect(url_for("index"))
    try:
        i = int(request.form.get("row"))
        j = int(request.form.get("col"))
    except (ValueError, TypeError):
        return redirect(url_for("index"))
    if (i, j) not in game.get_possible_moves():
        return redirect(url_for("index"))
    # Human move (Player 1)
    if game.make_move((i, j)):
        game_over = True
        winner_message = "Player X wins!"
        return redirect(url_for("index"))
    if game.is_draw():
        game_over = True
        winner_message = "It's a draw!"
        return redirect(url_for("index"))
    # AI move (Player 2) using NN-MCTS
    if game.current_player == -1:
        ai_move = best_move_nn(game, iterations=ai_iterations)
        if ai_move is not None:
            if game.make_move(ai_move):
                game_over = True
                winner_message = "Player O wins!"
            elif game.is_draw():
                game_over = True
                winner_message = "It's a draw!"
    return redirect(url_for("index"))

@app.route("/reset", methods=["POST"])
def reset():
    global game, game_over, winner_message
    game = MNKGame(board_m, board_n, win_condition)
    game_over = False
    winner_message = ""
    return redirect(url_for("index"))

@app.route("/update_settings", methods=["POST"])
def update_settings():
    global ai_iterations, board_m, board_n, win_condition, game, net, optimizer, evaluation_cache
    try:
        ai_iterations = int(request.form.get("ai_iterations", ai_iterations))
    except (ValueError, TypeError):
        pass
    try:
        new_board_m = int(request.form.get("board_m", board_m))
        new_board_n = int(request.form.get("board_n", board_n))
        new_win_condition = int(request.form.get("win_condition", win_condition))
        if new_board_m != board_m or new_board_n != board_n or new_win_condition != win_condition:
            board_m = new_board_m
            board_n = new_board_n
            win_condition = new_win_condition
            # Reinitialize game and network
            game = MNKGame(board_m, board_n, win_condition)
            evaluation_cache.clear()
            net = MNKNet((board_m, board_n)).to(device)
            optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    except (ValueError, TypeError):
        pass
    return redirect(url_for("index"))

@app.route("/selfplay_train", methods=["POST"])
def selfplay_train():
    global training_message, training_progress
    with progress_lock:
        if training_progress['running']:
            training_message = "Training is already running."
            return redirect(url_for("index"))
        training_progress['running'] = True
    try:
        num_games = int(request.form.get("num_games"))
        mcts_iterations_sp = int(request.form.get("mcts_iterations_sp"))
        training_epochs = int(request.form.get("training_epochs"))
        batch_size = int(request.form.get("batch_size"))
    except (ValueError, TypeError):
        training_message = "Invalid training parameters."
        with progress_lock:
            training_progress['running'] = False
        return redirect(url_for("index"))
    thread = threading.Thread(target=run_selfplay_training, args=(num_games, mcts_iterations_sp, training_epochs, batch_size))
    thread.start()
    training_message = "Training started in background."
    return redirect(url_for("index"))

@app.route("/progress")
def progress():
    with progress_lock:
        progress_copy = training_progress.copy()
    return jsonify(progress_copy)

@app.route("/get_training_stats")
def get_training_stats():
    """Return training stats in JSON format."""
    return jsonify(training_stats)

if __name__ == "__main__":
    app.run(debug=True)
