import logging
import random
import threading
import time
from collections import deque
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

global_lock = threading.Lock()
training_thread = None
TRAINING_SLEEP = 0.0
training_stop_event = threading.Event()
evaluation_thread = None
evaluation_stop_event = threading.Event()

def format_time(seconds):
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} sec"
    elif seconds < 3600:
        minutes = seconds // 60
        rem_sec = seconds % 60
        return f"{minutes} min {rem_sec} sec" if rem_sec else f"{minutes} min"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        rem_sec = seconds % 60
        result = f"{hours} hr"
        if minutes:
            result += f" {minutes} min"
        if rem_sec:
            result += f" {rem_sec} sec"
        return result

class TicTacToeGeneral:
    def __init__(self, rows=3, cols=3, k=3):
        self.rows = rows
        self.cols = cols
        self.k = k
        self.board = [' '] * (self.rows * self.cols)
        self.current_winner = None

    def reset(self, rows=None, cols=None, k=None):
        if rows is not None:
            self.rows = rows
        if cols is not None:
            self.cols = cols
        if k is not None:
            self.k = k
        self.board = [' '] * (self.rows * self.cols)
        self.current_winner = None

    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.check_winner(square, letter):
                self.current_winner = letter
            return True
        return False

    def check_winner(self, square, letter):
        row = square // self.cols
        col = square % self.cols

        def count_direction(delta_row, delta_col):
            count = 0
            r, c = row, col
            while 0 <= r < self.rows and 0 <= c < self.cols and self.board[r * self.cols + c] == letter:
                count += 1
                r += delta_row
                c += delta_col
            return count

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = count_direction(dr, dc) + count_direction(-dr, -dc) - 1
            if count >= self.k:
                return True
        return False

    def get_empty_squares(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def is_board_full(self):
        return ' ' not in self.board

    def get_game_state(self, current_player=None):
        state = f"{self.rows}-{self.cols}-" + ''.join(self.board)
        if current_player is not None:
            state += f"-{current_player}"
        return state

class QLearningAgent:
    def __init__(self, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.q_table = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.lock = threading.Lock()

    def get_q_value(self, state, action):
        with self.lock:
            return self.q_table.get((state, action), 0.0)

    def select_action(self, game, current_player=None):
        state = game.get_game_state(current_player)
        empty_squares = game.get_empty_squares()
        if not empty_squares:
            return None
        if random.random() < self.epsilon:
            return random.choice(empty_squares)
        q_values = [self.get_q_value(state, a) for a in empty_squares]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(empty_squares, q_values) if q == max_q]
        return random.choice(best_actions)

    def update_q_table(self, state, action, reward, next_state, done):
        with self.lock:
            old_q = self.q_table.get((state, action), 0.0)
            if not done:
                parts = next_state.split('-')
                board_str = parts[2] if len(parts) >= 3 else ""
                possible_actions = [i for i, spot in enumerate(board_str) if spot == ' ']
                max_next_q = max([self.q_table.get((next_state, a), 0.0) for a in possible_actions], default=0.0)
                target = reward + self.gamma * max_next_q
            else:
                target = reward
            self.q_table[(state, action)] = old_q + self.alpha * (target - old_q)

def create_model(rows, cols, k):
    return {
        "agent": QLearningAgent(),
        "game": TicTacToeGeneral(rows, cols, k),
        "training_stats": {"X": 0, "O": 0, "draw": 0, "episodes": 0},
        "latest_results": deque(maxlen=1000),
        "training_progress": {
            "session_total": 0,
            "session_completed": 0,
            "start_time": None
        },
        "evaluation_stats": {"win": 0, "loss": 0, "draw": 0, "games": 0},
        "evaluation_latest": deque(maxlen=100),
        "evaluation_progress": {
            "session_total": 0,
            "session_completed": 0,
            "start_time": None
        }
    }

current_model_key = (3, 3, 3)
models = {}
models[current_model_key] = create_model(*current_model_key)

def self_play_episode():
    model = models[current_model_key]
    current_game = model["game"]
    env = TicTacToeGeneral(current_game.rows, current_game.cols, current_game.k)
    agent_instance = model["agent"]
    current_player = 'X'
    last_move = {'X': None, 'O': None}

    while True:
        current_state = env.get_game_state(current_player)
        action = agent_instance.select_action(env, current_player)
        if action is None:
            break
        valid_move = env.make_move(action, current_player)
        if not valid_move:
            continue
        next_player = 'O' if current_player == 'X' else 'X'
        next_state = env.get_game_state(next_player)
        if env.current_winner:
            agent_instance.update_q_table(current_state, action, 1, next_state, True)
            if last_move[next_player] is not None:
                opponent_state, opponent_action = last_move[next_player]
                agent_instance.update_q_table(opponent_state, opponent_action, -1, next_state, True)
            return current_player
        elif env.is_board_full():
            agent_instance.update_q_table(current_state, action, 0, next_state, True)
            return "draw"
        else:
            agent_instance.update_q_table(current_state, action, 0, next_state, False)
        last_move[current_player] = (current_state, action)
        current_player = next_player
    return "draw"

def train_agent(episodes=1000):
    global training_thread
    model = models[current_model_key]
    start_time = model["training_progress"].get("start_time", time.time())
    if start_time is None:
        start_time = time.time()
        with global_lock:
            model["training_progress"]["start_time"] = start_time

    try:
        for _ in range(episodes):
            if training_stop_event.is_set():
                logger.info("Training cancelled.")
                break
            result = self_play_episode()
            with global_lock:
                model["training_stats"][result] += 1
                model["training_stats"]["episodes"] += 1
                model["latest_results"].append(result)
                model["training_progress"]["session_completed"] += 1
    except Exception as e:
        logger.exception("Exception during training: %s", e)
    finally:
        training_thread = None

def evaluate_game():
    model = models[current_model_key]
    game = TicTacToeGeneral(model["game"].rows, model["game"].cols, model["game"].k)
    agent_instance = model["agent"]
    current_player = 'X'
    original_epsilon = agent_instance.epsilon
    agent_instance.epsilon = 0.0

    while True:
        if current_player == 'X':
            state = game.get_game_state(current_player)
            action = agent_instance.select_action(game, current_player)
        else:
            possible_moves = game.get_empty_squares()
            action = random.choice(possible_moves) if possible_moves else None

        if action is None:
            break
        valid_move = game.make_move(action, current_player)
        if not valid_move:
            continue
        if game.current_winner:
            winner = current_player
            break
        elif game.is_board_full():
            winner = "draw"
            break
        current_player = 'O' if current_player == 'X' else 'X'
    agent_instance.epsilon = original_epsilon
    return winner

def evaluate_agent(episodes):
    model = models[current_model_key]
    start_time = model["evaluation_progress"].get("start_time", time.time())
    if start_time is None:
        start_time = time.time()
        with global_lock:
            model["evaluation_progress"]["start_time"] = start_time

    try:
        for _ in range(episodes):
            if evaluation_stop_event.is_set():
                logger.info("Evaluation cancelled.")
                break
            result = evaluate_game()
            with global_lock:
                if result == "X":
                    model["evaluation_stats"]["win"] += 1
                elif result == "O":
                    model["evaluation_stats"]["loss"] += 1
                else:
                    model["evaluation_stats"]["draw"] += 1
                model["evaluation_stats"]["games"] += 1
                model["evaluation_latest"].append(result)
                model["evaluation_progress"]["session_completed"] += 1
    except Exception as e:
        logger.exception("Exception during evaluation: %s", e)
    finally:
        evaluation_thread = None

@app.route("/", methods=["GET", "POST"])
def index():
    global current_model_key
    model = models[current_model_key]
    game = model["game"]
    message = ""
    train_message = ""
    eval_message = ""

    if request.method == "POST":
        action = request.form.get("action")
        if action == "reset_stats":
            with global_lock:
                model["training_stats"] = {"X": 0, "O": 0, "draw": 0, "episodes": 0}
                model["training_progress"] = {"session_total": 0, "session_completed": 0, "start_time": None}
                model["latest_results"].clear()
                model["evaluation_stats"] = {"win": 0, "loss": 0, "draw": 0, "games": 0}
                model["evaluation_progress"] = {"session_total": 0, "session_completed": 0, "start_time": None}
                model["evaluation_latest"].clear()
            train_message = "Training and Evaluation statistics have been reset."
        elif action == "set_board":
            if training_thread is not None and training_thread.is_alive() or (evaluation_thread is not None and evaluation_thread.is_alive()):
                message = "Cannot change board settings while training or evaluation is in progress."
            else:
                try:
                    rows = int(request.form.get("rows", 3))
                    cols = int(request.form.get("cols", 3))
                    k = int(request.form.get("k", 3))
                except (TypeError, ValueError):
                    rows, cols, k = 3, 3, 3
                new_key = (rows, cols, k)
                if new_key not in models:
                    models[new_key] = create_model(rows, cols, k)
                current_model_key = new_key
                model["game"].reset(rows=rows, cols=cols, k=k)
                message = f"Board settings updated: {rows}x{cols} with {k} in a row to win. Board reset."
        elif action == "reset_board":
            if training_thread is not None and training_thread.is_alive() or (evaluation_thread is not None and evaluation_thread.is_alive()):
                message = "Cannot reset board while training or evaluation is in progress."
            else:
                game.reset()
                message = "Board reset."
        elif action == "move":
            if training_thread is not None and training_thread.is_alive() or (evaluation_thread is not None and evaluation_thread.is_alive()):
                message = "Moves are disabled during training or evaluation."
            else:
                if game.current_winner or game.is_board_full():
                    message = "Game over. Please reset the board."
                else:
                    try:
                        square = int(request.form.get("square"))
                    except (TypeError, ValueError):
                        square = None
                    if square is None or square < 0 or square >= len(game.board):
                        message = "Invalid move. Square index out of range."
                    elif game.board[square] != " ":
                        message = "Square already taken."
                    else:
                        game.make_move(square, "X")
                        if not game.current_winner and not game.is_board_full():
                            opponent_move = model["agent"].select_action(game, "O")
                            if opponent_move is not None:
                                game.make_move(opponent_move, "O")
                        if game.current_winner:
                            message = f"Winner: {game.current_winner}"
                        elif game.is_board_full():
                            message = "It's a draw!"

    with global_lock:
        total_episodes = model["training_stats"]["episodes"]
        pct_X = round((model["training_stats"]["X"] / total_episodes * 100), 2) if total_episodes > 0 else 0
        pct_O = round((model["training_stats"]["O"] / total_episodes * 100), 2) if total_episodes > 0 else 0
        pct_draw = round((model["training_stats"]["draw"] / total_episodes * 100), 2) if total_episodes > 0 else 0

        latest_total = len(model["latest_results"])
        latest_counts = {"X": 0, "O": 0, "draw": 0}
        for res in model["latest_results"]:
            latest_counts[res] += 1
        latest_pct_X = round(latest_counts["X"] / latest_total * 100, 2) if latest_total > 0 else 0
        latest_pct_O = round(latest_counts["O"] / latest_total * 100, 2) if latest_total > 0 else 0
        latest_pct_draw = round(latest_counts["draw"] / latest_total * 100, 2) if latest_total > 0 else 0

        eval_total = model["evaluation_stats"]["games"] if model["evaluation_stats"]["games"] > 0 else 1
        eval_win_pct = round(model["evaluation_stats"]["win"] / eval_total * 100, 2)
        eval_loss_pct = round(model["evaluation_stats"]["loss"] / eval_total * 100, 2)
        eval_draw_pct = round(model["evaluation_stats"]["draw"] / eval_total * 100, 2)

        latest_eval_total = len(model["evaluation_latest"])
        latest_eval_counts = {"win": 0, "loss": 0, "draw": 0}
        for res in model["evaluation_latest"]:
            if res == "X":
                latest_eval_counts["win"] += 1
            elif res == "O":
                latest_eval_counts["loss"] += 1
            else:
                latest_eval_counts["draw"] += 1
        latest_eval_win_pct = round(latest_eval_counts["win"] / latest_eval_total * 100, 2) if latest_eval_total > 0 else 0
        latest_eval_loss_pct = round(latest_eval_counts["loss"] / latest_eval_total * 100, 2) if latest_eval_total > 0 else 0
        latest_eval_draw_pct = round(latest_eval_counts["draw"] / latest_eval_total * 100, 2) if latest_eval_total > 0 else 0

        progress = (model["training_progress"]["session_completed"] / model["training_progress"]["session_total"] * 100) if model["training_progress"]["session_total"] > 0 else 0
        eval_progress = (model["evaluation_progress"]["session_completed"] / model["evaluation_progress"]["session_total"] * 100) if model["evaluation_progress"]["session_total"] > 0 else 0


    html = '''
    <!doctype html>
    <html lang="en">
      <head>
        <title>Generalized Tic Tac Toe Agent Console</title>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
        <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
        <style>
          .game-table {
            width: 100%;
            table-layout: fixed;
          }
          td {
            vertical-align: middle;
            padding: 0;
            border: 1px solid #ddd;
            box-sizing: border-box;
            font-size: calc(36px / (( {{game.rows}} + {{game.cols}} ) / 6 )); /* Adjusted scaling factor */
            overflow: hidden; /* Prevent content overflow */
            text-overflow: ellipsis; /* Add ellipsis for text overflow */
          }
          .game-cell-button {
            font-size: 1em;
            width: 100%;
            height: 100%;
            padding: 0;
            border: none;
            background: none;
            text-align: center;
            vertical-align: middle;
          }
          .left-panel { border-right: 1px solid #ddd; }
          .mid-panel { border-right: 1px solid #ddd; border-left: 1px solid #ddd; }
          .card { margin-bottom: 20px; }
        </style>
      </head>
      <body class="bg-light">
        <div class="container mt-3">
          <div class="row">
            <!-- Left Panel: Controls -->
            <div class="col-md-3 left-panel">
              <!-- Training Controls -->
              <div class="card">
                <div class="card-header">
                  <h3>Training Controls</h3>
                </div>
                <div class="card-body">
                  <div id="trainAlert">
                    {% if train_message %}
                    <div class="alert alert-success">{{ train_message }}</div>
                    {% endif %}
                  </div>
                  <div class="form-group">
                    <label for="episodes">Episodes:</label>
                    <input type="number" id="episodes" class="form-control" value="1000" min="1">
                  </div>
                  <button id="trainBtn" class="btn btn-primary btn-block">Train Agent</button>
                  <button id="cancelTrainBtn" class="btn btn-danger btn-block mt-2">Cancel Training</button>
                </div>
              </div>
              <!-- Evaluation Controls -->
              <div class="card">
                <div class="card-header">
                  <h3>Evaluation Controls</h3>
                </div>
                <div class="card-body">
                  <div id="evalAlert">
                    {% if eval_message %}
                    <div class="alert alert-success">{{ eval_message }}</div>
                    {% endif %}
                  </div>
                  <div class="form-group">
                    <label for="evalGames">Number of Games:</label>
                    <input type="number" id="evalGames" class="form-control" value="100" min="1">
                  </div>
                  <button id="evalBtn" class="btn btn-primary btn-block">Evaluate Agent</button>
                  <button id="cancelEvalBtn" class="btn btn-danger btn-block mt-2">Cancel Evaluation</button>
                </div>
              </div>
              <!-- Board Settings with Sliders -->
              <div class="card mt-3">
                <div class="card-header">
                  <h5>Board Settings</h5>
                </div>
                <div class="card-body">
                  <form method="POST">
                    <input type="hidden" name="action" value="set_board">
                    <div class="form-group">
                      <label for="rowsSlider">Rows: <span id="rowsVal">{{ game.rows }}</span></label>
                      <input type="range" class="form-control-range" id="rowsSlider" name="rows" min="3" max="10" value="{{ game.rows }}">
                    </div>
                    <div class="form-group">
                      <label for="colsSlider">Columns: <span id="colsVal">{{ game.cols }}</span></label>
                      <input type="range" class="form-control-range" id="colsSlider" name="cols" min="3" max="10" value="{{ game.cols }}">
                    </div>
                    <div class="form-group">
                      <label for="kSlider">Winning Length (k): <span id="kVal">{{ game.k }}</span></label>
                      <input type="range" class="form-control-range" id="kSlider" name="k" min="3" max="{{ game.rows if game.rows < game.cols else game.cols }}" value="{{ game.k }}">
                    </div>
                    <button type="submit" class="btn btn-primary btn-block">Apply Board Settings</button>
                  </form>
                </div>
              </div>
              <button id="resetStatsBtn" class="btn btn-warning btn-block mt-2">Reset All Stats</button>
            </div>

            <!-- Middle Panel: Game Board and Progress -->
            <div class="col-md-6 mid-panel">
              <div class="card">
                <div class="card-header">
                  <h3>Play Game</h3>
                </div>
                <div class="card-body">
                  {% if message %}
                  <div class="alert alert-info">{{ message }}</div>
                  {% endif %}
                  <form method="POST">
                    <input type="hidden" name="action" value="move">
                    <table class="table table-bordered text-center game-table">
                      {% for i in range(game.rows) %}
                      <tr>
                        {% for j in range(game.cols) %}
                        {% set index = i * game.cols + j %}
                        <td>
                          {% if game.board[index] == ' ' %}
                          <button name="square" value="{{ index }}" class="btn btn-link game-cell-button"> </button>
                          {% else %}
                          <span >{{ game.board[index] }}</span>
                          {% endif %}
                        </td>
                        {% endfor %}
                      </tr>
                      {% endfor %}
                    </table>
                  </form>
                  <form method="POST" class="mt-3">
                    <input type="hidden" name="action" value="reset_board">
                    <button type="submit" class="btn btn-secondary btn-block">Reset Board</button>
                  </form>
                </div>
              </div>
              <!-- Training Progress -->
              <div class="card mt-3">
                <div class="card-header">
                  <h5>Training Progress</h5>
                </div>
                <div class="card-body">
                  <div class="progress">
                    <div id="progressBar" class="progress-bar" role="progressbar" style="width: {{ progress }}%;" aria-valuenow="{{ progress }}" aria-valuemin="0" aria-valuemax="100">{{ progress|round(2) }}%</div>
                  </div>
                  <p id="estimatedTime">Estimated time remaining: <span id="trainingEstimatedTimeStr"></span></p>
                </div>
              </div>
              <!-- Evaluation Progress -->
              <div class="card mt-3">
                <div class="card-header">
                  <h5>Evaluation Progress</h5>
                </div>
                <div class="card-body">
                  <div class="progress">
                    <div id="evalProgressBar" class="progress-bar" role="progressbar" style="width: {{ eval_progress }}%;" aria-valuenow="{{ eval_progress }}" aria-valuemin="0" aria-valuemax="100">{{ eval_progress|round(2) }}%</div>
                  </div>
                  <p id="estimatedEvalTime">Estimated time remaining: <span id="evaluationEstimatedTimeStr"></span></p>
                </div>
              </div>
            </div>

            <!-- Right Panel: Stats -->
            <div class="col-md-3">
              <!-- Cumulative Training Stats -->
              <div class="card">
                <div class="card-header">
                  <h5>Cumulative Training Stats</h5>
                </div>
                <div class="card-body">
                  <p><strong>Total Episodes:</strong> <span id="totalEpisodes">{{ total_episodes }}</span></p>
                  <p><strong>X Wins:</strong> <span id="xWins">{{ training_stats["X"] }}</span> (<span id="pctX">{{ pct_X }}</span>%)</p>
                  <p><strong>O Wins:</strong> <span id="oWins">{{ training_stats["O"] }}</span> (<span id="pctO">{{ pct_O }}</span>%)</p>
                  <p><strong>Draws:</strong> <span id="draws">{{ training_stats["draw"] }}</span> (<span id="pctDraw">{{ pct_draw }}</span>%)</p>
                </div>
              </div>
              <!-- Cumulative Evaluation Stats -->
              <div class="card">
                <div class="card-header">
                  <h5>Cumulative Evaluation Stats</h5>
                </div>
                <div class="card-body">
                  <p><strong>Total Games:</strong> <span id="evalTotal">{{ evaluation_stats.games }}</span></p>
                  <p><strong>Wins:</strong> <span id="evalWins">{{ evaluation_stats.win }}</span> (<span id="evalPctWin">{{ eval_win_pct }}</span>%)</p>
                  <p><strong>Losses:</strong> <span id="evalLosses">{{ evaluation_stats.loss }}</span> (<span id="evalPctLoss">{{ eval_loss_pct }}</span>%)</p>
                  <p><strong>Draws:</strong> <span id="evalDraws">{{ evaluation_stats.draw }}</span> (<span id="evalPctDraw">{{ eval_draw_pct }}</span>%)</p>
                </div>
              </div>
              <!-- Latest 100 Training Stats -->
              <div class="card">
                <div class="card-header">
                  <h5>Latest 100 Training Stats</h5>
                </div>
                <div class="card-body">
                  <p><strong>Total Episodes:</strong> <span id="latestTotal">{{ latest_total }}</span></p>
                  <p><strong>X Wins:</strong> <span id="latestXWins">{{ latest_counts["X"] }}</span> (<span id="latestPctX">{{ latest_pct_X }}</span>%)</p>
                  <p><strong>O Wins:</strong> <span id="latestOWins">{{ latest_counts["O"] }}</span> (<span id="latestPctO">{{ latest_pct_O }}</span>%)</p>
                  <p><strong>Draws:</strong> <span id="latestDraws">{{ latest_counts["draw"] }}</span> (<span id="latestPctDraw">{{ latest_pct_draw }}</span>%)</p>
                </div>
              </div>
              <!-- Latest 100 Evaluation Stats -->
              <div class="card">
                <div class="card-header">
                  <h5>Latest 100 Evaluation Stats</h5>
                </div>
                <div class="card-body">
                  <p><strong>Total Games:</strong> <span id="latestEvalTotal">{{ latest_eval_total }}</span></p>
                  <p><strong>Wins:</strong> <span id="latestEvalWins">{{ latest_eval_counts.win }}</span> (<span id="latestEvalPctWin">{{ latest_eval_win_pct }}</span>%)</p>
                  <p><strong>Losses:</strong> <span id="latestEvalLosses">{{ latest_eval_counts.loss }}</span> (<span id="latestEvalPctLoss">{{ latest_eval_loss_pct }}</span>%)</p>
                  <p><strong>Draws:</strong> <span id="latestEvalDraws">{{ latest_eval_counts.draw }}</span> (<span id="latestEvalPctDraw">{{ latest_eval_draw_pct }}</span>%)</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <script>
          function formatTime(seconds) {
              seconds = Math.round(seconds);
              if (seconds < 60) {
                  return seconds + " sec";
              } else if (seconds < 3600) {
                  var minutes = Math.floor(seconds / 60);
                  var remSec = seconds % 60;
                  return minutes + " min" + (remSec ? " " + remSec + " sec" : "");
              } else {
                  var hours = Math.floor(seconds / 3600);
                  var minutes = Math.floor((seconds % 3600) / 60);
                  var remSec = seconds % 60;
                  var result = hours + " hr";
                  if (minutes) {
                      result += " " + minutes + " min";
                  }
                  if (remSec) {
                      result += " " + remSec + " sec";
                  }
                  return result;
              }
          }

          function calculateEstimatedTimeStr(startTime, completed, total) {
              console.log("JS - startTime (received):", startTime);
              console.log("JS - completed:", completed);
              console.log("JS - total:", total);

              if (!startTime || completed === 0 || total === 0 || completed >= total) {
                  return "0 sec";
              }

              var startTimeMs = Number(startTime); // Explicitly convert to Number
              console.log("JS - startTimeMs (Number):", startTimeMs);

              if (isNaN(startTimeMs)) { // Check if conversion resulted in NaN
                  console.error("JS - startTimeMs is NaN! Invalid timestamp received.");
                  return "Invalid start time"; // Handle invalid timestamp
              }


              var elapsedTimeMs = Date.now() - startTimeMs; // Elapsed time in milliseconds
              console.log("JS - elapsedTimeMs:", elapsedTimeMs);

              if (elapsedTimeMs < 0) { // Check for negative elapsed time (clock issues)
                  console.error("JS - elapsedTimeMs is negative! Clock issue?");
                  return "Clock issue"; // Handle clock issue
              }


              var elapsedTimeSec = elapsedTimeMs / 1000; // Convert to seconds
              console.log("JS - elapsedTimeSec:", elapsedTimeSec);

              var avgTimePerUnit = elapsedTimeSec / completed;
              console.log("JS - avgTimePerUnit:", avgTimePerUnit);
              var remainingUnits = total - completed;
              var estimatedTimeSec = remainingUnits * avgTimePerUnit;
              console.log("JS - estimatedTimeSec (seconds):", estimatedTimeSec);


              return formatTime(estimatedTimeSec);
          }

          function updateStats() {
            $.getJSON("/stats", function(data) {
              $("#totalEpisodes").text(data.training_stats.episodes);
              $("#xWins").text(data.training_stats.X);
              $("#oWins").text(data.training_stats.O);
              $("#draws").text(data.training_stats.draw);
              $("#pctX").text(data.pct_X);
              $("#pctO").text(data.pct_O);
              $("#pctDraw").text(data.pct_draw);

              $("#latestTotal").text(data.latest_stats.episodes);
              $("#latestXWins").text(data.latest_stats.X);
              $("#latestOWins").text(data.latest_stats.O);
              $("#latestDraws").text(data.latest_stats.draw);
              $("#latestPctX").text(data.latest_pct_X);
              $("#latestPctO").text(data.latest_pct_O);
              $("#latestPctDraw").text(data.latest_pct_draw);

              var progress = 0;
              if(data.training_progress.session_total > 0){
                progress = (data.training_progress.session_completed / data.training_progress.session_total * 100).toFixed(2);
              }
              $("#progressBar").css("width", progress + "%").attr("aria-valuenow", progress).text(progress + "%");

              var trainingStartTime = data.training_progress.start_time;
              var trainingCompleted = data.training_progress.session_completed;
              var trainingTotal = data.training_progress.session_total;
              var trainingEstimatedTimeStr = calculateEstimatedTimeStr(trainingStartTime, trainingCompleted, trainingTotal);
              $("#trainingEstimatedTimeStr").text(trainingEstimatedTimeStr);


              $("#evalTotal").text(data.evaluation_stats.games);
              $("#evalWins").text(data.evaluation_stats.win);
              $("#evalLosses").text(data.evaluation_stats.loss);
              $("#evalDraws").text(data.evaluation_stats.draw);
              $("#evalPctWin").text(data.evaluation_stats.win_pct);
              $("#evalPctLoss").text(data.evaluation_stats.loss_pct);
              $("#evalPctDraw").text(data.evaluation_stats.draw_pct);

              $("#latestEvalTotal").text(data.latest_evaluation.episodes);
              $("#latestEvalWins").text(data.latest_evaluation.win);
              $("#latestEvalLosses").text(data.latest_evaluation.loss);
              $("#latestEvalDraws").text(data.latest_evaluation.draw);
              $("#latestEvalPctWin").text(data.latest_evaluation.win_pct);
              $("#latestEvalPctLoss").text(data.latest_evaluation.loss_pct);
              $("#latestEvalPctDraw").text(data.latest_evaluation.draw_pct);

              var evalProgress = 0;
              if(data.evaluation_progress.session_total > 0){
                evalProgress = (data.evaluation_progress.session_completed / data.evaluation_progress.session_total * 100).toFixed(2);
              }
              $("#evalProgressBar").css("width", evalProgress + "%").attr("aria-valuenow", evalProgress).text(evalProgress + "%");

              var evaluationStartTime = data.evaluation_progress.start_time;
              var evaluationCompleted = data.evaluation_progress.session_completed;
              var evaluationTotal = data.evaluation_progress.session_total;
              var evaluationEstimatedTimeStr = calculateEstimatedTimeStr(evaluationStartTime, evaluationCompleted, evaluationTotal);
              $("#evaluationEstimatedTimeStr").text(evaluationEstimatedTimeStr);


            });
          }
          setInterval(updateStats, 1000);

          $("#trainBtn").click(function(){
            var episodes = parseInt($("#episodes").val());
            if(episodes > 0){
              $.ajax({
                url: "/train",
                type: "POST",
                data: JSON.stringify({episodes: episodes}),
                contentType: "application/json",
                success: function(response){
                  $("#trainAlert").html('<div class="alert alert-success">' + response.message + '</div>');
                }
              });
            }
          });

          $("#cancelTrainBtn").click(function(){
            $.ajax({
              url: "/cancel_train",
              type: "POST",
              success: function(response){
                $("#trainAlert").html('<div class="alert alert-danger">' + response.message + '</div>');
              }
            });
          });

          $("#resetStatsBtn").click(function(){
            $.post("/", {action: "reset_stats"}, function(){
              updateStats();
            });
          });

          $("#evalBtn").click(function(){
            var games = parseInt($("#evalGames").val());
            if(games > 0){
              $.ajax({
                url: "/evaluate",
                type: "POST",
                data: JSON.stringify({games: games}),
                contentType: "application/json",
                success: function(response){
                  $("#evalAlert").html('<div class="alert alert-success">' + response.message + '</div>');
                }
              });
            }
          });

          $("#cancelEvalBtn").click(function(){
            $.ajax({
              url: "/cancel_eval",
              type: "POST",
              success: function(response){
                $("#evalAlert").html('<div class="alert alert-danger">' + response.message + '</div>');
              }
            });
          });

          $("#rowsSlider, #colsSlider").on("input", function(){
            var rows = parseInt($("#rowsSlider").val());
            var cols = parseInt($("#colsSlider").val());
            $("#rowsVal").text(rows);
            $("#colsVal").text(cols);
            var newMax = Math.min(rows, cols);
            $("#kSlider").attr("max", newMax);
            var kVal = parseInt($("#kSlider").val());
            if(kVal > newMax) {
              $("#kSlider").val(newMax);
              $("#kVal").text(newMax);
            }
          });
          $("#kSlider").on("input", function(){
            $("#kVal").text($(this).val());
          });
        </script>
      </body>
    </html>
    '''
    return render_template_string(html,
                                  game=game,
                                  message=message,
                                  train_message=train_message,
                                  eval_message=eval_message,
                                  training_stats=model["training_stats"],
                                  evaluation_stats=model["evaluation_stats"],
                                  total_episodes=total_episodes,
                                  pct_X=pct_X, pct_O=pct_O, pct_draw=pct_draw,
                                  latest_total=latest_total, latest_counts=latest_counts,
                                  latest_pct_X=latest_pct_X, latest_pct_O=latest_pct_O, latest_pct_draw=latest_pct_draw,
                                  training_progress=model["training_progress"],
                                  progress=progress,
                                  eval_win_pct=eval_win_pct, eval_loss_pct=eval_loss_pct, eval_draw_pct=eval_draw_pct,
                                  latest_eval_total=latest_eval_total, latest_eval_counts=latest_eval_counts,
                                  latest_eval_win_pct=latest_eval_win_pct, latest_eval_loss_pct=latest_eval_loss_pct, latest_eval_draw_pct=latest_eval_draw_pct,
                                  evaluation_progress=model["evaluation_progress"],
                                  eval_progress=eval_progress)

@app.route("/train", methods=["POST"])
def train():
    global training_thread
    model = models[current_model_key]
    data = request.get_json()
    try:
        episodes = int(data.get("episodes", 1000))
    except (ValueError, TypeError):
        episodes = 1000

    if training_thread is not None and training_thread.is_alive():
        return jsonify({"message": "Training is already in progress."})
    if evaluation_thread is not None and evaluation_thread.is_alive():
        return jsonify({"message": "Evaluation is already in progress, please cancel evaluation before training."})

    with global_lock:
        model["training_progress"]["session_total"] = episodes
        model["training_progress"]["session_completed"] = 0
        model["training_progress"]["start_time"] = time.time() * 1000  # Milliseconds timestamp in Python!
        logger.info(f"Python - Training start_time set (ms): {model['training_progress']['start_time']}") # Log in milliseconds

    training_stop_event.clear()
    training_thread = threading.Thread(target=train_agent, args=(episodes,))
    training_thread.start()
    logger.info("Training started for %d episodes.", episodes)
    return jsonify({"message": f"Training started for {episodes} episodes."})

@app.route("/cancel_train", methods=["POST"])
def cancel_train():
    global training_thread
    if training_thread is None or not training_thread.is_alive():
        return jsonify({"message": "No training is in progress."})
    training_stop_event.set()
    training_thread.join()
    training_thread = None
    logger.info("Training cancelled by user.")
    return jsonify({"message": "Training has been cancelled."})

@app.route("/evaluate", methods=["POST"])
def evaluate():
    global evaluation_thread
    model = models[current_model_key]
    data = request.get_json()
    try:
        games = int(data.get("games", 100))
    except (ValueError, TypeError):
        games = 100

    if evaluation_thread is not None and evaluation_thread.is_alive():
        return jsonify({"message": "Evaluation is already in progress."})
    if training_thread is not None and training_thread.is_alive():
        return jsonify({"message": "Training is already in progress, please cancel training before evaluation."})

    with global_lock:
        model["evaluation_progress"]["session_total"] = games
        model["evaluation_progress"]["session_completed"] = 0
        model["evaluation_progress"]["start_time"] = time.time() * 1000 # Milliseconds timestamp in Python!
        logger.info(f"Python - Evaluation start_time set (ms): {model['evaluation_progress']['start_time']}") # Log in milliseconds

    evaluation_stop_event.clear()
    evaluation_thread = threading.Thread(target=evaluate_agent, args=(games,))
    evaluation_thread.start()
    logger.info("Evaluation started for %d games.", games)
    return jsonify({"message": f"Evaluation started for {games} games."})

@app.route("/cancel_eval", methods=["POST"])
def cancel_eval():
    global evaluation_thread
    if evaluation_thread is None or not evaluation_thread.is_alive():
        return jsonify({"message": "No evaluation is in progress."})
    evaluation_stop_event.set()
    evaluation_thread.join()
    evaluation_thread = None
    logger.info("Evaluation cancelled by user.")
    return jsonify({"message": "Evaluation has been cancelled."})

@app.route("/stats")
def stats():
    model = models[current_model_key]
    with global_lock:
        total = model["training_stats"]["episodes"] if model["training_stats"]["episodes"] > 0 else 1
        pct_X = round((model["training_stats"]["X"] / total * 100), 2)
        pct_O = round((model["training_stats"]["O"] / total * 100), 2)
        pct_draw = round((model["training_stats"]["draw"] / total * 100), 2)

        latest_total = len(model["latest_results"])
        latest_counts = {"X": 0, "O": 0, "draw": 0}
        for res in model["latest_results"]:
            latest_counts[res] += 1
        latest_pct_X = round(latest_counts["X"] / latest_total * 100, 2) if latest_total > 0 else 0
        latest_pct_O = round(latest_counts["O"] / latest_total * 100, 2) if latest_total > 0 else 0
        latest_pct_draw = round(latest_counts["draw"] / latest_total * 100, 2) if latest_total > 0 else 0

        eval_total = model["evaluation_stats"]["games"] if model["evaluation_stats"]["games"] > 0 else 1
        eval_win_pct = round(model["evaluation_stats"]["win"] / eval_total * 100, 2)
        eval_loss_pct = round(model["evaluation_stats"]["loss"] / eval_total * 100, 2)
        eval_draw_pct = round(model["evaluation_stats"]["draw"] / eval_total * 100, 2)

        latest_eval_total = len(model["evaluation_latest"])
        latest_eval_counts = {"win": 0, "loss": 0, "draw": 0}
        for res in model["evaluation_latest"]:
            if res == "X":
                latest_eval_counts["win"] += 1
            elif res == "O":
                latest_eval_counts["loss"] += 1
            else:
                latest_eval_counts["draw"] += 1
        latest_eval_win_pct = round(latest_eval_counts["win"] / latest_eval_total * 100, 2) if latest_eval_total > 0 else 0
        latest_eval_loss_pct = round(latest_eval_counts["loss"] / latest_eval_total * 100, 2) if latest_eval_total > 0 else 0
        latest_eval_draw_pct = round(latest_eval_counts["draw"] / latest_eval_total * 100, 2) if latest_eval_total > 0 else 0

        stats_data = {
            "training_stats": model["training_stats"],
            "latest_stats": {
                "X": latest_counts["X"],
                "O": latest_counts["O"],
                "draw": latest_counts["draw"],
                "episodes": latest_total
            },
            "evaluation_stats": {
                "win": model["evaluation_stats"]["win"],
                "loss": model["evaluation_stats"]["loss"],
                "draw": model["evaluation_stats"]["draw"],
                "games": model["evaluation_stats"]["games"],
                "win_pct": eval_win_pct,
                "loss_pct": eval_loss_pct,
                "draw_pct": eval_draw_pct
            },
            "latest_evaluation": {
                "win": latest_eval_counts["win"],
                "loss": latest_eval_counts["loss"],
                "draw": latest_eval_counts["draw"],
                "episodes": latest_eval_total,
                "win_pct": latest_eval_win_pct,
                "loss_pct": latest_eval_loss_pct,
                "draw_pct": latest_eval_draw_pct
            },
            "training_progress": model["training_progress"],
            "evaluation_progress": model["evaluation_progress"],
            "pct_X": pct_X,
            "pct_O": pct_O,
            "pct_draw": pct_draw,
            "latest_pct_X": latest_pct_X,
            "latest_pct_O": latest_pct_O,
            "latest_pct_draw": latest_pct_draw
        }
        logger.info(f"Python - Stats endpoint - training_progress start_time (ms): {stats_data['training_progress']['start_time']}") # Log in milliseconds
        logger.info(f"Python - Stats endpoint - evaluation_progress start_time (ms): {stats_data['evaluation_progress']['start_time']}") # Log in milliseconds
    return jsonify(stats_data)

if __name__ == '__main__':
    app.run(debug=True)