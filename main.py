from flask import Flask, render_template_string, request, redirect, session, url_for, Response
import random
import matplotlib.pyplot as plt
import io
import os
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key

###############################
# Persistence Setup           #
###############################

TRAINING_DATA_FILE = 'training_data.pkl'

def save_training_data():
    """Persist the agent’s Q-table and training stats to a file."""
    with open(TRAINING_DATA_FILE, 'wb') as f:
        pickle.dump((agent.Q, training_stats), f)

def load_training_data():
    """Load the agent’s Q-table and training stats from a file, if it exists."""
    global agent, training_stats
    if os.path.exists(TRAINING_DATA_FILE):
        with open(TRAINING_DATA_FILE, 'rb') as f:
            agent.Q, training_stats = pickle.load(f)

###############################
# Global Training Statistics  #
###############################

training_stats = {
    "episodes": 0,
    "wins": 0,
    "losses": 0,
    "draws": 0,
    "history": []  # Each entry is a tuple: (total_episodes, wins, losses, draws)
}

###############################
# Game Logic Helper Functions #
###############################

def check_win(board, player):
    wins = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
        (0, 4, 8), (2, 4, 6)              # diagonals
    ]
    for a, b, c in wins:
        if board[a] == board[b] == board[c] == player:
            return True
    return False

def check_game_result(board):
    """
    Returns a tuple (reward, game_over) from the agent's perspective.
    Agent ('O') win: reward = 1
    Opponent ('X') win: reward = -1
    Draw: reward = 0
    """
    if check_win(board, 'X'):
        return (-1, True)
    elif check_win(board, 'O'):
        return (1, True)
    elif ' ' not in board:
        return (0, True)
    else:
        return (0, False)

def board_to_html(board):
    """Render board as an HTML table with clickable empty cells using responsive design."""
    html = "<table class='board-table' style='width:300px; margin: auto;'>"
    for i in range(3):
        html += "<tr>"
        for j in range(3):
            index = i * 3 + j
            cell = board[index]
            if cell == ' ':
                cell_html = (f"<a href='/move?pos={index}' class='cell-link'>"
                             f"<div class='cell'><div class='cell-content'>{cell}</div></div>"
                             "</a>")
            else:
                cell_html = (f"<div class='cell'><div class='cell-content'>{cell}</div></div>")
            html += f"<td>{cell_html}</td>"
        html += "</tr>"
    html += "</table>"
    return html

###############################
# Q-Learning Agent Definition #
###############################

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.Q = {}  # Q-table: key = (state, action), value = Q-value
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def canonical_state(self, board):
        return tuple(board)

    def get_valid_actions(self, board):
        return [i for i, cell in enumerate(board) if cell == ' ']

    def choose_action(self, board):
        state = self.canonical_state(board)
        valid_actions = self.get_valid_actions(board)
        if not valid_actions:
            return None
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            q_values = [self.Q.get((state, a), 0) for a in valid_actions]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
            return random.choice(best_actions)

    def update_q(self, board, action, reward, next_board):
        state = self.canonical_state(board)
        next_state = self.canonical_state(next_board)
        current_q = self.Q.get((state, action), 0)
        valid_actions = self.get_valid_actions(next_board)
        max_next_q = max([self.Q.get((next_state, a), 0) for a in valid_actions]) if valid_actions else 0
        self.Q[(state, action)] = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)

# Create our global agent.
agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.2)

# Attempt to load previous training data from file.
load_training_data()

#########################################
# Training Function (on-demand batch)   #
#########################################

def train_agent_batch(agent, episodes):
    """
    Train the agent for a given number of episodes and return win/loss/draw counts.
    """
    batch_wins = 0
    batch_losses = 0
    batch_draws = 0
    for _ in range(episodes):
        board = [' '] * 9
        game_over = False
        # Randomly decide if the opponent (playing as 'X') goes first.
        if random.random() < 0.5:
            valid = agent.get_valid_actions(board)
            move = random.choice(valid)
            board[move] = 'X'
        while not game_over:
            # Agent's turn as 'O'
            action = agent.choose_action(board)
            if action is None:
                break
            prev_board = board.copy()
            board[action] = 'O'
            reward, game_over = check_game_result(board)
            agent.update_q(prev_board, action, reward, board)
            if game_over:
                break
            # Opponent's turn as 'X'
            valid = agent.get_valid_actions(board)
            if not valid:
                break
            opponent_action = random.choice(valid)
            board[opponent_action] = 'X'
            reward, game_over = check_game_result(board)
        reward_final, _ = check_game_result(board)
        if reward_final == 1:
            batch_wins += 1
        elif reward_final == -1:
            batch_losses += 1
        else:
            batch_draws += 1
    return batch_wins, batch_losses, batch_draws

###############################
# Flask Routes (Web Interface)#
###############################

@app.route('/')
def index():
    if 'board' not in session:
        session['board'] = [' '] * 9
    board = session['board']
    message = session.pop('message', '')
    board_html = board_to_html(board)
    
    episodes = training_stats['episodes']
    wins = training_stats['wins']
    losses = training_stats['losses']
    draws = training_stats['draws']
    if episodes > 0:
        win_pct = wins / episodes * 100
        loss_pct = losses / episodes * 100
        draw_pct = draws / episodes * 100
    else:
        win_pct = loss_pct = draw_pct = 0

    stats = (f"<p>Total Training Episodes: {episodes}</p>"
             f"<p>Wins: {wins} ({win_pct:.2f}%) | Losses: {losses} ({loss_pct:.2f}%) | Draws: {draws} ({draw_pct:.2f}%)</p>")

    template = """
    <html>
    <head>
      <title>Tic Tac Toe - Q-Learning Agent</title>
      <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
      <style>
        body {
          background-color: #f8f9fa;
        }
        .container {
          max-width: 800px;
          margin-top: 20px;
        }
        .board-table {
          margin: auto;
        }
        .board-table td {
          padding: 0;
        }
        .cell {
          width: 100%;
          padding-top: 100%;
          position: relative;
          border: 1px solid #000;
          font-size: 2em;
          text-align: center;
        }
        .cell-content {
          position: absolute;
          top: 0;
          bottom: 0;
          left: 0;
          right: 0;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        a.cell-link {
          display: block;
          width: 100%;
          height: 100%;
          text-decoration: none;
          color: black;
        }
      </style>
      <script>
        function updateSlider(val) {
            document.getElementById('episode_value').innerText = val;
        }
      </script>
    </head>
    <body>
      <div class="container">
        <h1 class="text-center">Tic Tac Toe</h1>
        <div class="text-center">
          {{ board_html|safe }}
          <p class="mt-3">{{ message }}</p>
          <a href="{{ url_for('reset') }}" class="btn btn-warning">Reset Game</a>
        </div>
        <hr>
        <h2>Train the Agent</h2>
        <form action="/train_more" method="post" class="mb-3">
          <div class="form-group">
              <label for="episodes">Train Episodes:</label>
              <input type="range" id="episodes" name="episodes" min="100" max="50000" value="1000" step="100" oninput="updateSlider(this.value)" class="form-control-range">
              <span id="episode_value">1000</span>
          </div>
          <button type="submit" class="btn btn-primary">Train More</button>
        </form>
        <h3>Training Stats</h3>
        <div>
          {{ stats|safe }}
        </div>
        <h3>Training Graph</h3>
        <div>
          <img src="{{ url_for('plot_png') }}" alt="Training Graph" class="img-fluid">
        </div>
      </div>
    </body>
    </html>
    """
    return render_template_string(template, board_html=board_html, message=message, stats=stats)

@app.route('/move')
def move():
    pos = int(request.args.get('pos'))
    board = session.get('board', [' '] * 9)
    
    if board[pos] != ' ':
        session['message'] = "Invalid move! Please choose an empty cell."
        return redirect(url_for('index'))
    
    # Human move as 'X'
    board[pos] = 'X'
    reward, game_over = check_game_result(board)
    if game_over:
        if reward == -1:
            session['message'] = "Congratulations, you win!"
        elif reward == 0:
            session['message'] = "It's a draw!"
        session['board'] = board
        return redirect(url_for('index'))
    
    # Agent's move as 'O'
    valid = [i for i, cell in enumerate(board) if cell == ' ']
    if valid:
        action = agent.choose_action(board)
        board[action] = 'O'
        reward, game_over = check_game_result(board)
        if game_over:
            if reward == 1:
                session['message'] = "Agent wins! Better luck next time."
            elif reward == 0:
                session['message'] = "It's a draw!"
    session['board'] = board
    return redirect(url_for('index'))

@app.route('/reset')
def reset():
    session['board'] = [' '] * 9
    session['message'] = "Game reset! Your move."
    return redirect(url_for('index'))

@app.route('/train_more', methods=['POST'])
def train_more():
    try:
        episodes = int(request.form.get('episodes', 1000))
    except ValueError:
        episodes = 1000

    wins, losses, draws = train_agent_batch(agent, episodes)

    training_stats['episodes'] += episodes
    training_stats['wins'] += wins
    training_stats['losses'] += losses
    training_stats['draws'] += draws
    training_stats['history'].append(
        (training_stats['episodes'], training_stats['wins'], training_stats['losses'], training_stats['draws'])
    )
    
    save_training_data()
    
    session['message'] = (f"Trained for {episodes} episodes: Wins {wins}, Losses {losses}, Draws {draws}.")
    return redirect(url_for('index'))

@app.route('/plot.png')
def plot_png():
    history = training_stats['history']
    if not history:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No training data", horizontalalignment='center',
                verticalalignment='center', fontsize=16)
    else:
        episodes, wins, losses, draws = zip(*history)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(episodes, wins, label="Wins", color="green")
        ax.plot(episodes, losses, label="Losses", color="red")
        ax.plot(episodes, draws, label="Draws", color="blue")
        ax.set_xlabel("Total Episodes")
        ax.set_ylabel("Cumulative Count")
        ax.set_title("Training Progress")
        ax.legend()
        ax.grid(True)

    png_image = io.BytesIO()
    plt.tight_layout()
    plt.savefig(png_image, format='png')
    plt.close(fig)
    png_image.seek(0)
    return Response(png_image.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
