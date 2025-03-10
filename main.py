import os
import random
import math
import json
from flask import Flask, render_template_string, redirect, url_for, session

# --------------------
# Game Logic Functions
# --------------------
def check_win(board, player):
    """Return True if the given player has a winning combination."""
    win_combinations = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]              # Diagonals
    ]
    for combo in win_combinations:
        if all(board[i] == player for i in combo):
            return True
    return False

def get_valid_moves(board):
    """Return a list of indices for empty cells."""
    return [i for i, cell in enumerate(board) if cell == ' ']

def is_board_full(board):
    """Return True if there are no empty cells."""
    return ' ' not in board

# ------------------------
# Q-Learning Agent Class
# ------------------------
class QLearningAgent:
    def __init__(self, player_mark, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.3, filename=None):
        """
        Initialize the Q-Learning agent.
        Args:
            player_mark: 'X' or 'O'
            learning_rate: Step size for updating Q-values.
            discount_factor: Future reward discount factor.
            exploration_rate: Chance to take a random move.
            filename: File to load/save Q-values.
        """
        self.player_mark = player_mark
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        # Set filename for persistence
        self.filename = filename if filename is not None else f"q_table_{player_mark}.json"
        self.q_table = {}
        self.load_q_table()

    def get_state_key(self, board):
        """Convert board list to a tuple to use as a key."""
        return tuple(board)

    def get_q_value(self, state_key, action):
        """Return the Q-value for a given state and action (default is 0.0)."""
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        return self.q_table[state_key].get(action, 0.0)

    def update_q_value(self, state_key, action, reward, next_state_key):
        """Update the Q-value using the Q-learning rule."""
        current_q = self.get_q_value(state_key, action)
        max_next_q = 0
        if next_state_key is not None:
            next_valid_moves = get_valid_moves(list(next_state_key))
            if next_valid_moves:
                max_next_q = max(self.get_q_value(next_state_key, a) for a in next_valid_moves)
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state_key][action] = new_q

    def choose_action(self, board):
        """
        Choose an action using an ε-greedy policy.
        Returns:
            A valid move index.
        """
        state_key = self.get_state_key(board)
        valid_moves = get_valid_moves(board)
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(valid_moves)
        else:
            best_action = None
            max_q = -math.inf
            for action in valid_moves:
                q = self.get_q_value(state_key, action)
                if q > max_q:
                    max_q = q
                    best_action = action
            return best_action if best_action is not None else random.choice(valid_moves)

    def play_move(self, board):
        """
        For game play (vs. human) use exploitation only.
        Returns:
            Best move (index) from the Q-table.
        """
        state_key = self.get_state_key(board)
        valid_moves = get_valid_moves(board)
        best_action = None
        max_q = -math.inf
        for action in valid_moves:
            q = self.get_q_value(state_key, action)
            if q > max_q:
                max_q = q
                best_action = action
        return best_action if best_action is not None else random.choice(valid_moves)

    def train_episode_self_play(self, opponent_agent):
        """
        Train a single episode using self-play between this agent and the opponent.
        The agent playing as 'X' always starts.
        """
        board = [' '] * 9
        current_player_mark = 'X'
        agents = {'X': self, 'O': opponent_agent}

        while True:
            current_agent = agents[current_player_mark]
            state_key = current_agent.get_state_key(board)
            action = current_agent.choose_action(board)
            board[action] = current_player_mark

            if check_win(board, current_player_mark):
                # Winner gets positive reward, loser gets negative reward.
                current_agent.update_q_value(state_key, action, 1, None)
                # Update opponent (if applicable) with a loss.
                opponent_state = opponent_agent.get_state_key(board)
                opponent_agent.update_q_value(opponent_state, action, -1, None)
                break
            elif is_board_full(board):
                # Draw: no reward.
                current_agent.update_q_value(state_key, action, 0, None)
                break
            else:
                next_state_key = current_agent.get_state_key(board)
                current_agent.update_q_value(state_key, action, 0, next_state_key)
            current_player_mark = 'O' if current_player_mark == 'X' else 'X'

    def train_self_play(self, opponent_agent, num_episodes):
        """Train using self-play for a number of episodes."""
        for episode in range(num_episodes):
            self.train_episode_self_play(opponent_agent)
            if (episode + 1) % 1000 == 0:
                print(f"Trained {episode + 1} episodes.")

    def load_q_table(self):
        """Load Q-table from file if it exists."""
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                data = json.load(f)
            self.q_table = {}
            # Convert state strings back to tuple keys, and inner keys to int.
            for state_str, action_dict in data.items():
                state_tuple = tuple(state_str)
                self.q_table[state_tuple] = {int(action): q for action, q in action_dict.items()}
            print(f"Loaded Q-table from {self.filename}")
        else:
            self.q_table = {}
            print(f"No existing Q-table found for {self.player_mark}; starting fresh.")

    def save_q_table(self):
        """Save the Q-table to a file."""
        data = {}
        for state_tuple, action_dict in self.q_table.items():
            state_str = "".join(state_tuple)
            data[state_str] = {str(action): q for action, q in action_dict.items()}
        with open(self.filename, 'w') as f:
            json.dump(data, f)
        print(f"Saved Q-table to {self.filename}")

# ------------------------------
# Train or Load Q-Learning Agents
# ------------------------------
# Define file names for persistence.
agent_x_file = "q_table_X.json"
agent_o_file = "q_table_O.json"

# Initialize agents with persistence filenames.
agent_x = QLearningAgent(player_mark='X', exploration_rate=0.3, filename=agent_x_file)
agent_o = QLearningAgent(player_mark='O', exploration_rate=0.3, filename=agent_o_file)

# If no saved Q-tables exist, train the agents and then save their learnings.
if not (os.path.exists(agent_x_file) and os.path.exists(agent_o_file)):
    print("Training Q-Learning agents through self-play...")
    agent_x.train_self_play(agent_o, num_episodes=10000)
    print("Training complete.")
    agent_x.save_q_table()
    agent_o.save_q_table()
else:
    print("Using previously saved Q-learning agents.")

# For human vs AI, set exploration to 0 and use agent_o.
agent_o.exploration_rate = 0.0
trained_agent = agent_o  # The AI opponent

# ------------------------------
# Flask Web Application Setup
# ------------------------------
app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Replace with a secure key in production

def new_game():
    """Initialize a new game and store it in the session."""
    session['board'] = [' '] * 9
    session['message'] = ""
    session['game_over'] = False

# Bootstrap-enhanced HTML template with a modern UI.
TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8">
    <title>Tic Tac Toe - Q-Learning Agent</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
      body { background: #f8f9fa; }
      .board {
        display: grid;
        grid-template-columns: repeat(3, 100px);
        grid-gap: 5px;
        justify-content: center;
        margin-top: 30px;
      }
      .cell {
        width: 100px;
        height: 100px;
        background: #fff;
        border: 2px solid #343a40;
        font-size: 2em;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: background-color 0.3s ease;
      }
      .cell:hover { background-color: #e9ecef; }
      .cell.disabled {
        cursor: default;
        background-color: #dee2e6;
      }
      .message { text-align: center; margin-top: 20px; font-size: 1.5em; }
      .new-game { margin-top: 20px; text-align: center; }
    </style>
  </head>
  <body>
    <div class="container">
      <h1 class="text-center mt-4">Tic Tac Toe</h1>
      <h2 class="text-center text-info">{{ message }}</h2>
      <div class="board">
        {% for i in range(9) %}
          {% if board[i] == ' ' and not game_over %}
            <a href="{{ url_for('human_move', move=i) }}" class="cell text-dark text-decoration-none"></a>
          {% else %}
            <div class="cell disabled">{{ board[i] }}</div>
          {% endif %}
        {% endfor %}
      </div>
      <div class="new-game">
        <a href="{{ url_for('reset') }}" class="btn btn-primary">New Game</a>
      </div>
    </div>
  </body>
</html>
'''

@app.route('/')
def index():
    if 'board' not in session:
        new_game()
    board = session.get('board')
    message = session.get('message')
    game_over = session.get('game_over')
    return render_template_string(TEMPLATE, board=board, message=message, game_over=game_over)

@app.route('/move/<int:move>')
def human_move(move):
    if 'board' not in session:
        new_game()
    board = session['board']
    game_over = session.get('game_over', False)
    message = ""

    # Process move only if game is ongoing.
    if not game_over and board[move] == ' ':
        # Human plays as 'X'
        board[move] = 'X'
        if check_win(board, 'X'):
            message = "Congratulations, you win!"
            game_over = True
        elif is_board_full(board):
            message = "It's a draw!"
            game_over = True
        else:
            # AI's turn (plays as 'O')
            ai_move = trained_agent.play_move(board)
            board[ai_move] = 'O'
            if check_win(board, 'O'):
                message = "AI wins!"
                game_over = True
            elif is_board_full(board):
                message = "It's a draw!"
                game_over = True

    session['board'] = board
    session['message'] = message
    session['game_over'] = game_over
    return redirect(url_for('index'))

@app.route('/reset')
def reset():
    new_game()
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Run the Flask app in debug mode.
    app.run(debug=True)
