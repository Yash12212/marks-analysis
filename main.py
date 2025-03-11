from flask import Flask, request, jsonify, render_template_string, Response
import random, json, os, time

app = Flask(__name__)

# Q‑learning parameters and global Q‑table
q_table = {}  # state -> {action: Q-value}
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1  # exploration probability

# --- Tic Tac Toe helper functions ---
def get_empty_board():
    return [' '] * 9

def board_to_str(board):
    return ''.join(board)

def available_actions(board):
    return [i for i, cell in enumerate(board) if cell == ' ']

def check_winner(board):
    wins = [
        (0,1,2), (3,4,5), (6,7,8),   # rows
        (0,3,6), (1,4,7), (2,5,8),   # columns
        (0,4,8), (2,4,6)             # diagonals
    ]
    for a, b, c in wins:
        if board[a] != ' ' and board[a] == board[b] == board[c]:
            return board[a]
    if ' ' not in board:
        return 'Draw'
    return None

# --- Q‑learning functions ---
def choose_action(state, board):
    global q_table, epsilon
    if random.random() < epsilon:
        return random.choice(available_actions(board))
    else:
        if state not in q_table:
            q_table[state] = {}
        for action in available_actions(board):
            if action not in q_table[state]:
                q_table[state][action] = 0.0
        actions = q_table[state]
        max_val = max(actions.values()) if actions else 0
        best_actions = [a for a, v in actions.items() if v == max_val]
        return random.choice(best_actions) if best_actions else random.choice(available_actions(board))

def update_q_value(state, action, reward, next_state, next_board):
    global q_table, learning_rate, discount_factor
    if state not in q_table:
        q_table[state] = {}
    if action not in q_table[state]:
        q_table[state][action] = 0.0
    if next_state not in q_table or not available_actions(next_board):
        max_next = 0
    else:
        for act in available_actions(next_board):
            if act not in q_table[next_state]:
                q_table[next_state][act] = 0.0
        max_next = max(q_table[next_state].values()) if q_table[next_state] else 0
    q_table[state][action] = q_table[state][action] + learning_rate * (reward + discount_factor * max_next - q_table[state][action])

def simulate_game(train=False):
    board = get_empty_board()
    state = board_to_str(board)
    current_player = 'X'
    moves = []  # record moves for Q‑learning updates
    while True:
        if current_player == 'X':
            action = choose_action(state, board)
        else:
            action = random.choice(available_actions(board))
        moves.append((state, action, current_player))
        board[action] = current_player
        winner = check_winner(board)
        next_state = board_to_str(board)
        if winner is not None:
            reward = 1 if winner == 'X' else (-1 if winner != 'Draw' else 0.5)
            if train:
                for s, a, p in moves:
                    if p == 'X':
                        update_q_value(s, a, reward, next_state, board)
            return winner, moves
        state = next_state
        current_player = 'O' if current_player == 'X' else 'X'

# --- Flask Endpoints ---
@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Tic Tac Toe Agent Interface</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
      body { font-family: Arial, sans-serif; }
      .board { 
          display: grid; 
          grid-template-columns: repeat(3, 100px); 
          grid-gap: 5px; 
          justify-content: center; 
          margin-bottom: 20px; 
      }
      .cell { 
          background: #fff; 
          border: 1px solid #333; 
          display: flex; 
          align-items: center; 
          justify-content: center; 
          font-size: 2em; 
          width: 100px; 
          height: 100px; 
          cursor: pointer; 
      }
      .widget { margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="jumbotron text-center">
            <h1>Tic Tac Toe Agent Interface</h1>
        </div>
        <div class="row">
            <!-- Left Sidebar -->
            <div class="col-md-4">
                <div class="card widget" id="train_widget">
                    <div class="card-header">Train Agent</div>
                    <div class="card-body">
                        <input type="number" id="train_episodes" class="form-control mb-2" value="1000" min="1">
                        <button id="train_btn" class="btn btn-primary btn-block">Train</button>
                        <div class="progress mt-2" style="height: 25px;">
                            <div id="train_progress_bar" class="progress-bar" role="progressbar" style="width: 0%;" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">0%</div>
                        </div>
                    </div>
                </div>
                <div class="card widget" id="simulate_widget">
                    <div class="card-header">Simulate Game</div>
                    <div class="card-body">
                        <button id="simulate_btn" class="btn btn-info btn-block">Simulate Single Game</button>
                        <div id="simulate_result" class="mt-2"></div>
                    </div>
                </div>
                <div class="card widget" id="stats_widget">
                    <div class="card-header">Run Simulations for Statistics</div>
                    <div class="card-body">
                        <input type="number" id="num_games" class="form-control mb-2" value="100" min="1">
                        <button id="run_sim_btn" class="btn btn-info btn-block">Run Simulations</button>
                    </div>
                </div>
            </div>
            <!-- Main Content -->
            <div class="col-md-8">
                <div class="card widget" id="play_widget">
                    <div class="card-header">Play Against Agent</div>
                    <div class="card-body text-center">
                        <div class="board" id="game_board">
                            {% for i in range(9) %}
                            <div class="cell" data-index="{{ i }}"></div>
                            {% endfor %}
                        </div>
                        <button id="reset_btn" class="btn btn-warning">Reset Game</button>
                        <div id="game_status" class="mt-2"></div>
                    </div>
                </div>
                <div class="card widget" id="chart_widget">
                    <div class="card-header">Game Statistics (%)</div>
                    <div class="card-body">
                        <canvas id="statsChart" width="400" height="400"></canvas>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- jQuery and Bootstrap JS -->
    <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
    
    <script>
    var board = Array(9).fill(" ");
    function updateBoard() {
        $(".cell").each(function(){
            var idx = $(this).data("index");
            $(this).text(board[idx]);
        });
    }
    function checkWinner() {
        const wins = [
            [0,1,2],[3,4,5],[6,7,8],
            [0,3,6],[1,4,7],[2,5,8],
            [0,4,8],[2,4,6]
        ];
        for(let win of wins) {
            const [a,b,c] = win;
            if(board[a] !== " " && board[a] === board[b] && board[a] === board[c]) {
                return board[a];
            }
        }
        if(board.indexOf(" ") === -1) return "Draw";
        return null;
    }
    
    // Game play: Human move then agent move
    $("#game_board").on("click", ".cell", function(){
        var idx = $(this).data("index");
        if(board[idx] !== " " || checkWinner() !== null) return;
        board[idx] = "X";
        updateBoard();
        var winner = checkWinner();
        if(winner) {
            $("#game_status").text("Winner: " + winner);
            return;
        }
        $.ajax({
            url: "/play",
            method: "POST",
            contentType: "application/json",
            data: JSON.stringify({board: board}),
            success: function(data) {
                board = data.board;
                updateBoard();
                var winner = checkWinner();
                if(winner) {
                    $("#game_status").text("Winner: " + winner);
                }
            }
        });
    });
    $("#reset_btn").click(function(){
        board = Array(9).fill(" ");
        updateBoard();
        $("#game_status").text("");
    });
    
    // Train agent with real-time progress using Server-Sent Events (SSE)
    $("#train_btn").click(function(){
        var episodes = parseInt($("#train_episodes").val());
        var source = new EventSource("/train_stream?episodes=" + episodes);
        source.onmessage = function(event) {
             var data = JSON.parse(event.data);
             var progress = data.progress;
             $("#train_progress_bar").css("width", progress + "%")
                .attr("aria-valuenow", progress)
                .text(Math.floor(progress) + "%");
             // Update the pie chart with current stats
             var wins_x = data.wins_x;
             var wins_o = data.wins_o;
             var draws = data.draws;
             var total = wins_x + wins_o + draws;
             if(total === 0) total = 1;
             statsChart.data.datasets[0].data = [
                        ((wins_x/total)*100).toFixed(1),
                        ((wins_o/total)*100).toFixed(1),
                        ((draws/total)*100).toFixed(1)
                    ];
             statsChart.update();
             if(progress >= 100) {
                  source.close();
                  alert("Training complete!");
             }
        };
    });
    
    // Simulate a single game
    $("#simulate_btn").click(function(){
        $.ajax({
            url: "/simulate",
            method: "GET",
            success: function(data) {
                $("#simulate_result").text("Result: " + data.result);
            }
        });
    });
    
    // Run simulations and update pie chart for statistics
    $("#run_sim_btn").click(function(){
        var numGames = parseInt($("#num_games").val());
        $.ajax({
            url: "/simulate_stats?games=" + numGames,
            method: "GET",
            success: function(data) {
                var total = data.X + data.O + data.Draw;
                if(total === 0) total = 1;
                statsChart.data.datasets[0].data = [
                    ((data.X/total)*100).toFixed(1),
                    ((data.O/total)*100).toFixed(1),
                    ((data.Draw/total)*100).toFixed(1)
                ];
                statsChart.update();
            }
        });
    });
    
    // Initialize Chart.js pie chart
    var ctx = document.getElementById('statsChart').getContext('2d');
    var statsChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['X Wins', 'O Wins', 'Draws'],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: ['#36A2EB', '#FF6384', '#FFCE56']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            title: {
                display: true,
                text: 'Game Statistics (%)'
            }
        }
    });
    </script>
</body>
</html>
    ''')

# Agent plays as 'O' in this endpoint.
@app.route('/play', methods=['POST'])
def play():
    data = request.get_json()
    board = data.get("board")
    state = ''.join(board)
    if ' ' in board and check_winner(board) is None:
        action = choose_action(state, board)
        board[action] = 'O'
    return jsonify({"board": board})

# Simulate a single game (without training updates)
@app.route('/simulate', methods=['GET'])
def simulate():
    result, moves = simulate_game(train=False)
    return jsonify({"result": result, "moves": moves})

# Simulate multiple games for win/loss/draw statistics
@app.route('/simulate_stats', methods=['GET'])
def simulate_stats():
    num_games = int(request.args.get("games", 100))
    stats = {"X": 0, "O": 0, "Draw": 0}
    for _ in range(num_games):
        result, _ = simulate_game(train=False)
        if result in stats:
            stats[result] += 1
    return jsonify(stats)

# New endpoint for training with real-time progress via SSE
@app.route('/train_stream', methods=['GET'])
def train_stream():
    episodes = int(request.args.get("episodes", 1000))
    def generate():
        wins_x = 0
        wins_o = 0
        draws = 0
        for i in range(episodes):
            result, _ = simulate_game(train=True)
            if result == 'X':
                wins_x += 1
            elif result == 'O':
                wins_o += 1
            elif result == 'Draw':
                draws += 1
            progress = (i+1) / episodes * 100
            data = json.dumps({"progress": progress, "wins_x": wins_x, "wins_o": wins_o, "draws": draws})
            yield f"data: {data}\n\n"
            # Optional: add a tiny delay to help the browser update UI
            # time.sleep(0.001)
    return Response(generate(), mimetype='text/event-stream')

# Save layout and Q‑table data (removed from UI per instructions; endpoints remain)
@app.route('/save', methods=['POST'])
def save():
    data = request.get_json()
    layout = data.get("layout", {})
    with open("tictactoe_data.json", "w") as f:
        json.dump({"q_table": q_table, "layout": layout}, f)
    return jsonify({"status": "Data saved"})

# Load saved layout and Q‑table data (removed from UI per instructions; endpoints remain)
@app.route('/load', methods=['GET'])
def load():
    if os.path.exists("tictactoe_data.json"):
        with open("tictactoe_data.json", "r") as f:
            data = json.load(f)
        global q_table
        q_table = data.get("q_table", {})
        layout = data.get("layout", {})
        return jsonify({"status": "Data loaded", "layout": layout})
    else:
        return jsonify({"status": "No saved data found", "layout": {}})

if __name__ == '__main__':
    app.run(debug=True)
