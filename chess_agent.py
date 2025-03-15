import os
import chess
import chess.pgn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.tensorboard import SummaryWriter

# ----------------------------
# Device Setup and Optimizations
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True  # Optimize convolutions for fixed input sizes

# ----------------------------
# 1. Board Encoding (with CUDA)
# ----------------------------
def board_to_tensor(board):
    """
    Converts a python-chess board into a tensor with shape (1, 12, 8, 8) on the target device.
    Channels 0-5: White's pawn, knight, bishop, rook, queen, king.
    Channels 6-11: Black's corresponding pieces.
    """
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    piece_map = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            piece_type = piece.piece_type
            channel = piece_map[piece_type] if piece.color == chess.WHITE else piece_map[piece_type] + 6
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            tensor[channel, row, col] = 1.0
    return torch.tensor(tensor, device=device).unsqueeze(0)

def fen_to_tensor(fen):
    board = chess.Board(fen)
    return board_to_tensor(board)

# ----------------------------
# 2. Neural Network Definition
# ----------------------------
class ChessNet(nn.Module):
    def __init__(self, num_moves=4672):
        """
        A simple CNN that outputs:
         - policy_logits: a vector of length num_moves (for move probabilities).
         - value: a scalar in [-1, 1] evaluating the board state.
        """
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128 * 8 * 8, 1024)
        # Policy head
        self.policy_fc = nn.Linear(1024, num_moves)
        # Value head
        self.value_fc1 = nn.Linear(1024, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc(x))
        policy_logits = self.policy_fc(x)
        value = F.relu(self.value_fc1(x))
        value = torch.tanh(self.value_fc2(value))
        return policy_logits, value

def evaluate_board(board, net):
    """
    Evaluates a board state using the provided network.
    Returns a softmax over policy logits and a scalar value.
    """
    net.eval()
    with torch.no_grad():
        board_tensor = board_to_tensor(board)
        policy_logits, value = net(board_tensor)
        policy_probs = F.softmax(policy_logits, dim=1)
    return policy_probs.squeeze(0), value.item()

# ----------------------------
# 3. MCTS with Adaptive Simulation & Caching
# ----------------------------
class MCTSNode:
    def __init__(self, board, parent=None):
        self.board = board.copy()
        self.parent = parent
        self.children = {}       # move -> child node
        self.N = 0               # visit count
        self.W = 0.0             # total value
        self.Q = 0.0             # mean value (W / N)
        self.P = {}              # prior probabilities for moves (move -> prior)
        self.is_expanded = False

def expand(node, net, eval_cache):
    """
    Expands a node by creating children for all legal moves using network policy as priors.
    Uses a cache to avoid redundant evaluations.
    """
    if node.board.is_game_over():
        node.is_expanded = True
        return

    legal_moves = list(node.board.legal_moves)
    board_fen = node.board.fen()
    if board_fen in eval_cache:
        policy_probs, _ = eval_cache[board_fen]
    else:
        policy_probs, _ = evaluate_board(node.board, net)
        eval_cache[board_fen] = (policy_probs, _)
    
    legal_policy = []
    for move in legal_moves:
        idx = move_to_index(move)
        legal_policy.append(policy_probs[idx].item())
    legal_policy = np.array(legal_policy, dtype=np.float32)
    legal_policy = np.exp(legal_policy - np.max(legal_policy))
    legal_policy /= (np.sum(legal_policy) + 1e-8)
    
    for i, move in enumerate(legal_moves):
        node.board.push(move)
        child_board = node.board.copy()
        node.board.pop()
        child_node = MCTSNode(child_board, parent=node)
        node.children[move] = child_node
        node.P[move] = legal_policy[i]
    node.is_expanded = True

def mcts_search(board, net, base_simulations=50):
    """
    Runs MCTS from the given board state.
    Uses adaptive simulation count based on the branching factor.
    Returns a dictionary mapping moves to their normalized visit counts.
    """
    legal_moves_count = len(list(board.legal_moves))
    num_simulations = base_simulations if legal_moves_count > 10 else max(10, base_simulations // 2)
    
    root = MCTSNode(board)
    eval_cache = {}
    expand(root, net, eval_cache)
    
    for _ in range(num_simulations):
        node = root
        # --- Selection ---
        while node.is_expanded and not node.board.is_game_over():
            best_score = -float('inf')
            best_move = None
            for move, child in node.children.items():
                c = 1.0  # Exploration constant
                score = child.Q + c * node.P[move] * np.sqrt(node.N + 1) / (1 + child.N)
                if score > best_score:
                    best_score = score
                    best_move = move
            node = node.children[best_move]
        # --- Evaluation ---
        if node.board.is_game_over():
            outcome = node.board.outcome()
            if outcome is None or outcome.winner is None:
                value = 0
            else:
                value = 1 if outcome.winner == board.turn else -1
        else:
            _, value = evaluate_board(node.board, net)
        # --- Backpropagation ---
        while node is not None:
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            node = node.parent

    move_probs = {move: child.N for move, child in root.children.items()}
    total = sum(move_probs.values())
    for move in move_probs:
        move_probs[move] /= total
    return move_probs

# ----------------------------
# 4. Mapping Moves to Indices (Placeholder)
# ----------------------------
def move_to_index(move):
    """
    Placeholder: Maps a chess.Move (via its UCI string) to a unique index.
    In a full implementation, use a consistent mapping for all moves.
    """
    num_moves = 4672
    return hash(move.uci()) % num_moves

# ----------------------------
# 5. Self-Play Game Generation with PGN Logging and Epsilon-Greedy
# ----------------------------
def self_play_game(net, num_simulations=50, epsilon=0.0):
    """
    Plays a self-play game using MCTS with ε-greedy exploration.
    Returns (training_examples, pgn_game).
    """
    board = chess.Board()
    game = chess.pgn.Game()
    node = game

    game.headers["Event"] = "Self-Play Training"
    game.headers["Site"] = "Local"
    game.headers["Date"] = "2025.03.15"
    
    game_history = []
    while not board.is_game_over():
        move_probs = mcts_search(board, net, base_simulations=num_simulations)
        moves, probs = zip(*move_probs.items())
        # With probability ε, choose a random move instead of the MCTS move.
        if random.random() < epsilon:
            move = random.choice(list(board.legal_moves))
        else:
            move = np.random.choice(moves, p=np.array(probs))
        
        game_history.append((board.fen(), move_probs))
        node = node.add_variation(move)
        board.push(move)
    
    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        reward = 0
        result = "*"
    else:
        reward = 1 if outcome.winner == chess.WHITE else -1
        result = "1-0" if outcome.winner == chess.WHITE else "0-1"
    game.headers["Result"] = result

    training_examples = []
    for fen, move_probs in game_history:
        training_examples.append((fen, move_probs, reward))
        reward = -reward  # Alternate perspective.
    
    return training_examples, game

# ----------------------------
# 6. Training Loop with CUDA, AMP, and TensorBoard Logging
# ----------------------------
def train_model(model, optimizer, training_examples, writer, global_step, batch_size=32, epochs=1):
    """
    Trains the network on self-play examples using mixed-precision training.
    Logs losses to TensorBoard.
    """
    model.train()
    scaler = torch.amp.GradScaler(device_type=device.type)
    
    for epoch in range(epochs):
        random.shuffle(training_examples)
        for i in range(0, len(training_examples), batch_size):
            batch = training_examples[i:i+batch_size]
            boards = []
            target_policies = []
            target_values = []
            for fen, move_probs, outcome in batch:
                board_tensor = fen_to_tensor(fen)
                boards.append(board_tensor)
                target_policy = np.zeros(4672, dtype=np.float32)
                for move, prob in move_probs.items():
                    idx = move_to_index(move)
                    target_policy[idx] = prob
                target_policies.append(target_policy)
                target_values.append(outcome)
            boards = torch.cat(boards, dim=0)
            target_policies = torch.tensor(np.array(target_policies), device=device)
            target_values = torch.tensor(target_values, device=device, dtype=torch.float32).unsqueeze(1)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type):
                pred_policy_logits, pred_values = model(boards)
                policy_loss = -torch.sum(target_policies * F.log_softmax(pred_policy_logits, dim=1)) / boards.size(0)
                value_loss = F.mse_loss(pred_values, target_values)
                loss = policy_loss + value_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Log the losses.
            writer.add_scalar("Loss/Total", loss.item(), global_step)
            writer.add_scalar("Loss/Policy", policy_loss.item(), global_step)
            writer.add_scalar("Loss/Value", value_loss.item(), global_step)
            global_step += 1
            print("Epoch {}, Batch {}: Loss = {:.4f}".format(epoch, i // batch_size, loss.item()))
    return global_step

# ----------------------------
# 7. Checkpointing
# ----------------------------
CHECKPOINT_PATH = "chess_agent_checkpoint.pth"

def save_checkpoint(model, optimizer, iteration, training_data):
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_data': training_data,
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print("Checkpoint saved at iteration", iteration)

def load_checkpoint(model, optimizer):
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        iteration = checkpoint.get('iteration', 0)
        training_data = checkpoint.get('training_data', [])
        print("Checkpoint loaded from iteration", iteration)
        return iteration, training_data
    return 0, []

# ----------------------------
# 8. Main Loop with Parallel Self-Play, Epsilon Decay, and TensorBoard Logging
# ----------------------------
if __name__ == "__main__":
    model = ChessNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # TensorBoard writer for logging losses and epsilon.
    writer = SummaryWriter(log_dir="runs/chess_agent")
    
    start_iteration, training_data = load_checkpoint(model, optimizer)
    
    # Create a TorchScript traced version for faster inference during MCTS.
    dummy_board = torch.zeros((1, 12, 8, 8), device=device)
    inference_model = torch.jit.trace(model, dummy_board)
    
    num_self_play_games = 10    # Number of concurrent games.
    num_iterations = 5          # Total training iterations.
    num_simulations = 50        # Base number of MCTS simulations per move.
    
    # Epsilon decay parameters.
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = (epsilon - epsilon_min) / num_iterations
    
    global_step = 0  # For TensorBoard logging.
    
    for iteration in range(start_iteration, num_iterations):
        print("=== Iteration:", iteration, "===")
        writer.add_scalar("Epsilon", epsilon, iteration)
        
        # Parallel self-play using ThreadPoolExecutor.
        examples_list = []
        with ThreadPoolExecutor(max_workers=num_self_play_games) as executor:
            futures = [executor.submit(self_play_game, inference_model, num_simulations, epsilon)
                       for _ in range(num_self_play_games)]
            for future in as_completed(futures):
                examples, pgn_game = future.result()
                training_data.extend(examples)
                print("PGN Game:\n")
                print(pgn_game)
                print("\n---------------------\n")
        
        # Train the model on the accumulated training data.
        global_step = train_model(model, optimizer, training_data, writer, global_step, batch_size=32, epochs=1)
        save_checkpoint(model, optimizer, iteration + 1, training_data)
        # Update TorchScript inference model.
        inference_model = torch.jit.trace(model, dummy_board)
        
        # Decay epsilon after each iteration.
        epsilon = max(epsilon_min, epsilon - epsilon_decay)
    
    writer.close()
