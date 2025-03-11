import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import math
import random
from concurrent.futures import ProcessPoolExecutor
import threading
from functools import lru_cache

#########################################
# Global Device Setup
#########################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#########################################
# 1. Game Environment
#########################################
class TicTacToe6x6:
    def __init__(self):
        self.board = np.zeros((6, 6), dtype=int)  # 6x6 board: 0=empty, 1=player1, -1=player2
        self.current_player = 1

    def reset(self):
        self.board[:] = 0
        self.current_player = 1
        return self.get_state()

    def get_state(self):
        return self.board.copy()

    def available_moves(self):
        return list(zip(*np.where(self.board == 0)))

    def step(self, move):
        i, j = move
        if self.board[i, j] != 0:
            raise ValueError("Illegal move!")
        self.board[i, j] = self.current_player
        reward, done = self.check_game_over(i, j)
        self.current_player *= -1
        return self.get_state(), reward, done

    def check_game_over(self, i, j):
        player = self.board[i, j]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for di, dj in directions:
            count = 1
            for sign in [1, -1]:
                ni, nj = i, j
                while True:
                    ni += sign * di
                    nj += sign * dj
                    if 0 <= ni < 6 and 0 <= nj < 6 and self.board[ni, nj] == player:
                        count += 1
                    else:
                        break
            if count >= 4:
                return 1, True
        if np.all(self.board != 0):
            return 0, True
        return 0, False

def is_terminal(game_state):
    return len(game_state.available_moves()) == 0

def get_terminal_value(game_state):
    board = game_state.get_state()
    for i in range(6):
        for j in range(6):
            if board[i, j] != 0:
                player = board[i, j]
                for di, dj in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                    count = 1
                    for sign in [1, -1]:
                        ni, nj = i, j
                        while True:
                            ni += sign * di
                            nj += sign * dj
                            if 0 <= ni < 6 and 0 <= nj < 6 and board[ni, nj] == player:
                                count += 1
                            else:
                                break
                    if count >= 4:
                        return 1 if player == 1 else -1
    return 0

#########################################
# 2. Neural Network with Residual Blocks and SE module
#########################################
class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout_prob=0.3, use_se=True, se_reduction=16):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout_prob)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)
        self.use_se = use_se
        if use_se:
            self.se_fc1 = nn.Linear(channels, channels // se_reduction)
            self.se_fc2 = nn.Linear(channels // se_reduction, channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        if self.use_se:
            # Squeeze: Global Average Pooling.
            b, c, _, _ = out.size()
            se = F.adaptive_avg_pool2d(out, 1).view(b, c)
            se = F.relu(self.se_fc1(se))
            se = torch.sigmoid(self.se_fc2(se)).view(b, c, 1, 1)
            out = out * se
        out += residual
        return F.relu(out)

class TicTacToeNet(nn.Module):
    def __init__(self, num_res_blocks=3):
        super(TicTacToeNet, self).__init__()
        self.conv = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.bn   = nn.BatchNorm2d(64)
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])
        self.fc_policy = nn.Linear(64 * 6 * 6, 36)
        self.fc_value  = nn.Linear(64 * 6 * 6, 1)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.res_blocks(x)
        x = x.view(x.size(0), -1)
        policy_logits = self.fc_policy(x)
        value = torch.tanh(self.fc_value(x))
        return policy_logits, value

def state_to_tensor(board, current_player):
    board_tensor = np.zeros((2, 6, 6), dtype=np.float32)
    board_tensor[0] = (board == current_player).astype(np.float32)
    board_tensor[1] = (board == -current_player).astype(np.float32)
    tensor = torch.tensor(board_tensor).unsqueeze(0)
    return tensor.to(device)

#########################################
# 3. Refined Batched MCTS with Adaptive Simulation & Progressive Widening and Caching
#########################################
class MCTSNode:
    def __init__(self, state, parent=None, move=None, prior=0.0, current_player=1):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = {}  # move -> child node
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.current_player = current_player

    def q_value(self):
        return 0 if self.visit_count == 0 else self.value_sum / self.visit_count

def uct_value(parent, child, c_puct):
    return child.q_value() + c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)

# Simple LRU-style cache for evaluations (max size 10000)
class EvalCache:
    def __init__(self, max_size=10000):
        self.cache = {}
        self.max_size = max_size
    def get(self, key):
        return self.cache.get(key, None)
    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            # Remove a random key (for simplicity)
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value

eval_cache = EvalCache()

def hash_state(state, current_player):
    # Use board.tobytes() and current player for hashing
    return (current_player, state.get_state().tobytes())

def mcts_search(root, network, num_simulations=100, c_puct=1.5, batch_size=16, convergence_threshold=0.01):
    simulation = 0
    leaf_values = []
    while simulation < num_simulations:
        batch_leaves = []  # list of (leaf_node, state_copy, search_path)
        while simulation < num_simulations and len(batch_leaves) < batch_size:
            node = root
            state_copy = copy.deepcopy(root.state)
            search_path = [node]
            done = False
            # SELECTION
            while node.children and not done:
                best_move, best_child = None, None
                best_uct = -float('inf')
                for move, child in node.children.items():
                    uct = uct_value(node, child, c_puct)
                    if uct > best_uct:
                        best_uct = uct
                        best_move = move
                        best_child = child
                _, reward, done = state_copy.step(best_move)
                node = best_child
                search_path.append(node)
                if is_terminal(state_copy):
                    done = True
            batch_leaves.append((node, state_copy, search_path))
            simulation += 1

        # Prepare batch for evaluation (non-terminal leaves)
        batch_tensors = []
        eval_indices = []
        for idx, (leaf_node, state_copy, search_path) in enumerate(batch_leaves):
            if not is_terminal(state_copy):
                board = state_copy.get_state()
                batch_tensors.append(state_to_tensor(board, state_copy.current_player))
                eval_indices.append(idx)
        if batch_tensors:
            batch_input = torch.cat(batch_tensors, dim=0)
            network.eval()
            with torch.no_grad():
                policy_logits_batch, value_batch = network(batch_input)
            policy_batch = torch.softmax(policy_logits_batch, dim=1).cpu().numpy()
        # Process each leaf in batch
        for idx, (leaf_node, state_copy, search_path) in enumerate(batch_leaves):
            if is_terminal(state_copy):
                node_value = get_terminal_value(state_copy)
            else:
                try:
                    batch_idx = eval_indices.index(idx)
                except ValueError:
                    batch_idx = None
                if batch_idx is not None:
                    policy = policy_batch[batch_idx].flatten()
                    value = value_batch[batch_idx].item()
                    legal_moves = state_copy.available_moves()
                    # Progressive widening: limit expansion to top-k moves.
                    allowed = max(1, int(math.sqrt(leaf_node.visit_count + 1)))
                    move_priors = []
                    for move in legal_moves:
                        pos = move[0] * 6 + move[1]
                        move_priors.append((move, policy[pos]))
                    move_priors.sort(key=lambda x: x[1], reverse=True)
                    for i, (move, p) in enumerate(move_priors):
                        if i < allowed and move not in leaf_node.children:
                            leaf_node.children[move] = MCTSNode(
                                state=None,
                                parent=leaf_node,
                                move=move,
                                prior=p,
                                current_player=state_copy.current_player
                            )
                    node_value = value
                    leaf_values.append(value)
                else:
                    node_value = 0
            for n in reversed(search_path):
                n.visit_count += 1
                n.value_sum += node_value
                node_value = -node_value
        # Check for convergence of leaf values
        if len(leaf_values) > 0:
            std_val = np.std(leaf_values)
            if std_val < convergence_threshold:
                break
    return root

def select_move(root, temperature=1.0):
    moves = list(root.children.keys())
    counts = np.array([child.visit_count for child in root.children.values()], dtype=np.float32)
    if temperature == 0:
        best_idx = np.argmax(counts)
        policy = np.zeros_like(counts)
        policy[best_idx] = 1.0
        return moves[best_idx], policy
    else:
        counts_temp = counts ** (1 / temperature)
        policy = counts_temp / np.sum(counts_temp)
        chosen_move = random.choices(moves, weights=policy)[0]
        return chosen_move, policy

#########################################
# 4. Data Augmentation via Board Symmetries
#########################################
def get_symmetries(board, policy_target):
    # Returns list of (augmented_board, augmented_policy) pairs
    aug_states = []
    board = np.array(board)
    policy_grid = policy_target.reshape(6, 6)
    for k in range(4):
        # Rotate k times 90 degrees.
        rot_board = np.rot90(board, k)
        rot_policy = np.rot90(policy_grid, k).flatten()
        aug_states.append((rot_board.copy(), rot_policy.copy()))
        # Reflect horizontally.
        ref_board = np.fliplr(rot_board)
        ref_policy = np.fliplr(rot_policy.reshape(6, 6)).flatten()
        aug_states.append((ref_board.copy(), ref_policy.copy()))
    return aug_states

#########################################
# 5. Advanced Prioritized Replay Buffer and Training Stability
#########################################
# Our replay buffer stores tuples: (state, policy_target, outcome, priority)
beta = 0.4  # Importance sampling exponent

def sample_replay_buffer(replay_buffer, batch_size):
    priorities = np.array([item[3] for item in replay_buffer], dtype=np.float32)
    probs = priorities / priorities.sum() if priorities.sum() > 0 else np.ones_like(priorities) / len(priorities)
    indices = np.random.choice(len(replay_buffer), size=batch_size, replace=False, p=probs)
    samples = [replay_buffer[i] for i in indices]
    # Compute importance sampling weights.
    weights = (1.0 / (len(replay_buffer) * probs[indices])) ** beta
    weights /= weights.max()
    return samples, indices, weights

def train_network(network, optimizer, replay_buffer, batch_size=32, max_grad_norm=1.0):
    network.train()
    if len(replay_buffer) < batch_size:
        return 0
    batch, indices, weights = sample_replay_buffer(replay_buffer, batch_size)
    # Data augmentation: for each sample, we also use symmetric variants.
    batch_states = []
    batch_policy_targets = []
    batch_value_targets = []
    for state, policy_target, outcome, _ in batch:
        # Original
        batch_states.append(state)
        batch_policy_targets.append(policy_target)
        batch_value_targets.append(outcome)
        # Augmentations
        aug_data = get_symmetries(state, policy_target)
        for aug_state, aug_policy in aug_data:
            batch_states.append(aug_state)
            batch_policy_targets.append(aug_policy)
            batch_value_targets.append(outcome)
    batch_tensors = []
    for state in batch_states:
        tensor_state = state_to_tensor(state, 1)
        batch_tensors.append(tensor_state)
    batch_input = torch.cat(batch_tensors, dim=0)
    policy_logits, values = network(batch_input)
    policy_targets = torch.tensor(batch_policy_targets, dtype=torch.float32, device=device)
    value_targets = torch.tensor(batch_value_targets, dtype=torch.float32, device=device).unsqueeze(1)
    target_indices = torch.argmax(policy_targets, dim=1)
    loss_policy = F.cross_entropy(policy_logits, target_indices, reduction='none')
    loss_value = F.mse_loss(values, value_targets, reduction='none')
    loss = loss_policy + loss_value.squeeze()
    # Split loss back into original sample losses (averaging over augmentations)
    # For simplicity, we take the mean here.
    loss = loss.mean()
    # Reweight loss with importance sampling weights.
    loss = loss * torch.tensor(weights, dtype=torch.float32, device=device).mean()
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping.
    torch.nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm)
    optimizer.step()
    new_priority = loss.item()
    for idx in indices:
        replay_buffer[idx] = (replay_buffer[idx][0], replay_buffer[idx][1], replay_buffer[idx][2], new_priority + 1e-5)
    return loss.item()

#########################################
# 6. Multi-Process Self-Play and Asynchronous Checkpointing
#########################################
def self_play_game(network_state, base_num_simulations=100):
    """
    Runs a self-play game using a local network instance loaded from network_state.
    """
    # Rebuild local network and load weights.
    local_net = TicTacToeNet(num_res_blocks=3).to(device)
    local_net.load_state_dict(network_state)
    local_net.eval()
    
    game = TicTacToe6x6()
    states, mcts_policies, outcomes = [], [], []
    move_count = 0
    root = None
    last_move = None
    local_eval_cache = EvalCache()
    while True:
        if root is None or last_move is None:
            root = MCTSNode(state=copy.deepcopy(game), current_player=game.current_player)
        else:
            new_root = root.children.get(last_move, None)
            if new_root is None:
                root = MCTSNode(state=copy.deepcopy(game), current_player=game.current_player)
            else:
                new_root.parent = None
                root = new_root
        legal_moves = game.available_moves()
        if not root.children:
            board = game.get_state()
            tensor_state = state_to_tensor(board, game.current_player)
            local_net.eval()
            with torch.no_grad():
                policy_logits, _ = local_net(tensor_state)
            policy = torch.softmax(policy_logits, dim=1).cpu().numpy().flatten()
            epsilon = 0.25
            alpha = 0.3
            noise = np.random.dirichlet([alpha] * len(legal_moves))
            for i, move in enumerate(legal_moves):
                pos = move[0] * 6 + move[1]
                noisy_prior = (1 - epsilon) * policy[pos] + epsilon * noise[i]
                root.children[move] = MCTSNode(
                    state=None,
                    parent=root,
                    move=move,
                    prior=noisy_prior,
                    current_player=game.current_player
                )
        num_simulations = 50 if move_count < 5 else base_num_simulations
        root = mcts_search(root, local_net, num_simulations=num_simulations, c_puct=1.5, batch_size=16, convergence_threshold=0.01)
        chosen_move, _ = select_move(root, temperature=1.0 - move_count/20.0)
        states.append(game.get_state())
        policy_target = np.zeros(36)
        for move_key, child in root.children.items():
            pos = move_key[0] * 6 + move_key[1]
            policy_target[pos] = child.visit_count
        if np.sum(policy_target) > 0:
            policy_target /= np.sum(policy_target)
        mcts_policies.append(policy_target)
        _, reward, done = game.step(chosen_move)
        last_move = chosen_move
        move_count += 1
        if done:
            outcome = reward
            outcomes = [outcome] * len(states)
            break
    return states, mcts_policies, outcomes

# Asynchronous checkpoint saving.
def async_checkpoint(network_state, filename):
    def save_state(state, fname):
        torch.save(state, fname)
    t = threading.Thread(target=save_state, args=(network_state, filename))
    t.start()

#########################################
# 7. Regular Evaluation Matches
#########################################
def evaluate_network(current_net, opponent_net, num_games=10):
    current_wins = 0
    for game_idx in range(num_games):
        game = TicTacToe6x6()
        current_player = 1 if game_idx % 2 == 0 else -1
        while True:
            board = game.get_state()
            tensor_state = state_to_tensor(board, game.current_player)
            current_net.eval()
            opponent_net.eval()
            with torch.no_grad():
                if game.current_player == current_player:
                    policy_logits, _ = current_net(tensor_state)
                else:
                    policy_logits, _ = opponent_net(tensor_state)
            policy = torch.softmax(policy_logits, dim=1).cpu().numpy().flatten()
            legal_moves = game.available_moves()
            move_probs = {}
            for move in legal_moves:
                pos = move[0]*6 + move[1]
                move_probs[move] = policy[pos]
            best_move = max(move_probs, key=move_probs.get)
            _, reward, done = game.step(best_move)
            if done:
                if reward == 1 and game.current_player != current_player:
                    current_wins += 1
                break
    win_rate = current_wins / num_games
    return win_rate

#########################################
#########################################
# 8. Main Training Loop with Multiprocessing Self-Play and Checkpointing
#########################################
if __name__ == "__main__":
    network = TicTacToeNet(num_res_blocks=3).to(device)
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    num_iterations = 20
    games_per_iteration = 10
    max_replay_buffer_size = 1000
    replay_buffer = []  # Each sample: (state, policy_target, outcome, priority)
    
    # Placeholder for best network (for evaluation).
    best_network = TicTacToeNet(num_res_blocks=3).to(device)
    best_network.load_state_dict(network.state_dict())
    
    # Use ProcessPoolExecutor for multi-process self-play.
    with ProcessPoolExecutor(max_workers=4) as executor:
        for iteration in range(num_iterations):
            # Update network state to pass to workers.
            network_state = network.state_dict()
            futures = [executor.submit(self_play_game, network_state, base_num_simulations=100) 
                       for _ in range(games_per_iteration)]
            for future in futures:
                states, policies, outcomes = future.result()
                for s, p, o in zip(states, policies, outcomes):
                    replay_buffer.append((s, p, o, 1.0))
                if len(replay_buffer) > max_replay_buffer_size:
                    replay_buffer = replay_buffer[-max_replay_buffer_size:]
            for _ in range(20):
                loss = train_network(network, optimizer, replay_buffer, batch_size=32, max_grad_norm=1.0)
            scheduler.step()
            if iteration % 10 == 0:
                async_checkpoint(network.state_dict(), f"checkpoint_{iteration}.pth")
                eval_win_rate = evaluate_network(network, best_network, num_games=10)
                print(f"Iteration {iteration}, Loss: {loss:.4f}, Replay Buffer Size: {len(replay_buffer)}, Eval Win Rate: {eval_win_rate:.2f}")
                # Update best network if current performance is improved.
                if eval_win_rate > 0.55:
                    best_network.load_state_dict(network.state_dict())
                    
    # For hyperparameter tuning, consider writing wrappers that loop over grids of parameters.
    # E.g., for dropout_rate in [0.2, 0.3, 0.4]: for lr in [0.001, 0.0005]: ... etc.