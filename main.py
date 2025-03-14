import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class AntiTicTacToe:
    def __init__(self):
        self.board = [0] * 9
        self.current_player = 1
        self.winner = None
    def reset(self):
        self.board = [0] * 9
        self.current_player = 1
        self.winner = None
        return self.get_state()
    def get_state(self):
        return (tuple(self.board), self.current_player)
    def available_moves(self):
        return [i for i, cell in enumerate(self.board) if cell == 0]
    def check_loss(self, player):
        b = self.board
        lines = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        for line in lines:
            if b[line[0]] == player and b[line[1]] == player and b[line[2]] == player:
                return True
        return False
    def step(self, move):
        if self.board[move] != 0:
            raise ValueError("Illegal move")
        self.board[move] = self.current_player
        if self.check_loss(self.current_player):
            reward = -1
            done = True
            self.winner = 3 - self.current_player
        elif all(cell != 0 for cell in self.board):
            reward = 0
            done = True
            self.winner = 0
        else:
            reward = 0
            done = False
        if not done:
            self.current_player = 3 - self.current_player
        return self.get_state(), reward, done
    def render(self):
        symbols = {0:" ",1:"X",2:"O"}
        board_str = ""
        for i in range(9):
            board_str += symbols[self.board[i]]
            board_str += "\n" if (i+1)%3==0 else "|"
        print(board_str)

def canonicalize_state_with_transform(state):
    board, current_player = state
    board = tuple(board)
    def identity(r,c): return (r,c)
    def rot90(r,c): return (c,2-r)
    def rot180(r,c): return (2-r,2-c)
    def rot270(r,c): return (2-c,r)
    def ref_horizontal(r,c): return (r,2-c)
    def ref_vertical(r,c): return (2-r,c)
    def ref_main_diag(r,c): return (c,r)
    def ref_anti_diag(r,c): return (2-c,2-r)
    trans_funcs = [identity,rot90,rot180,rot270,ref_horizontal,ref_vertical,ref_main_diag,ref_anti_diag]
    def transform_board_and_mapping(board, func):
        new_board = [None]*9
        mapping = [None]*9
        for i in range(9):
            r,c = divmod(i,3)
            new_r,new_c = func(r,c)
            new_index = new_r*3+new_c
            new_board[new_index] = board[i]
            mapping[i] = new_index
        return tuple(new_board), mapping
    candidates = []
    for func in trans_funcs:
        tb, mapping = transform_board_and_mapping(board, func)
        candidates.append((tb, mapping))
    canonical_board, best_mapping = min(candidates, key=lambda x: x[0])
    return (canonical_board, current_player), best_mapping

def state_to_tensor(state):
    board, current_player = state
    board = np.array(board)
    board_tensor = np.where(board==current_player,1, np.where(board==0,0,-1))
    return torch.FloatTensor(board_tensor).to(device)

class DQN(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, output_dim=9):
        super(DQN,self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position+1) % self.capacity
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "dqn_model.pt"

def select_action_training(canonical_state, legal_moves, mapping, inverse_mapping, net, epsilon):
    if random.random() < epsilon:
        actual_move = random.choice(legal_moves)
        canonical_move = mapping[actual_move]
    else:
        state_tensor = state_to_tensor(canonical_state)
        q_values = net(state_tensor).detach().cpu().numpy()
        legal_moves_canonical = [(m, mapping[m]) for m in legal_moves]
        best = max(legal_moves_canonical, key=lambda x: q_values[x[1]])
        canonical_move = best[1]
        actual_move = inverse_mapping[canonical_move]
    return canonical_move, actual_move

def train_dqn(num_episodes=10000, batch_size=64, gamma=1.0, epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.9999, target_update=1000):
    env = AntiTicTacToe()
    net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(net.state_dict())
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    buffer = ReplayBuffer(10000)
    epsilon = epsilon_start
    steps_done = 0
    writer = SummaryWriter("runs/anti_ttt")
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_loss = 0.0
        count = 0
        while not done:
            canonical_state, mapping = canonicalize_state_with_transform(state)
            inverse_mapping = [None]*9
            for i, m in enumerate(mapping):
                inverse_mapping[m] = i
            legal_moves = env.available_moves()
            canonical_action, actual_action = select_action_training(canonical_state, legal_moves, mapping, inverse_mapping, net, epsilon)
            next_state, reward, done = env.step(actual_action)
            next_tensor = state_to_tensor(canonicalize_state_with_transform(next_state)[0]) if not done else torch.zeros(9).to(device)
            buffer.push(state_to_tensor(canonical_state), canonical_action, reward, next_tensor, done)
            state = next_state
            steps_done += 1
            if len(buffer) >= batch_size:
                transitions = buffer.sample(batch_size)
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
                batch_state = torch.stack(batch_state)
                batch_action = torch.LongTensor(batch_action).to(device)
                batch_reward = torch.FloatTensor(batch_reward).to(device)
                batch_next_state = torch.stack(batch_next_state)
                batch_done = torch.FloatTensor(batch_done).to(device)
                q_values = net(batch_state)
                q_value = q_values.gather(1, batch_action.unsqueeze(1)).squeeze(1)
                next_q_values = target_net(batch_next_state).max(1)[0]
                target = batch_reward + gamma * next_q_values * (1 - batch_done)
                loss = nn.MSELoss()(q_value, target.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                episode_loss += loss.item()
                count += 1
            if steps_done % target_update == 0:
                target_net.load_state_dict(net.state_dict())
        if count > 0:
            writer.add_scalar("Loss", episode_loss/count, episode)
        writer.add_scalar("Epsilon", epsilon, episode)
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        if (episode+1) % 1000 == 0:
            print(f"Episode {episode+1}/{num_episodes}, epsilon: {epsilon:.4f}")
    writer.close()
    torch.save(net.state_dict(), model_path)
    return net

def play_game(net):
    env = AntiTicTacToe()
    state = env.reset()
    print("0|1|2\n3|4|5\n6|7|8\n")
    while True:
        env.render()
        if env.current_player == 1:
            try:
                move = int(input("Enter move (0-8): "))
            except:
                continue
            if move not in env.available_moves():
                continue
        else:
            canonical_state, mapping = canonicalize_state_with_transform(state)
            inverse_mapping = [None]*9
            for i, m in enumerate(mapping):
                inverse_mapping[m] = i
            legal_moves = env.available_moves()
            state_tensor = state_to_tensor(canonical_state)
            q_values = net(state_tensor).detach().cpu().numpy()
            legal_moves_canonical = [(m, mapping[m]) for m in legal_moves]
            best = max(legal_moves_canonical, key=lambda x: q_values[x[1]])
            canonical_move = best[1]
            move = inverse_mapping[canonical_move]
            print(f"Agent chooses {move}")
        state, reward, done = env.step(move)
        if done:
            env.render()
            if reward == 0:
                print("Draw!")
            elif reward == -1:
                if env.winner == 1:
                    print("Agent made a losing move. You win!")
                elif env.winner == 2:
                    print("You made a losing move. Agent wins!")
            break

if os.path.exists(model_path):
    net = DQN().to(device)
    net.load_state_dict(torch.load(model_path))
    print("Loaded model.")
else:
    net = train_dqn()

while True:
    play_game(net)
    ans = input("Play again? (y/n): ")
    if ans.lower() != "y":
        break
