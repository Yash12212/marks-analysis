import os
import random
import logging
import argparse
from dataclasses import dataclass
from typing import Tuple, List, Optional, Callable

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class AntiTicTacToe:
    def __init__(self) -> None:
        self.board = [0] * 9
        self.current_player = 1
        self.winner: Optional[int] = None

    def reset(self) -> Tuple[Tuple[int, ...], int]:
        self.board = [0] * 9
        self.current_player = 1
        self.winner = None
        return self.get_state()

    def get_state(self) -> Tuple[Tuple[int, ...], int]:
        return (tuple(self.board), self.current_player)

    def available_moves(self) -> List[int]:
        return [i for i, cell in enumerate(self.board) if cell == 0]

    def check_loss(self, player: int) -> bool:
        b = self.board
        lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                 (0, 3, 6), (1, 4, 7), (2, 5, 8),
                 (0, 4, 8), (2, 4, 6)]
        for line in lines:
            if b[line[0]] == b[line[1]] == b[line[2]] == player:
                return True
        return False

    def step(self, move: int) -> Tuple[Tuple[Tuple[int, ...], int], int, bool]:
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
            self.winner = 0  # Draw
        else:
            reward = 0
            done = False
        if not done:
            self.current_player = 3 - self.current_player
        return self.get_state(), reward, done

    def render(self) -> None:
        symbols = {0: " ", 1: "X", 2: "O"}
        board_str = ""
        for i in range(9):
            board_str += symbols[self.board[i]]
            board_str += "\n" if (i + 1) % 3 == 0 else "|"
        print(board_str)

def canonicalize_state_with_transform(
    state: Tuple[Tuple[int, ...], int]
) -> Tuple[Tuple[Tuple[int, ...], int], List[int]]:
    board, current_player = state
    board = tuple(board)

    def identity(r: int, c: int) -> Tuple[int, int]:
        return (r, c)
    def rot90(r: int, c: int) -> Tuple[int, int]:
        return (c, 2 - r)
    def rot180(r: int, c: int) -> Tuple[int, int]:
        return (2 - r, 2 - c)
    def rot270(r: int, c: int) -> Tuple[int, int]:
        return (2 - c, r)
    def ref_horizontal(r: int, c: int) -> Tuple[int, int]:
        return (r, 2 - c)
    def ref_vertical(r: int, c: int) -> Tuple[int, int]:
        return (2 - r, c)
    def ref_main_diag(r: int, c: int) -> Tuple[int, int]:
        return (c, r)
    def ref_anti_diag(r: int, c: int) -> Tuple[int, int]:
        return (2 - c, 2 - r)

    trans_funcs: List[Callable[[int, int], Tuple[int, int]]] = [
        identity, rot90, rot180, rot270,
        ref_horizontal, ref_vertical, ref_main_diag, ref_anti_diag
    ]

    def transform_board_and_mapping(
        board: Tuple[int, ...], func: Callable[[int, int], Tuple[int, int]]
    ) -> Tuple[Tuple[int, ...], List[int]]:
        new_board = [None] * 9
        mapping = [None] * 9
        for i in range(9):
            r, c = divmod(i, 3)
            new_r, new_c = func(r, c)
            new_index = new_r * 3 + new_c
            new_board[new_index] = board[i]
            mapping[i] = new_index
        return tuple(new_board), mapping

    candidates = []
    for func in trans_funcs:
        tb, mapping = transform_board_and_mapping(board, func)
        candidates.append((tb, mapping))
    canonical_board, best_mapping = min(candidates, key=lambda x: x[0])
    return (canonical_board, current_player), best_mapping

def state_to_tensor(state: Tuple[Tuple[int, ...], int]) -> torch.FloatTensor:
    board, current_player = state
    board_arr = np.array(board)
    board_tensor = np.where(board_arr == current_player, 1, np.where(board_arr == 0, 0, -1))
    return torch.FloatTensor(board_tensor).to(device)

class DQN(nn.Module):
    def __init__(self, input_dim: int = 9, hidden_dim: int = 128, output_dim: int = 9) -> None:
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

@dataclass
class Transition:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool

class ReplayBuffer:
    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = capacity
        self.buffer: List[Transition] = []
        self.position = 0

    def push(self, transition: Transition) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)

class EpsilonScheduler:
    def __init__(self, start: float = 1.0, end: float = 0.1, decay: float = 0.9995) -> None:
        self.epsilon = start
        self.epsilon_end = end
        self.decay = decay

    def step(self) -> float:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.decay)
        return self.epsilon

    def get_epsilon(self) -> float:
        return self.epsilon

class DQNAgent:
    def __init__(
        self,
        lr: float = 0.001,
        hidden_dim: int = 128,
        capacity: int = 10000,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.1,
        epsilon_decay: float = 0.9995,
        gamma: float = 1.0,
        target_update: int = 1000,
        lr_step_size: int = 10000,
        lr_gamma: float = 0.9
    ) -> None:
        self.net = DQN(hidden_dim=hidden_dim).to(device)
        self.target_net = DQN(hidden_dim=hidden_dim).to(device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=lr_step_size, gamma=lr_gamma)
        self.buffer = ReplayBuffer(capacity)
        self.epsilon_scheduler = EpsilonScheduler(epsilon_start, epsilon_min, epsilon_decay)
        self.gamma = gamma
        self.target_update = target_update
        self.steps_done = 0

    def select_action(self, state: Tuple[Tuple[int, ...], int], legal_moves: List[int]) -> Tuple[int, int]:
        canonical_state, mapping = canonicalize_state_with_transform(state)
        inverse_mapping = [None] * 9
        for i, m in enumerate(mapping):
            inverse_mapping[m] = i
        if random.random() < self.epsilon_scheduler.get_epsilon():
            actual_move = random.choice(legal_moves)
            canonical_move = mapping[actual_move]
        else:
            state_tensor = state_to_tensor(canonical_state)
            q_values = self.net(state_tensor).detach().cpu().numpy()
            legal_moves_canonical = [(m, mapping[m]) for m in legal_moves]
            best = max(legal_moves_canonical, key=lambda x: q_values[x[1]])
            canonical_move = best[1]
            actual_move = inverse_mapping[canonical_move]
        return canonical_move, actual_move

    def optimize_model(self, batch_size: int) -> Optional[float]:
        if len(self.buffer) < batch_size:
            return None
        transitions = self.buffer.sample(batch_size)
        batch_state = torch.stack([t.state for t in transitions])
        batch_action = torch.LongTensor([t.action for t in transitions]).to(device)
        batch_reward = torch.FloatTensor([t.reward for t in transitions]).to(device)
        batch_next_state = torch.stack([t.next_state for t in transitions])
        batch_done = torch.FloatTensor([t.done for t in transitions]).to(device)
        q_values = self.net(batch_state)
        state_action_values = q_values.gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_actions = self.net(batch_next_state).argmax(dim=1, keepdim=True)
        next_q_values = self.target_net(batch_next_state).gather(1, next_actions).squeeze(1)
        expected_state_action_values = batch_reward + self.gamma * next_q_values * (1 - batch_done)
        loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.net.state_dict())

    def save_model(self, model_path: str = "dqn_model.pt") -> None:
        torch.save(self.net.state_dict(), model_path)
        logger.info("Model saved to %s", model_path)

    def load_model(self, model_path: str = "dqn_model.pt") -> None:
        self.net.load_state_dict(torch.load(model_path, map_location=device))
        self.update_target_network()
        logger.info("Model loaded from %s", model_path)

    def train(
        self,
        num_episodes: int = 50000,
        batch_size: int = 64,
        log_interval: int = 1000,
        model_path: str = "dqn_model.pt"
    ) -> None:
        env = AntiTicTacToe()
        writer = SummaryWriter("runs/anti_ttt")
        total_updates = 0
        for episode in tqdm(range(num_episodes), desc="Training Episodes"):
            state = env.reset()
            done = False
            episode_loss = 0.0
            updates = 0
            while not done:
                canonical_state, mapping = canonicalize_state_with_transform(state)
                inverse_mapping = [None] * 9
                for i, m in enumerate(mapping):
                    inverse_mapping[m] = i
                legal_moves = env.available_moves()
                _, actual_move = self.select_action(state, legal_moves)
                try:
                    next_state, reward, done = env.step(actual_move)
                except ValueError as e:
                    logger.error("Encountered error: %s", e)
                    continue
                if not done:
                    next_canonical, _ = canonicalize_state_with_transform(next_state)
                    next_tensor = state_to_tensor(next_canonical)
                else:
                    next_tensor = torch.zeros(9).to(device)
                current_tensor = state_to_tensor(canonical_state)
                transition = Transition(current_tensor, mapping[actual_move], reward, next_tensor, done)
                self.buffer.push(transition)
                state = next_state
                self.steps_done += 1
                loss = self.optimize_model(batch_size)
                if loss is not None:
                    episode_loss += loss
                    updates += 1
                    total_updates += 1
                if self.steps_done % self.target_update == 0:
                    self.update_target_network()
            if updates > 0:
                writer.add_scalar("Loss", episode_loss / updates, episode)
            writer.add_scalar("Epsilon", self.epsilon_scheduler.get_epsilon(), episode)
            self.epsilon_scheduler.step()
            if (episode + 1) % log_interval == 0:
                avg_loss = episode_loss / updates if updates > 0 else 0.0
                logger.info("Episode %d/%d, Epsilon: %.4f, Total Updates: %d, Avg Loss: %.4f",
                            episode + 1, num_episodes, self.epsilon_scheduler.get_epsilon(),
                            total_updates, avg_loss)
        writer.close()
        self.save_model(model_path)
        logger.info("Training complete.")

    def evaluate_against_random(self, num_episodes: int = 1000) -> None:
        wins, losses, draws = 0, 0, 0
        for _ in range(num_episodes):
            env = AntiTicTacToe()
            state = env.reset()
            # Randomly assign agent as player 1 or 2.
            agent_player = random.choice([1, 2])
            # If agent is not starting, let the random opponent move first.
            if env.current_player != agent_player:
                legal_moves = env.available_moves()
                state, _, _ = env.step(random.choice(legal_moves))
            done = False
            while not done:
                if env.current_player == agent_player:
                    canonical_state, mapping = canonicalize_state_with_transform(state)
                    inverse_mapping = [None] * 9
                    for i, m in enumerate(mapping):
                        inverse_mapping[m] = i
                    legal_moves = env.available_moves()
                    state_tensor = state_to_tensor(canonical_state)
                    q_values = self.net(state_tensor).detach().cpu().numpy()
                    legal_moves_canonical = [(m, mapping[m]) for m in legal_moves]
                    best = max(legal_moves_canonical, key=lambda x: q_values[x[1]])
                    canonical_move = best[1]
                    actual_move = inverse_mapping[canonical_move]
                    state, reward, done = env.step(actual_move)
                else:
                    legal_moves = env.available_moves()
                    state, reward, done = env.step(random.choice(legal_moves))
            if reward == -1:
                if env.winner == agent_player:
                    wins += 1
                else:
                    losses += 1
            else:
                draws += 1
        total = wins + losses + draws
        logger.info("Evaluation Against Random over %d episodes: Wins: %d (%.2f%%), Losses: %d (%.2f%%), Draws: %d (%.2f%%)",
                    total, wins, wins/total*100, losses, losses/total*100, draws, draws/total*100)

    def evaluate_self_play(self, num_episodes: int = 1000) -> None:
        wins_player1, wins_player2, draws = 0, 0, 0
        for _ in range(num_episodes):
            env = AntiTicTacToe()
            state = env.reset()
            done = False
            while not done:
                canonical_state, mapping = canonicalize_state_with_transform(state)
                inverse_mapping = [None] * 9
                for i, m in enumerate(mapping):
                    inverse_mapping[m] = i
                legal_moves = env.available_moves()
                state_tensor = state_to_tensor(canonical_state)
                q_values = self.net(state_tensor).detach().cpu().numpy()
                legal_moves_canonical = [(m, mapping[m]) for m in legal_moves]
                best = max(legal_moves_canonical, key=lambda x: q_values[x[1]])
                canonical_move = best[1]
                actual_move = inverse_mapping[canonical_move]
                state, reward, done = env.step(actual_move)
            if reward == -1:
                if env.winner == 1:
                    wins_player1 += 1
                elif env.winner == 2:
                    wins_player2 += 1
            else:
                draws += 1
        total = wins_player1 + wins_player2 + draws
        logger.info("Self-Play Evaluation over %d episodes: Player1 Wins: %d (%.2f%%), Player2 Wins: %d (%.2f%%), Draws: %d (%.2f%%)",
                    total, wins_player1, wins_player1/total*100, wins_player2, wins_player2/total*100, draws, draws/total*100)

    def evaluate(self, num_episodes: int = 1000) -> None:
        self.evaluate_against_random(num_episodes)
        self.evaluate_self_play(num_episodes)

    def play(self) -> None:
        env = AntiTicTacToe()
        state = env.reset()
        print("Board indices:\n0|1|2\n3|4|5\n6|7|8\n")
        while True:
            env.render()
            if env.current_player == 1:
                try:
                    move = int(input("Enter move (0-8): "))
                except Exception:
                    continue
                if move not in env.available_moves():
                    print("Illegal move. Try again.")
                    continue
            else:
                canonical_state, mapping = canonicalize_state_with_transform(state)
                inverse_mapping = [None] * 9
                for i, m in enumerate(mapping):
                    inverse_mapping[m] = i
                legal_moves = env.available_moves()
                state_tensor = state_to_tensor(canonical_state)
                q_values = self.net(state_tensor).detach().cpu().numpy()
                legal_moves_canonical = [(m, mapping[m]) for m in legal_moves]
                best = max(legal_moves_canonical, key=lambda x: q_values[x[1]])
                canonical_move = best[1]
                move = inverse_mapping[canonical_move]
                print(f"Agent chooses move {move}")
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

def main() -> None:
    parser = argparse.ArgumentParser(description="Train, evaluate, or play the Anti-Tic Tac Toe DQN agent.")
    parser.add_argument("--mode", choices=["train", "evaluate", "play"], default="train",
                        help="Mode to run: train the agent, evaluate it, or play against it.")
    parser.add_argument("--episodes", type=int, default=50000, help="Default number of training episodes.")
    parser.add_argument("--eval_episodes", type=int, default=1000, help="Number of evaluation episodes.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--model_path", type=str, default="dqn_model.pt", help="Path to save/load the model.")
    args = parser.parse_args()
    agent = DQNAgent()
    if args.mode == "train":
        if os.path.exists(args.model_path):
            agent.load_model(args.model_path)
            logger.info("Model loaded.")
        else:
            logger.info("No existing model found.")
        try:
            training_input = input(f"Enter number of training episodes (default {args.episodes}): ")
            training_episodes = args.episodes if training_input.strip() == "" else int(training_input)
        except ValueError:
            logger.error("Invalid input. Using default %d episodes.", args.episodes)
            training_episodes = args.episodes
        if training_episodes > 0:
            agent.train(num_episodes=training_episodes, batch_size=args.batch_size, model_path=args.model_path)
        else:
            logger.info("Skipping training as 0 episodes were specified.")
        agent.evaluate(num_episodes=args.eval_episodes)
        while True:
            print("\nSelect an option:")
            print("T: Train further")
            print("L: Load a different model")
            print("P: Play")
            print("E: Evaluate")
            print("X: Exit")
            choice = input("Enter your choice: ").strip().lower()
            if choice == "t":
                try:
                    further_input = input(f"Enter number of additional training episodes (default {args.episodes}): ")
                    further_episodes = args.episodes if further_input.strip() == "" else int(further_input)
                except ValueError:
                    logger.error("Invalid input. Skipping further training.")
                    further_episodes = 0
                if further_episodes > 0:
                    agent.train(num_episodes=further_episodes, batch_size=args.batch_size, model_path=args.model_path)
                    agent.evaluate(num_episodes=args.eval_episodes)
                else:
                    logger.info("Skipping further training.")
            elif choice == "l":
                new_model = input("Enter the model path to load: ").strip()
                if os.path.exists(new_model):
                    agent.load_model(new_model)
                else:
                    logger.error("Model file not found.")
            elif choice == "p":
                while True:
                    agent.play()
                    cont = input("Play again? (y/n): ").strip().lower()
                    if cont != "y":
                        break
            elif choice == "e":
                try:
                    eval_input = input(f"Enter number of evaluation episodes (default {args.eval_episodes}): ")
                    eval_episodes = args.eval_episodes if eval_input.strip() == "" else int(eval_input)
                except ValueError:
                    logger.error("Invalid input. Using default %d evaluation episodes.", args.eval_episodes)
                    eval_episodes = args.eval_episodes
                agent.evaluate(num_episodes=eval_episodes)
            elif choice == "x":
                break
            else:
                print("Invalid option. Please try again.")
    elif args.mode == "evaluate":
        if os.path.exists(args.model_path):
            agent.load_model(args.model_path)
        else:
            logger.error("Model file not found. Train the model first.")
            return
        agent.evaluate(num_episodes=args.eval_episodes)
    elif args.mode == "play":
        if os.path.exists(args.model_path):
            agent.load_model(args.model_path)
        else:
            logger.error("Model file not found. Train the model first.")
            return
        while True:
            agent.play()
            cont = input("Play again? (y/n): ").strip().lower()
            if cont != "y":
                break

if __name__ == "__main__":
    main()
