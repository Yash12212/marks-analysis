import numpy as np
# Patch for deprecated np.int
if not hasattr(np, 'int'):
    np.int = int

import gym
import gym_chess
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape  # e.g., (8, 8, 119) for ChessAlphaZero-v0
        self.action_size = action_size  # e.g., 4672 actions
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95            # Discount factor
        self.epsilon = 1.0           # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Define the model using Keras functional API.
        inputs = Input(shape=self.state_shape)
        x = layers.Flatten()(inputs)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(self.action_size, activation='linear')(x)
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, legal_actions):
        if np.random.rand() <= self.epsilon:
            return random.choice(legal_actions)
        q_values = self.model.predict(state, verbose=0)
        best_action = np.argmax(q_values[0])
        if best_action in legal_actions:
            return best_action
        else:
            return random.choice(legal_actions)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent(episodes=1000, batch_size=32):
    env = gym.make('ChessAlphaZero-v0')
    state_shape = env.observation_space.shape  # e.g., (8, 8, 119)
    action_size = env.action_space.n           # e.g., 4672
    agent = DQNAgent(state_shape, action_size)

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, (1, *state_shape))
        done = False
        total_reward = 0

        while not done:
            legal_actions = env.legal_actions
            action = agent.act(state, legal_actions)
            next_state, reward, done, info = env.step(action)
            if next_state is None:
                break
            next_state = np.reshape(next_state, (1, *state_shape))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        print(f"Episode {e+1}/{episodes} - Total Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.4f}")

    env.close()

if __name__ == '__main__':
    train_agent(episodes=1000, batch_size=32)
