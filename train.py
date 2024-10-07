import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import pandas as pd
from tqdm import tqdm

class SnakeGame:
    def __init__(self, width=35, height=40):
        """
        Initializes the SnakeGame environment.

        Args:
            width (int): The width of the game grid. Default is 35.
            height (int): The height of the game grid. Default is 40.
        """
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        """
        Resets the game state to start a new game.

        Returns:
            np.ndarray: The initial state of the game, including snake position, direction, and food location.
        """
        self.snake = [(self.height // 2, self.width // 2)]
        self.direction = (0, 1)  # Start moving right
        self.food = self._place_food()
        self.score = 0
        self.steps = 0
        return self._get_state()

    def _place_food(self):
        """
        Places food randomly on the grid, ensuring it doesn't overlap with the snake.

        Returns:
            tuple: The (row, column) coordinates of the food on the grid.
        """
        while True:
            food = (random.randint(0, self.height - 1), random.randint(0, self.width - 1))
            if food not in self.snake:
                return food

    def _get_state(self):
        """
        Retrieves the current state of the game as an array.

        Returns:
            np.ndarray: An array representing the state of the game, including potential collisions 
            and food locations relative to the snake's head.
        """
        head = self.snake[0]
        point_l = (head[0] + self.direction[1], head[1] - self.direction[0])
        point_r = (head[0] - self.direction[1], head[1] + self.direction[0])

        state = [
            self._is_collision((head[0] + self.direction[0], head[1] + self.direction[1])),
            self._is_collision(point_r),
            self._is_collision(point_l),
            self.direction == (0, 1),   # Right
            self.direction == (0, -1),  # Left
            self.direction == (-1, 0),  # Up
            self.direction == (1, 0),   # Down
            self.food[0] < head[0],     # Food up
            self.food[0] > head[0],     # Food down
            self.food[1] < head[1],     # Food left
            self.food[1] > head[1]      # Food right
        ]
        return np.array(state, dtype=int)

    def _is_collision(self, point):
        """
        Checks if a given point collides with the snake or goes out of bounds.

        Args:
            point (tuple): The (row, column) coordinates to check for collision.

        Returns:
            bool: True if there is a collision, False otherwise.
        """
        return (point in self.snake[1:] or
                point[0] >= self.height or
                point[0] < 0 or
                point[1] >= self.width or
                point[1] < 0)

    def step(self, action):
        """
        Performs a game step based on the chosen action and updates the game state.

        Args:
            action (int): The action to perform (0: no turn, 1: right turn, 2: left turn).

        Returns:
            tuple: A tuple containing:
                - np.ndarray: The next state of the game.
                - float: The reward for the action taken.
                - bool: A flag indicating whether the game is over.
        """
        clock_wise = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        idx = clock_wise.index(self.direction)

        if action == 1:  # right turn
            new_dir = clock_wise[(idx + 1) % 4]
        elif action == 2:  # left turn
            new_dir = clock_wise[(idx - 1) % 4]
        else:
            new_dir = self.direction

        self.direction = new_dir
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        done = self._is_collision(new_head)
        reward = -10 if done else 0

        if not done:
            self.snake.insert(0, new_head)
            if new_head == self.food:
                self.score += 1
                reward = 10
                self.food = self._place_food()
            else:
                self.snake.pop()

        self.steps += 1
        if self.steps > 100 * len(self.snake):  # Prevent infinite loops
            done = True

        return self._get_state(), reward, done

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the Deep Q-Network (DQN) model.

        Args:
            input_size (int): The size of the input layer (number of input features).
            hidden_size (int): The size of the hidden layers.
            output_size (int): The number of possible actions (output layer size).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        """
        Forward pass of the DQN model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output predictions for the given input.
        """
        return self.net(x)


class SnakeAI:
    def __init__(self):
        """
        Attributes:
            n_games (int): The number of games played.
            epsilon (float): The exploration rate for the epsilon-greedy policy.
            gamma (float): The discount rate for future rewards.
            memory (deque): A deque to store experience tuples for training.
            model (DQN): The neural network model used for decision making.
            optimizer (Optimizer): The optimizer for training the model.
            criterion (Loss): The loss function used to compute the loss.
            batch_size (int): The size of the batch for training.
            epsilon_decay (float): The rate at which epsilon decreases.
            epsilon_min (float): The minimum value for epsilon.
            metrics (list): A list to store training metrics for saving.
        """
        self.n_games = 0
        self.epsilon = 0.9
        self.gamma = 0.9
        self.memory = deque(maxlen=100000)
        self.model = DQN(11, 32, 3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.batch_size = 1000
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.metrics = []

    def remember(self, state, action, reward, next_state, done):
        """
        Stores an experience tuple in memory for later training.

        Args:
            state (np.ndarray): The state before taking the action.
            action (int): The action taken.
            reward (float): The reward received after taking the action.
            next_state (np.ndarray): The state after taking the action.
            done (bool): A flag indicating whether the episode has ended.
        """
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        """
        Performs a training step by sampling a batch of experiences from memory,
        computing the loss, and updating the model weights accordingly.

        This function does nothing if there are not enough experiences in memory.
        """
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.bool)

        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.model(next_states).max(1)[0]
        target_q = rewards + (1 - dones.float()) * self.gamma * next_q

        loss = self.criterion(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, state, train=True):
        """
        Determines the action to take based on the current state using the 
        epsilon-greedy policy.

        Args:
            state (np.ndarray): The current state of the game.
            train (bool): A flag indicating whether the model is in training mode.

        Returns:
            int: The action to take (0, 1, or 2).
        """
        if train and random.random() < self.epsilon:
            return random.randint(0, 2)

        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        prediction = self.model(state)
        return torch.argmax(prediction).item()

    def train(self, n_episodes=1000):
        """
        Trains the SnakeAI model over a specified number of episodes.

        Args:
            n_episodes (int): The number of episodes to train the model for.
        """
        scores = []
        for episode in tqdm(range(n_episodes), desc="Training Progress", unit="episode"):
            game = SnakeGame()
            state = game.reset()
            done = False
            score = 0
        
            while not done:
                action = self.get_action(state)
                next_state, reward, done = game.step(action)
                score += reward
                self.remember(state, action, reward, next_state, done)
                state = next_state
                self.train_step()

            self.n_games += 1
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            scores.append(score)
            self.metrics.append({
                'episode': episode,
                'score': score,
                'epsilon': self.epsilon
            })

            if episode % 100 == 0:
                avg_score = sum(scores[-100:]) / 100
                tqdm.write(f'Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {self.epsilon:.2f}')

        self.save_metrics()
        self.save()

    def save_metrics(self):
        """
        Saves the training metrics to a CSV file for analysis.
        """
        df = pd.DataFrame(self.metrics)
        df.to_csv('training_metrics.csv', index=False)
        print("Training metrics saved successfully!")

    def save(self):
        """
        Saves the trained model weights to a file for future use.
        """
        torch.save(self.model.state_dict(), 'snake_ai_model.pth')
        print("Model saved successfully!")


if __name__ == "__main__":
    ai = SnakeAI()
    ai.train(1000)  