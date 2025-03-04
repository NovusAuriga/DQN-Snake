import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
from torch.utils.tensorboard import SummaryWriter
import time
from colorama import init, Fore, Style

init()

# Game constants
TABLE_WIDTH = 25
TABLE_HEIGHT = 25
UP, DOWN, RIGHT, LEFT = 0, 1, 2, 3
DIRECTIONS_VECT = np.array([[0, -1], [0, 1], [1, 0], [-1, 0]], dtype=np.int32)
MAX_STEPS = 10000
STEPS_WITHOUT_FOOD_LIMIT = 1000

class DQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )
    
    def forward(self, x):
        return self.fc(x)

class SnakeGame:
    def __init__(self):
        self.max_dist = np.sqrt(TABLE_WIDTH ** 2 + TABLE_HEIGHT ** 2)
        self.move_history = []
        self.prev_position = None
        self.body_collisions = 0
        self.border_collisions = 0
        self.total_body_collisions = 0
        self.total_border_collisions = 0
        self.reset()
    
    def reset(self):
        self.table = np.zeros((TABLE_HEIGHT, TABLE_WIDTH), dtype=np.int32)
        self.snake = [(TABLE_HEIGHT // 2, TABLE_WIDTH // 2)]
        self.food = self._spawn_food()
        self.direction = random.choice([UP, DOWN, RIGHT, LEFT])
        self.table[self.snake[0][0], self.snake[0][1]] = 1
        self.table[self.food[0], self.food[1]] = 2
        self.food_eaten = 0
        self.steps_since_food = 0
        self.move_history = []
        self.prev_position = None
        self.body_collisions = 0
        self.border_collisions = 0
        return self._get_state()
    
    def _spawn_food(self):
        while True:
            food = (np.random.randint(0, TABLE_HEIGHT), np.random.randint(0, TABLE_WIDTH))
            if food not in self.snake:
                return food
    
    def _get_state(self):
        head_y, head_x = self.snake[0]
        food_y, food_x = self.food
        dist_to_food = np.sqrt((head_y - food_y) ** 2 + (head_x - food_x) ** 2) / self.max_dist
        snake_length_normalized = len(self.snake) / (TABLE_WIDTH * TABLE_HEIGHT)
        food_eaten_normalized = self.food_eaten / 100.0
        
        # 8 directional danger signals
        danger = [0] * 8  # N, NE, E, SE, S, SW, W, NW
        directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        for i, (dy, dx) in enumerate(directions):
            check_y, check_x = head_y + dy, head_x + dx
            if (check_y >= TABLE_HEIGHT or check_y < 0 or check_x >= TABLE_WIDTH or check_x < 0 or 
                (check_y, check_x) in self.snake[1:]):
                danger[i] = 1
        
        tail_y, tail_x = self.snake[-1] if len(self.snake) > 1 else (head_y, head_x)
        body1_y, body1_x = self.snake[1] if len(self.snake) > 2 else (head_y, head_x)
        body2_y, body2_x = self.snake[2] if len(self.snake) > 3 else (head_y, head_x)
        body3_y, body3_x = self.snake[3] if len(self.snake) > 4 else (head_y, head_x)
        
        dist_to_tail = np.sqrt((head_y - tail_y) ** 2 + (head_x - tail_x) ** 2) / self.max_dist
        dist_to_body1 = np.sqrt((head_y - body1_y) ** 2 + (head_x - body1_x) ** 2) / self.max_dist
        dist_to_body2 = np.sqrt((head_y - body2_y) ** 2 + (head_x - body2_x) ** 2) / self.max_dist
        dist_to_body3 = np.sqrt((head_y - body3_y) ** 2 + (head_x - body3_x) ** 2) / self.max_dist
        
        return np.array([
            head_y / TABLE_HEIGHT, head_x / TABLE_WIDTH,
            food_y / TABLE_HEIGHT, food_x / TABLE_WIDTH,
            self.direction / 3.0,
            dist_to_food,
            float(food_y < head_y), float(food_y > head_y),
            float(food_x < head_x), float(food_x > head_x),
            float(head_y == 0), float(head_y == TABLE_HEIGHT - 1),
            float(head_x == 0), float(head_x == TABLE_WIDTH - 1),
            snake_length_normalized,
            food_eaten_normalized,
            *danger,
            dist_to_tail,
            dist_to_body1,
            dist_to_body2,
            dist_to_body3,
            float(tail_y < head_y)
        ], dtype=np.float32)  # 29 inputs
    
    def step(self, action):
        head_y, head_x = self.snake[0]
        reward = 0
        done = False
        
        dy, dx = DIRECTIONS_VECT[action]
        new_head = (head_y + dy, head_x + dx)
        
        move_info = {'action': action, 'direction': self.direction, 'new_head': new_head}
        
        # Check rules
        if (new_head[0] >= TABLE_HEIGHT or new_head[0] < 0 or 
            new_head[1] >= TABLE_WIDTH or new_head[1] < 0):
            reward -= 1
            done = True
            self.border_collisions += 1
            self.total_border_collisions += 1
            self.move_history.append(move_info)
            return self._get_state(), reward, done
        elif new_head in self.snake[1:]:
            reward -= 1
            done = True
            self.body_collisions += 1
            self.total_body_collisions += 1
            self.move_history.append(move_info)
            return self._get_state(), reward, done
        
        self.snake.insert(0, new_head)
        self.steps_since_food += 1
        reward += 0.05
        if new_head == self.food:
            reward += 1
            self.food_eaten += 1
            self.food = self._spawn_food()
            self.steps_since_food = 0
        else:
            self.prev_position = self.snake.pop()
        
        if self.steps_since_food >= STEPS_WITHOUT_FOOD_LIMIT:
            reward -= 1
            done = True
        
        self.table.fill(0)
        for y, x in self.snake:
            self.table[y, x] = 1
        self.table[self.food[0], self.food[1]] = 2
        
        self.direction = action
        self.move_history.append(move_info)
        
        return self._get_state(), reward, done
    
    def render(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        for y in range(TABLE_HEIGHT):
            for x in range(TABLE_WIDTH):
                if (y, x) == self.snake[0]:
                    print(f"{Fore.GREEN}S{Style.RESET_ALL}", end=" ")
                elif (y, x) in self.snake[1:]:
                    print(f"{Fore.GREEN}s{Style.RESET_ALL}", end=" ")
                elif (y, x) == self.food:
                    print(f"{Fore.RED}F{Style.RESET_ALL}", end=" ")
                else:
                    print(".", end=" ")
            print()
        print(f"Food Eaten: {self.food_eaten}, Steps: {self.steps_since_food}")

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQNetwork(input_size=29, output_size=4).to(device)
checkpoint_path = "models/snake_dqn_full_32.pth"
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Loaded model from {checkpoint_path}")
else:
    print(f"Model file {checkpoint_path} not found. Please train the model first.")
    exit(1)
model.eval()

# Play the game
game = SnakeGame()
state = game.reset()
total_reward = 0
steps = 0

print("\nStarting game with trained model...")
with torch.no_grad():
    while steps < MAX_STEPS:
        state_tensor = torch.FloatTensor(state).to(device)
        q_values = model(state_tensor)
        
        # Enforce Snake game rules
        opposite = {UP: DOWN, DOWN: UP, RIGHT: LEFT, LEFT: RIGHT}
        current_direction = game.direction
        head_y, head_x = game.snake[0]
        valid_actions = []
        for a in range(4):
            if a == opposite.get(current_direction):  # No reverse moves
                continue
            new_head = (head_y + DIRECTIONS_VECT[a][0], head_x + DIRECTIONS_VECT[a][1])
            if game.prev_position and new_head == game.prev_position:  # No backtracking
                continue
            valid_actions.append(a)
        
        if not valid_actions:
            print("No valid actions available. Game over.")
            break
        
        # Filter Q-values for valid actions only
        q_values_adjusted = q_values.clone()
        for a in range(4):
            if a not in valid_actions:
                q_values_adjusted[a] = float('-inf')
        action = torch.argmax(q_values_adjusted).item()
        
        next_state, reward, done = game.step(action)
        total_reward += reward
        state = next_state
        steps += 1
        
        game.render()
        time.sleep(0.01)
        
        if done:
            break

print(f"Game Over: Total Food Eaten={game.food_eaten}, Total Reward={total_reward:.2f}, Steps={steps}, Body Collisions={game.body_collisions}, Border Collisions={game.border_collisions}")
