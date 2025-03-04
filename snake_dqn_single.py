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
FOOD_AVG_GOAL = 50
WINDOW_SIZE = 35
NEURON_FOOD_GOAL = 25
BEST_MOVES_LIMIT = 10

# DQN hyperparameters
GAMMA = 0.9
EPSILON = 0.5
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.99
MEMORY_SIZE = 10000
BATCH_SIZE = 32
LR = 0.001
MAX_EPISODES = 10000

class SnakeGame:
    def __init__(self):
        self.max_dist = np.sqrt(TABLE_WIDTH ** 2 + TABLE_HEIGHT ** 2)
        self.move_history = deque(maxlen=2)
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
        self.move_history.clear()
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
        ], dtype=np.float32)
    
    def step(self, action):
        head_y, head_x = self.snake[0]
        reward = 0
        done = False
        
        dy, dx = DIRECTIONS_VECT[action]
        new_head = (head_y + dy, head_x + dx)
        
        move_info = {'action': action, 'direction': self.direction, 'new_head': new_head}
        
        tail_y, tail_x = self.snake[-1] if len(self.snake) > 1 else (head_y, head_x)
        dist_to_tail = np.sqrt((new_head[0] - tail_y) ** 2 + (new_head[1] - tail_x) ** 2)
        if dist_to_tail < 2 or new_head[0] < 2 or new_head[0] > TABLE_HEIGHT - 3 or new_head[1] < 2 or new_head[1] > TABLE_WIDTH - 3:
            reward -= 0.1
        
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
        reward += 0.001
        if new_head == self.food:
            reward += 10
            self.food_eaten += 1
            self.food = self._spawn_food()
            self.steps_since_food = 0
            if self.food_eaten in [10, 20, 30, 40]:
                reward += [50, 100, 200, 300][[10, 20, 30, 40].index(self.food_eaten)]
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pretrain individual neurons with best moves
def train_neuron(neuron_idx):
    weights = torch.randn(4, 29, requires_grad=True, device=device)  # Shape: [4, 29]
    bias = torch.randn(4, requires_grad=True, device=device)        # Shape: [4]
    optimizer = optim.Adam([weights, bias], lr=LR)
    memory = deque(maxlen=MEMORY_SIZE)  # Replay buffer for online learning
    food_history = deque(maxlen=WINDOW_SIZE)
    epsilon = EPSILON
    
    # TensorBoard writer for this neuron
    neuron_writer = SummaryWriter(log_dir=f"runs/neuron_{neuron_idx}")
    
    print(f"Training neuron {neuron_idx}")
    for episode in range(MAX_EPISODES):
        game = SnakeGame()
        state = game.reset()
        episode_reward = 0
        steps = 0
        
        while steps < MAX_STEPS:
            state_tensor = torch.FloatTensor(state).to(device)  # Shape: [29]
            with torch.no_grad():
                q_values = torch.matmul(weights, state_tensor) + bias  # Shape: [4]
            
            opposite = {UP: DOWN, DOWN: UP, RIGHT: LEFT, LEFT: RIGHT}
            current_direction = game.direction
            head_y, head_x = game.snake[0]
            valid_actions = []
            for a in range(4):
                if a == opposite.get(current_direction):
                    continue
                new_head = (head_y + DIRECTIONS_VECT[a][0], head_x + DIRECTIONS_VECT[a][1])
                if game.prev_position and new_head == game.prev_position:
                    continue
                valid_actions.append(a)
            
            if not valid_actions:
                break
            
            if random.random() < epsilon:
                action = random.choice(valid_actions)
            else:
                q_values_adjusted = q_values.clone()
                for a in range(4):
                    if a not in valid_actions:
                        q_values_adjusted[a] = float('-inf')
                action = torch.argmax(q_values_adjusted).item()
            
            next_state, reward, done = game.step(action)
            episode_reward += reward
            steps += 1
            
            memory.append((state, action, reward, next_state, done))
            state = next_state
            
            # Online learning
            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.FloatTensor(np.array(states)).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(np.array(next_states)).to(device)
                dones = torch.FloatTensor(dones).to(device)
                
                q_values = torch.matmul(states, weights.T) + bias  # Shape: [batch, 4]
                q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # Shape: [batch]
                with torch.no_grad():
                    next_q_values = torch.matmul(next_states, weights.T) + bias  # Shape: [batch, 4]
                    next_q_values = next_q_values.max(1)[0]  # Shape: [batch]
                targets = rewards + (1 - dones) * GAMMA * next_q_values
                
                loss = nn.MSELoss()(q_values, targets.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if done:
                break
        
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        food_history.append(game.food_eaten)
        avg_food = np.mean(food_history) if len(food_history) == WINDOW_SIZE else 0
        
        # Log to TensorBoard
        neuron_writer.add_scalar("Avg Food Eaten (Last 35)", avg_food, episode)
        
        print(f"Neuron {neuron_idx}, Episode {episode}: Food Eaten={game.food_eaten}, Avg Food (last {len(food_history)})= {avg_food:.2f}, Steps={steps}")
        
        if len(food_history) == WINDOW_SIZE and avg_food >= NEURON_FOOD_GOAL:
            print(f"Neuron {neuron_idx} reached avg food goal of {NEURON_FOOD_GOAL}!")
            break
    
    # Final refinement with best moves
    if len(memory) > BEST_MOVES_LIMIT:
        sorted_transitions = sorted(memory, key=lambda x: x[2], reverse=True)
        best_transitions = sorted_transitions[:BEST_MOVES_LIMIT]
        
        for _ in range(50):
            states, actions, rewards, next_states, dones = zip(*best_transitions)
            states = torch.FloatTensor(np.array(states)).to(device)
            actions = torch.LongTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.FloatTensor(np.array(next_states)).to(device)
            dones = torch.FloatTensor(dones).to(device)
            
            q_values = torch.matmul(states, weights.T) + bias
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q_values = torch.matmul(next_states, weights.T) + bias
                next_q_values = next_q_values.max(1)[0]
            targets = rewards + (1 - dones) * GAMMA * next_q_values
            
            loss = nn.MSELoss()(q_values, targets.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Neuron {neuron_idx} refined with {len(best_transitions)} best moves")
    
    # Close the neuron-specific writer
    neuron_writer.close()
    
    # Save neuron weights (keeping action-specific weights)
    save_path = f"neuron_models/neuron_{neuron_idx}.pth"
    if not os.path.exists("neuron_models"):
        os.makedirs("neuron_models")
    torch.save({
        'weights': weights,
        'bias': bias
    }, save_path)
    print(f"Saved neuron {neuron_idx} weights to {save_path}")
    
    # Average weights for merging into the final model
    avg_weights = weights.mean(dim=0)  # Shape: [29]
    avg_bias = bias.mean()             # Shape: [1]
    return torch.cat((avg_weights, avg_bias.unsqueeze(0)))  # Shape: [30]

# Create neuron_models directory
if not os.path.exists("neuron_models"):
    os.makedirs("neuron_models")

# Train 64 neurons individually
neuron_weights_list = []
for i in range(64):
    neuron_weights = train_neuron(i)
    neuron_weights_list.append(neuron_weights)

# Merge neuron weights into a single tensor of shape [64, 30]
merged_weights = torch.stack(neuron_weights_list, dim=0)  # Shape: [64, 30]

# Save merged weights to a single .pth file
save_path = "models/merged_neurons.pth"
if not os.path.exists("models"):
    os.makedirs("models")
torch.save(merged_weights, save_path)
print(f"Saved merged neuron weights to {save_path}")
