import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_mask):
        self.buffer.append((state, action, reward, next_state, done, next_mask))

    def sample(self, batch_size):
        state, action, reward, next_state, done, next_mask = zip(*random.sample(self.buffer, batch_size))
        return (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.int64),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.float32),
            np.array(next_mask, dtype=bool)
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim=54, lr=1e-4, gamma=0.99, batch_size=64, buffer_capacity=100000, tau=0.005):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_capacity)
        
        self.loss_fn = nn.SmoothL1Loss()  # Huber Loss

    def select_action(self, state, action_mask, epsilon=0.0):
        # action_mask is a boolean array of size 54, True for valid actions
        valid_indices = np.where(action_mask)[0]
        
        if len(valid_indices) == 0:
            # Should not happen as we always have at least one card unless done
            return 0
            
        if random.random() < epsilon:
            # Random exploration over valid actions
            return random.choice(valid_indices)
        
        # Greedy exploitation
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_t).squeeze(0).cpu().numpy()
        self.policy_net.train()
        
        # Apply action masking by setting Q-values of invalid actions to a very low value
        masked_q_values = np.full_like(q_values, -1e9)
        masked_q_values[action_mask] = q_values[action_mask]
        
        return int(np.argmax(masked_q_values))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None
            
        # Sample batch
        states, actions, rewards, next_states, dones, next_masks = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        next_masks_t = torch.BoolTensor(next_masks).to(self.device)
        
        # Current Q-values for chosen actions
        current_q = self.policy_net(states_t).gather(1, actions_t).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            next_q = self.target_net(next_states_t)
            # Apply action masking to next states
            masked_next_q = torch.full_like(next_q, -1e9)
            masked_next_q[next_masks_t] = next_q[next_masks_t]
            max_next_q = masked_next_q.max(dim=1)[0]
            # If done, next state Q value is 0 (or we override using dones_t)
            # For done states, max_next_q should not contribute
            # Since masked_next_q might be filled with -1e9 if mask is all False, max() could be -1e9,
            # so multiplying by (1 - dones_t) is correct.
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * max_next_q
            # In case next_masks_t is all False for a done state, max_next_q could be -1e9,
            # so we explicitly set target_q to reward if done.
            target_q = torch.where(dones_t > 0.5, rewards_t, target_q)
            
        # Compute loss
        loss = self.loss_fn(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Soft update target network: target = tau * policy + (1 - tau) * target
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
            
        return loss.item()

    def save(self, filepath):
        torch.save(self.policy_net.state_dict(), filepath)

    def load(self, filepath):
        self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
