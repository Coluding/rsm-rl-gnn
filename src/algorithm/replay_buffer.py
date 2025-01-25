import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as gnn
import numpy as np
import random
from collections import deque
from torch_geometric.data import Data
import gym


class ReplayBuffer:
    def __init__(self, capacity, priority=False):
        self.buffer = deque(maxlen=capacity)
        self.priority = priority
        self.priorities = deque(maxlen=capacity) if priority else None

    def push(self, state, action, reward, next_state, done, priority=1.0):
        self.buffer.append((state, action, reward, next_state, done))
        if self.priority:
            self.priorities.append(priority)

    def sample(self, batch_size):
        if self.priority:
            probabilities = np.array(self.priorities) / sum(self.priorities)
            indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
            batch = [self.buffer[idx] for idx in indices]
        else:
            batch = random.sample(self.buffer, batch_size)
        return batch

    def update_priorities(self, indices, new_priorities):
        if self.priority:
            for idx, priority in zip(indices, new_priorities):
                self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)

