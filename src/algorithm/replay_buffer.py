import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as gnn
import numpy as np
import random
from collections import deque
from torch_geometric.data import Data
import gym


class OffPolicyReplayBuffer:
    def __init__(self, capacity, priority=False):
        self.buffer = deque(maxlen=capacity)
        self.priority = priority
        self.priorities = deque(maxlen=capacity) if priority else None

    def push(self, state, action, reward, next_state, done, timestep, priority=1.0):
        self.buffer.append((state, action, reward, next_state, done, timestep))
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


class OnPolicyReplayBuffer:
    def __init__(self, capacity, ):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, log_prob, reward, next_state, done, timestep):
        self.buffer.append((state, action, log_prob, reward, next_state, done, timestep))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch

    def __call__(self, batch_size, shuffle=True):
        random.shuffle(self.buffer) if shuffle else None
        buffer = list(self.buffer)
        for i in range(0, len(self.buffer), batch_size):
            batch = buffer[i:i + batch_size]
            yield batch

    def normalize_rewards(self):
        rewards = [t[3] for t in self.buffer]
        mean = np.mean(rewards)
        std = np.std(rewards)
        for i in range(len(self.buffer)):
            self.buffer[i] = list(self.buffer[i])
            self.buffer[i][3] = (self.buffer[i][3] - mean) / std

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


class TrajectoryReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, trajectory):
        self.buffer.append(trajectory)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch

    def __len__(self):
        return len(self.buffer)

    def gen(self, batch_size, shuffle=True):
        random.shuffle(self.buffer) if shuffle else None
        for i in range(0, len(self.buffer), batch_size):
            yield self.buffer[i:i + batch_size]