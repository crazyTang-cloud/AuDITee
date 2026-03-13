import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, config):
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.memory = deque(maxlen=config["memory_size"])
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config["lr"])
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon"]
        self.batch_size = config["batch_size"]
        self.update_target_steps = config["update_target_steps"]
        self.learn_step = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        with torch.no_grad():
            state = np.array(state, dtype=np.float32)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            return torch.argmax(self.q_net(state_tensor)).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def warm_start(self, labeled_dataset, n_iters=1000):
        print("[RL] Warm-starting DQN with labeled data...")
        for state, label in labeled_dataset:
            if random.random() < self.epsilon:
                action = random.choice([0, 1])
            else:
                action = label
            # action = 1 if label == 1 else 0
            # reward = 1.0 if label == 1 else -0.2
            
            # 根据行为结果设计 reward
            if action == 1 and label == 1:
                reward = 1.0       # 测中缺陷：正反馈
            elif action == 1 and label == 0:
                reward = -0.3      # 测试无缺陷：轻惩罚
            elif action == 0 and label == 1:
                reward = -1.0      # 漏测缺陷：严重惩罚
            else:  # action == 0 and label == 0
                reward = 0.2       # 节省资源：轻奖励
            
            next_state = state
            done = True
            self.store_transition(state, action, reward, next_state, done)

        for _ in range(n_iters):
            self.learn()
        print("[RL] Warm-start complete.")
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1)[0].unsqueeze(1)
            q_target = rewards + self.gamma * q_next * (1 - dones)

        loss = nn.functional.mse_loss(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.update_target_steps == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
