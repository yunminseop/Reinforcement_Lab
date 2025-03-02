import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import copy
from collections import deque

class Agent(nn.Module):
    """ Hyper Parameters initialized"""
    def __init__(self, state_size, action_size):
        super(Agent, self).__init__()
        self.alpha = 0.001
        self.gamma = 0.99
        self.epsilon = 0.9
        self.epsilon_decay = 0.9995
        self.batch_size = 64
        self.n_episode = 100
        self.D = deque(maxlen=2000)

        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def transition(self, state, action):
        curr_state = state[0].clone() 

        match action:
            case 0: curr_state[1] -= 1 
            case 1: curr_state[0] += 1 
            case 2: curr_state[1] += 1  
            case 3: curr_state[0] -= 1  

        if curr_state[0] < 0 or curr_state[0] > 3 or curr_state[1] < 0 or curr_state[1] > 3:
            return state.clone()

        return curr_state


    def reward(self, state):
        if np.array_equal(state, np.array([1, 1])) or np.array_equal(state, np.array([1, 2])) or np.array_equal(state, np.array([3, 1])):
            r = -10
        elif np.array_equal(state, np.array([3, 3])):
            r = 100
        elif state[0][0].item() < 0 or state[0][1].item() > 3:
            r = -10
        else:
            r = 0
        return r
    
    def done(self, state):
        if np.array_equal(state, np.array([3, 3])):
            return True
        else:
            return False
        
    def store_transition(self, state, action, reward, next_state, done):
        self.D.append((state, action, reward, next_state, done))

    def sample_batch(self):
        if len(self.D) < self.batch_size:
            return None
        return random.sample(self.D, self.batch_size)
    
    def train(self):
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        for episode in range(self.n_episode):
            state = np.asarray([0, 0]); #np.random.randint(state_size) 
            done = False
            total_reward = 0
            
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self(state_tensor)
                action = torch.argmax(q_values).item() if random.random() > self.epsilon else random.randint(0, action_size - 1)
                print(f"state: {state}")
                print(f"action: {action}")
                print(f"q_values: {q_values}")

                next_state = self.transition(state_tensor, action)
                reward = self.reward(next_state)
                done = self.done(state)

                self.store_transition(state, action, reward, next_state, done)

              
                batch = self.sample_batch()
                if batch:
                    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
                    
                    state_batch = torch.FloatTensor(state_batch)
                    if type(state_batch) == tuple:
                        state_batch = torch.stack(state_batch)
                    action_batch = torch.LongTensor(action_batch).unsqueeze(1)
                    reward_batch = torch.FloatTensor(reward_batch)
                    next_state_batch = torch.FloatTensor(next_state_batch)
                    done_batch = torch.FloatTensor(done_batch)

                    q_values = self(state_batch).gather(1, action_batch).squeeze()
                    next_q_values = self(next_state_batch).max(1)[0].detach()
                    target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

                    loss = loss_fn(q_values, target_q_values)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                state = next_state
                total_reward += reward

            self.epsilon *= self.epsilon_decay 
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}")



state_size = 2 
action_size = 4 
agent = Agent(state_size, action_size)

agent.train()
