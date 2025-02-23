import numpy as np
import random

""" State transition probability distribution = 1.0 (deterministic env)
    Probability of moving left or right from state 1 and state 2. = 0.5 (random policy, Monte Carlo method)"""

S = [0, 1, 2, 3] # state. 0, 1, 2, 3

A = [-1, 1] # action. left: -1, right: 1

R = [0, 1] # reward. arrival at state 3 reward: 1, The others reward: 0

class agent:
    def __init__(self):
        self.state = S[0]
        self.gamma = 0.9
        self.value_function = {s: 0 for s in S} # optimal value dict
        self.optimal_action = {s: 0 for s in S} # optimal policy dict
        self.action = None
        self.reward = None
        self.next_state = None
        self.value = None
    
    # state transition function
    def transition(self):
        p = random.random()

        if self.next_state is not None:
            self.state = self.next_state

        if self.state == S[0]: # if state = 0,
            self.action = A[1]; self.reward = R[0] # only move right and no reward

        elif self.state == S[1] or self.state == S[2]: # if state = 1 or 2,
            if p >= 0.5:    # move left or right with a probability of 0.5
                self.action = A[0]
            else: self.action = A[1]
            
            self.reward = R[0] # and no reward


        elif self.state == S[3]: # if state = 3,
            self.action = 0 # not move
            self.reward = R[1] # reward = 1

        self.next_state = self.state + self.action

        return(self.state, self.action, self.reward, self.next_state)

    # value iteration algorithm
    def value_iteration(self, threshold=1e-6):
        delta = float('inf')
        while delta > threshold:
            delta = 0
            for s in S: # For iteration per each state
                v = self.value_function[s]  # record value function of curr state
                max_value = float('-inf')
                
                for a in A: # For iteration per each action
                    next_state = s + a
                    if next_state not in S:  # ignore unvalid states
                        continue
                    
                    # Bellman Optimality Equation
                    expected_value = R[0] + self.gamma * self.value_function.get(next_state, 0) if next_state != 3 else R[1]
                    
                    # optimal action(Policy)
                    if expected_value > max_value:
                        if s == 3:
                            self.optimal_action[s] = 'stop'
                        else:
                            if a == -1:
                                self.optimal_action[s] = 'move left'                              
                            else:
                                self.optimal_action[s] = 'move right'

                    # find optimal value
                    max_value = max(max_value, expected_value)
                    """ The optimal value function returns the expected reward that can be obtained from future states when the optimal action is taken.
                    However, we don't know what the optimal action is yet. This is because, as we iterate through the actions (such as -1 and 1),
                    we calculate the expected rewards for each action and store the maximum reward in max_value.
                    But, just by doing this, we can't determine which action is optimal.

                    If we want to know the optimal action for each state,
                    we can simply compare the values calculated for each action during the iteration.
                    Each time a higher value is found, we can store the action that leads to this higher value as the optimal action."""
                    

                
                # update value funcs
                if s == 3:
                    self.value_function[s] = 1
                else:
                    self.value_function[s] = max_value
                delta = max(delta, abs(v - self.value_function[s]))

        return self.value_function, self.optimal_action



my_agent = agent()

# state transition
while (my_agent.state != 3):
    res = my_agent.transition()
    print(f"s: {res[0]}, a: {res[1]}, r: {res[2]}, s': {res[3]}")

# value iteration
value_function = my_agent.value_iteration()
print(f"optimal value function: {value_function[0]}")
print(f"optimal policy: {value_function[1]}")
