import random

S = [0, 1, 2, 3]  # state.
A = [-1, 1]  # action.
R = [0, 1]  # reward 

class Agent:
    def __init__(self):
        self.Q = {(s, a): 0 for s in S for a in A if s != 3}
        self.Q[(3, 0)] = 1.0  
        self.epsilon = 0.4  # greedy algorithm
        self.gamma = 0.9  # discount factor
        self.alpha = 0.5  # learning rate

    def epsilon_greedy(self, state):
        if state == 3:
            return 0
        
        elif random.random() < self.epsilon:
            if state == 0:
                return 1
            return random.choice(A)  # randomly choose the action

        else:
            q_values = [self.Q.get((state, a), 0) for a in A]
            max_q = max(q_values)
            return A[q_values.index(max_q)]  # choose the action that has the highest value(Q)

    def Q_learning(self, threshold=0.01):
        delta = float('inf')
        while delta > threshold:
            delta = 0
            curr_state = 0

            while curr_state != 3:
                action = self.epsilon_greedy(curr_state)

                next_state = curr_state + action
                next_state = max(0, min(next_state, 3))  # limit state from 0 to 3

                # get reward
                reward = R[1] if next_state == 3 else R[0]

                # update Q
                old_q = self.Q[(curr_state, action)]

                self.Q[(curr_state, action)] += self.alpha * (
                    reward + self.gamma * max(self.Q.get((next_state, a), 0) for a in A) - old_q
                )

                delta = max(delta, abs(old_q - self.Q[(curr_state, action)]))

                # update only state (different from SARSA)
                curr_state = next_state

        self.Q.pop((0, -1))
        print(f"Q: {self.Q}")

my_agent = Agent()
my_agent.Q_learning()