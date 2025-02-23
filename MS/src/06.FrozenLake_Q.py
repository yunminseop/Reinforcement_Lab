"""FrozenLake -> Q_Learning"""
import numpy as np
import random
import matplotlib.pyplot as plt


S = list(range(0, 16))  # state 0~15
A = [0, 1, 2, 3]  # actions

class Agent:
    def __init__(self):
        self.Q = {(s, a): 0 for s in S for a in A if s != 16}
        self.hole_state = [5, 7, 11, 12]
        self.n_episode = 1000000
        self.Q[(15, -1)] = 1.0  # when state is 15, its action is -1(exception) and 1.0 Q
        self.epsilon = 0.4 
        self.gamma = 0.99  # discount factor
        self.alpha = 0.0  # learning rate

        self.total_state = {each:[] for each in S}
        self.optimal_policy = {each: 0 for each in S}


    def deterministic_transition(self, state, action):
        row, col = divmod(state, 4)  # Convert the current state to (row, col)

        if action == 0:  # move left
            col = max(0, col - 1)
        elif action == 1:  # move down
            row = min(3, row + 1)
        elif action == 2:  # move right
            col = min(3, col + 1)
        elif action == 3:  # move up
            row = max(0, row - 1)

        return row * 4 + col  # convert the (row, col) into state num
    

    def stochastic_transition(self, state, action):
        row, col = divmod(state, 4)

        if action == 0:  # move left
            next_states = [row * 4 + max(0, col - 1), row * 4 + min(3, col + 1)]  # left, right
            weights = [0.8, 0.2]  # left 80%, right 20%

        elif action == 1:  # move down
            next_states = [min(3, row + 1) * 4 + col, max(3, row - 1) * 4 + col]  # down, up
            weights = [0.8, 0.2]  # down 80%, up 20%

        elif action == 2:  # move right
            next_states = [row * 4 + min(3, col + 1), row * 4 + max(3, col + 1)]  # right, left
            weights = [0.8, 0.2]  # right 80%, left 20%

        elif action == 3:  # move up
            next_states = [max(0, row - 1) * 4 + col, max(0, row + 1) * 4 + col]  # up, down
            weights = [0.8, 0.2]  # up 80%, down 80%

        next_state = random.choices(next_states, weights=weights, k=1)[0]
        return next_state



    def epsilon_greedy(self, state):
        if state == 15:
            return -1
        
        elif random.random() < self.epsilon:
            return random.choice(A)  # randomly choose the action

        else:
            q_values = [self.Q.get((state, a), 0) for a in A]
            max_q = max(q_values)
            return A[q_values.index(max_q)]  # choose the action that has the highest value(Q)
        

    def get_reward(self, state):
        if state == 15:  # goal state
            return 1
        elif state in [5, 7, 11, 12]:  # Holes number
            return -1
        return 0  # The others have no reward

    def is_done(self, state):
        return state == 15 or state in [5, 7, 11, 12]  # goal state or holes


    def Q_learning(self):
        cnt = 0
        for _ in range(self.n_episode):
            curr_state = 0
            cnt += 1
            self.alpha = max(0.00015, 1 / (0.001 * cnt + 1))
            self.epsilon = max(0.1, 1 - (cnt / self.n_episode))
            print(self.alpha, self.epsilon)

            while curr_state != 15:

                action = self.epsilon_greedy(curr_state)

                next_state = self.stochastic_transition(curr_state, action)

                next_state = max(0, min(next_state, 15))  # limit state from 0 to 15

                # get reward
                reward = self.get_reward(next_state)

                # update Q
                old_q = self.Q[(curr_state, action)]

                self.Q[(curr_state, action)] += self.alpha * (
                    reward + self.gamma * max(self.Q.get((next_state, a), 0) for a in A) - old_q
                )

                # update only state
                curr_state = next_state

                if self.is_done(curr_state): # if the next state is a goal or a hole, exit the while loop.
                    break
        
        del self.Q[(15,0)]
        del self.Q[(15,1)]
        del self.Q[(15,2)]
        del self.Q[(15,3)]


    def show_optimal_policy(self):
        
        for state in S:
            for key in self.Q.keys():
                if state == key[0]:
                    self.total_state[state].append(self.Q[key])

        for item in self.total_state.items():
            match np.argmax(item[1]):
                case 0: self.optimal_policy[item[0]] = "←"
                case 1: self.optimal_policy[item[0]] = "↓"
                case 2: self.optimal_policy[item[0]] = "→"
                case 3: self.optimal_policy[item[0]] = "↑"

        print(self.optimal_policy)

        grid_size = 4
        grid = np.zeros((grid_size, grid_size))

        fig, ax = plt.subplots(figsize=(5, 5))
        
        highlight_cells = []

        for hole in self.hole_state:
            highlight_cells.append(divmod(hole, 4))

        ax.set_xticks(np.arange(-0.5, grid_size, 1))
        ax.set_yticks(np.arange(-0.5, grid_size, 1))
        ax.imshow(grid, cmap='Blues', alpha=0.3)

       
        for i in range(grid_size):
            for j in range(grid_size):
                state = i * grid_size + j

                # Hole
                if (i, j) in highlight_cells:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='lightcoral', alpha=0.5))

                # Policy
                if state in self.optimal_policy:
                    if state in self.hole_state:
                        ax.text(j, i, "Hole", ha='center', va='center', fontsize=14, fontweight='bold')
                    elif state == 15:
                        ax.text(j, i, "Goal", ha='center', va='center', fontsize=14, fontweight='bold')
                    else:
                        ax.text(j, i, self.optimal_policy[state], ha='center', va='center', fontsize=16, fontweight='bold')

        ax.set_xticks([])
        ax.set_yticks([])

        plt.show()

                    



my_agent = Agent()
my_agent.Q_learning()
my_agent.show_optimal_policy()