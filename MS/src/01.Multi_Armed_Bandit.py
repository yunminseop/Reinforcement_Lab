import numpy as np

arms_profit = [0.4, 0.12, 0.52, 0.6, 0.25]
n_arms = len(arms_profit)

n_trial = 1000

def pull_bandit(handle):
    q = np.random.random()
    if q <arms_profit[handle]:
        return 1
    else:
        return -1
    
""" 1. random policy """

# def random_exploration():
#     episode = []
#     num = np.zeros(n_arms)
#     wins = np.zeros(n_arms)

#     for i in range(n_trial):
#         h = np.random.randint(0, n_arms) # handle num 0 ~ 4
#         reward = pull_bandit(h) # result by operating handle

#         episode.append([h, reward]) # record handle num and what reward agent got
#         num[h] += 1 # increase operating cnt of the handle
#         wins[h] += 1 if reward == 1 else 0 # win cnt increased if reward == 1

#     return episode, (num, wins)

# e, r = random_exploration() # e = list of total episodes, r = operating cnt and win cnt

# print("Chance of winning(per handle):", ["%6.4f"% (r[1][i]/r[0][i]) if r[0][i] > 0 else 0.0 for i in range(n_arms)])
# print("profit per handle:($):", ["%d"% (2*r[1][i]-r[0][i]) for i in range(n_arms)])
# print("net profit($):", sum(np.asarray(e)[:,1]))


""" 2. epsilon_greedy algorithm
 - Using random number to approach mathemetical problems, 'Monte Carlo method' 
 - it's a basically greedy algorithm but it applies exploration at a specific rate(epsilon) to pursue a balance between exploration and exploitation."""

def epsilon_greedy(eps):
    episode = []
    num = np.zeros(n_arms)
    wins = np.zeros(n_arms)

    for i in range(n_trial):
        r = np.random.random()
        if (r<eps or sum(wins)==0):
            h = np.random.randint(0, n_arms)
        else:
            prob = np.asarray([wins[i]/num[i] if num[i]>0 else 0.0 for i in range(n_arms)])
            prob = prob/sum(prob)
            h = np.random.choice(range(n_arms), p=prob)
        reward = pull_bandit(h)
        episode.append([h, reward])
        num[h]+=1
        wins[h]+=1 if reward==1 else 0
    return episode, (num, wins)

e, r = epsilon_greedy(0.1)

print("Chance of winning(per handle:", ["%6.4f"% (r[1][i]/r[0][i]) if r[0][i] > 0 else 0.0 for i in range(n_arms)])
print("profit per handle:($):", ["%d"% (2*r[1][i]-r[0][i]) for i in range(n_arms)])
print("net profit($):", sum(np.asarray(e)[:,1]))