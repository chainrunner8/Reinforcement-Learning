# Aske Plaat Leiden University
# 2 sep 2024
# value iteration
import numpy as np
import random
from statistics import mean
import sys

print("Example code for Value Iteration, Monte Carlo, and SARSA")

montecarlo_episodes = 1000
sarsa_episodes = 100

A = 4
# +--+--+--+--+--+
# |  | 0|  |  |  |
# +--+--+--+--+--+
# | 3| X| 1|  |  |
# +--+--+--+--+--+
# |  | 2|  |  |  |
# +--+--+--+--+--+
# |  |  |  |  |  |
# +--+--+--+--+--+
# |  |  |  |  |  |
# +--+--+--+--+--+

Sd = 5
S = Sd * Sd
# S:
# +--+--+--+--+--+
# | 0| 1| 2| 3| 4|
# +--+--+--+--+--+
# | 5| 6| 7| 8| 9|
# +--+--+--+--+--+
# |10|11|12|13|14|
# +--+--+--+--+--+
# |15|16|17|18|19|
# +--+--+--+--+--+
# |20|21|22|23|24|
# +--+--+--+--+--+

# R:
# +--+--+--+--+--+
# | 0| 0| 0| 0| 0|
# +--+--+--+--+--+
# | 0| 0| 0| 0| 0|
# +--+--+--+--+--+
# | 0| 0| 1| 0| 0|
# +--+--+--+--+--+
# | 0| 0| 0| 0| 0|
# +--+--+--+--+--+
# | 0| 0| 0| 0| 1|
# +--+--+--+--+--+



gamma = 0.99
V = np.zeros(S)
R = np.zeros((S,A,S))
P = np.zeros((S,A,S))

# initialize V arbitrary
for s in range(S):
    V[s] = 0

# initialize R
for s in range(S):
    for a in range(A):
        for sp in range(S):
            R[s, a, sp] = 0
R[19, 2, 24] = 1
R[23, 1, 24] = 1
R[11,1,12] = 1
R[7,2, 12] = 1
R[17,0,12] = 1
R[13,3,12] = 1


# value iteration needs P transition probabilities

# initialize P
# all actions have equal probability of one quarter to go to the other state via an action
for s in range(S):
    for a in range(A):
        for sp in range(S):
            P[s, a, sp] = 0.0
            
for s in range(S):
    if s-Sd >= 0:
        P[s, 0, s-Sd] = 1.0 # in this environment, after action a we transition deterministically to that particular state sp
    if (s+1) % Sd != 0:
        P[s, 1, s+1] = 1.0
    if s < S - Sd:
        P[s, 2, s+Sd] = 1.0
    if s % Sd != 0:
        P[s, 3, s-1] = 1.0

pi = np.zeros(S, dtype=int)

def print_A(a):
    for s in range(Sd):
        for i in range(Sd):
            print("%1.1f" % (a[s*Sd+i]), end=' ')
        print()

maxlength = S

Q = np.zeros((S,A))
# initialize Q arbitrary
for s in range(S):
    for a in range(A):
        Q[s,a] = 0

err_val = -1

# transition function: follow the action deterministically to the next state
# this is done by the environment, not the agent
def transition(s, a):
    if a == 0 and s > Sd - 1:
        return s - Sd
    if a == 1 and (s+1) % Sd != 0:
        return s+1
    if a == 2 and s < S - Sd:
        return s + Sd
    if a == 3 and s % Sd != 0:
        return s-1
    return err_val


# initialize pi arbitrary
for s in range(S):
    a = random.randint(0,3)
    while transition(s, a) == err_val:
        a = random.randint(0,3)
    pi[s] = a

#print("Pi: ", pi)
        
# initialize Returns as empty list for all s, a; so a 2D array of lists, or a 2D list of lists
Returns = []
for s in range(S):
    Returns.append([])
    for a in range(A):
        Returns[s].append([])

def terminal(s):
    return s == S - 1

# check for a safe action that does nut push us off the grid
def epsilon_greedy(s, a):
    epsilon = 0.1
    def epsilon_gr(aa):
        if random.random() < epsilon:
            x = random.randint(0,3)
        else:
            x = aa
            #    print("X: ", x)
        return x
    x = epsilon_gr(a)
    while transition(s, x) == err_val:
        x = epsilon_gr(a)
    return x



# sarsa need initialized policy
# we choose a deterministic policy with a random action
for s in range(S):
    pi[s] = random.randint(0,3)

Q = np.zeros((S,A))
# initialize Q arbitrary
for s in range(S):
    for a in range(A):
        Q[s,a] = 0

    
alpha = 0.1


# print("Q: ", Q)
# print("Pi")                
# print_A(pi)


print("\n\nSARSA")
        

###################################################################
###################################################################
############# SARSA ###############################################
###################################################################
###################################################################

# temporal difference on-policy sarsa
def sarsa(Q):
    # for eacch episode
    for i in range(sarsa_episodes): # for each episode
        s = 0
        a = epsilon_greedy(s, pi[s])

        # for each timestep
        while s != None and not terminal(s): # for each time step of the episode
            sp = transition(s, a)
            r = R[s,a,sp]
            ap = epsilon_greedy(s, pi[sp])
            Q[s,a] = Q[s,a] + alpha * (r + gamma * Q[sp,ap] - Q[s,a])
            s = sp
            a = ap

    # recover policy as best actions from state/action values
    for s in range(S):
        ma = 0
        for a in range(A):
            if Q[s,a] > ma:
                ma = Q[s,a]
                pi[s] = a
            

sarsa(Q)

print("Q: ", Q)
print("Pi")
print_A(pi)

