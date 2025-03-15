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

# mc needs pi Policy
# we choose a deterministic policy with a random action
pi = np.zeros(S, dtype=int)
for s in range(S):
    pi[s] = random.randint(0,3)

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
#    print("ERROR!! s: ", s, " a: ", a)
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

# The triple is the start of an episode that ends in a terminal.
# Compute the Return = the sum of the rewards
def compute_return(s):
    if s == err_val:
        print("ERROR in COMPUTE RETURN")
    ret = 0
    i = 0
    while s != None and not terminal(s) and i < maxlength:
        i = i + 1
        a = pi[s]
        sp = transition(s, a)
        r = R[s,a,sp]
        ret = ret + r
        s = sp
    return ret

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



print("\n\nMonte Carlo")


###################################################################
###################################################################
############# MONTE CARLO #########################################
###################################################################
###################################################################


    
# monte carlo control      
def mc(Q):
    ep = []    
    for i in range(montecarlo_episodes):
        
        # 1. generate one episode using policy pi
        start_s = s = random.randint(0,S-1) # pick a random start point
        ep = [] # empty list
        while s != None and not terminal(s) and len(ep) < maxlength:
            a = epsilon_greedy(s, pi[s])
            sp = transition(s, a)
            if sp == err_val:
                print("ERROR in MC")
            r = R[s,a,sp]
            ep.append((s,a,r)) # store triple in episode list
            s = sp # go to next state

        # 2. evaluate the episode
        for (s, a, r) in ep:
            cr = compute_return(transition(s,a)) # start at the action a, hence the extra transition
            Returns[s][a].append(cr)
            rl = Returns[s][a]
            su = sum(rl)
            le = len(rl)
            Q[s,a] = su/le

        # 3. improve the policy
        for (s, a, r) in ep:
            ma = np.argmax(Q[s])
            if Q[s,ma] > Q[s,pi[s]]: 
                pi[s] = ma


print("BEFORE MC")
print_A(pi)

mc(Q)

print("AFTER MC")
print("Q: ", Q)
print()
print_A(pi)


