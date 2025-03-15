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




###################################################################
###################################################################
###########  VALUE ITERATION ######################################
###################################################################
###################################################################



        
print("\n\nValue Iteration")
        
# value iteration            
def vi(V):
    def sum_sa(s, a):
        vs = 0
        for sp in range(S):
            vs = vs + P[s,a,sp] * (R[s,a,sp] + gamma * V[sp])
        return vs
        
    theta = 0.1
    delta = 1.0
    while delta > theta:
        delta = 0
        for s in range(S):
            v = V[s]
            # V(s) = max_a sum_s' P_ss'a * [R_ss'a + gamma * V(s')]
            # max_a
            ma = 0
            for a in range(A):
                # sum_s'
                vs = sum_sa(s, a)
                ma = max(ma, vs)
            V[s] = ma
            delta = max(delta, abs(v - V[s]))

    # recover policy as best actions
    for s in range(S):
        ma = 0
        for a in range(A):
            vs = sum_sa(s, a)
            if vs > ma:
                ma = vs
                pi[s] = a
                print("Best s[", s, "]=", a)


    print("V:")
    print(V)
    print()
    print("A0=up, A1=right, A2=down, A3=left")
    print("Pi:")
    print_A(pi)

vi(V)
