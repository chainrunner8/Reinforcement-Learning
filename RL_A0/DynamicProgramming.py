#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self,s):
        ''' Returns the greedy best action in state s ''' 
        actions = self.Q_sa[s]
        amax = np.argmax(actions)
        return amax
        
    def update(self, s, a, p_sas, r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        Q = self.Q_sa[s, a]
        self.Q_sa[s, a] = np.dot(
            p_sas[s, a,]
            , (r_sas[s, a,] + self.gamma * np.max(self.Q_sa, axis=1))
        )
        error = abs(self.Q_sa[s, a] - Q)
        return error



def Q_value_iteration(env, gamma=1.0, eta=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)
 
    # TO DO: IMPLEMENT Q-VALUE ITERATION HERE
    delta = 1
    i = 0
    while delta > eta:
        max_error = 0
        for s in range(QIagent.n_states):
            for a in range(QIagent.n_actions):
                error = QIagent.update(s, a, env.p_sas, env.r_sas)
                max_error = max(max_error, error)
        delta = max_error
        i += 1
        print("Q-value iteration, iteration {}, max error {}".format(i,delta))
        # if i%5 == 0:
        #     env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.2)

    return QIagent

def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent = Q_value_iteration(env,gamma,threshold)
    
    # view optimal policy
    done = False
    s = env.reset()
    rewards = []
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.5)
        s = s_next
        rewards.append(r)
    mean_reward_per_timestep = np.mean(rewards)
    # TO DO: Compute mean reward per timestep under the optimal policy
    print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))
    
if __name__ == '__main__':
    experiment()
