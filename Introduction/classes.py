import gymnasium as gym
import numpy as np


class GridEnv(gym.Env):

    def __init__(self, square_grid_size):
        
        super(GridEnv, self).__init__()
        self.size = square_grid_size
    

    def reset(self):

        self.state = (0, 0)
        self.total_reward = 0

        return self.state


    def step(self, action):
        # actions: 0 - Up, 1 - Down, 2 - Left, 3 - Right
        # returns new state & reward & whether agent is done
        if action == 0 and self.state[0] < self.size - 1:
            self.state = (self.state[0] + 1, self.state[1])
        elif action == 1 and self.state[0] > 0:
            self.state = (self.state[0] - 1, self.state[1])
        elif action == 2 and self.state[1] > 0:
            self.state = (self.state[0], self.state[1] - 1)
        elif action == 3 and self.state[1] < self.size - 1:
            self.state = (self.state[0], self.state[1] + 1)
        
        if self.state == (self.size - 1, self.size - 1):
            reward = 100
        else:
            reward = -1

        done = self.state == (self.size - 1, self.size - 1)

        return self.state, reward, done


class Agent:

    def __init__(self, lr, gamma, n_actions, eps_start, eps_end, eps_dec):

        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.epsilon = eps_start
        self.eps_min = eps_end
        self.eps_dec = eps_dec

        self.Q = {}


    # greedy epsilon:
    def choose_actions(self, state):

        if np.random.rand() < self.epsilon:
            action = np.random.choice([i for i in range(self.n_actions)])
        else:
            actions = np.array([self.Q.get((state, a), 0.0) for a in range(self.n_actions)])
            action = np.argmax(actions)

        return action


    def learn(self, state, action, reward, next_state):

        actions_next = np.array([self.Q.get((next_state, a), 0.0) for a in range(self.n_actions)])
        a_max = np.argmax(actions_next)

        # learning rate equation
        Q = self.Q.get((state, action), 0.0)
        self.Q[(state, action)] = Q + self.lr * (reward + self.gamma * actions_next[a_max] - Q)

        self.decrement_epsilon()


    def decrement_epsilon(self):

        self.epsilon = max(self.epsilon * self.eps_dec, self.eps_min)


    def final_model(self, state):

        actions = np.array([self.Q.get((state, a), 0.0) for a in range(self.n_actions)])
        action = np.argmax(actions)

        if action == 0:
            out = "up"
        elif action == 1:
            out = "down"
        elif action == 2:
            out = "left"
        elif action == 3:
            out = "right"
        
        return out
