import numpy as np
import torch


class Buffer:
    def __init__(self, size, state_dim, gamma=0.99):
        self.size = size

        self.states = np.zeros((size, state_dim))
        self.actions = np.zeros((size,))
        self.rewards = np.zeros((size,))
        self.old_log_probs = np.zeros((size,))
        self.old_values = np.zeros((size,))
        self.advantages = np.zeros((size,))
        self.disc_returns = np.zeros((size,))

        self.gamma = gamma
        self.pointer = 0
        self.traj_start = 0

    def log(self, s, a, r, logp, v):
        ''' saves experience '''
        assert self.pointer < self.size
        self.states[self.pointer] = s
        self.actions[self.pointer] = a
        self.rewards[self.pointer] = r
        self.old_log_probs[self.pointer] = logp
        self.old_values[self.pointer] = v
        self.pointer += 1

    def get_discounted_returns(self, rewards):
        returns_batch = []
        sum = 0
        for r in rewards[::-1]:
            sum = r + self.gamma*sum
            returns_batch.append(sum)
        return returns_batch[::-1]

    def get_advantages(self, disc_returns, old_values, normalise:bool):
        advantages = disc_returns - old_values
        if normalise:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def finish_trajectory(self, last_value):
        ''' when batch is full or episode is done'''
        traj_slice = slice(self.traj_start, self.pointer)
        rewards = np.append(self.rewards[traj_slice], last_value)
        disc_returns = self.get_discounted_returns(rewards)
        self.disc_returns[traj_slice] = disc_returns[:-1]
        self.traj_start = self.pointer

    def get_data(self, normalise_advantages:bool, normalise_observations:bool):
        ''' returns batch for gradient descent '''
        if normalise_advantages:
            self.advantages = self.get_advantages(self.disc_returns, self.old_values, normalise_advantages)
        if normalise_observations:
            self.states = (self.states - self.states.mean(axis=0)) / (self.states.std(axis=0) + 1e-8)

        data = dict(
            states=self.states
            , actions=self.actions
            , returns=self.disc_returns
            , advantages=self.advantages
            , old_logpi=self.old_log_probs
        )
        return {key:torch.from_numpy(value).float() for key, value in data.items()}

    def shuffle(self, data:dict):
        ''' shuffles batch to reduce bias due to temporal correlation '''
        perm = torch.randperm(self.size)
        data = {key:value[perm] for key, value in data.items()}
        data['returns'] = data['returns'].detach()
        data['advantages'] = data['advantages'].detach()
        data['old_logpi'] = data['old_logpi'].detach()
        return data
