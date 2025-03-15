import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from torch import nn, optim
from stable_baselines3.common.vec_env import SubprocVecEnv


NUM_ENVS = 8  # my cpu is Ryzen 7 5800h with 8 cores



class NNet(nn.Module):
    
    def __init__(self, state_dim, actions_dim, nb_hidden, hidden_dim):
        super(NNet, self).__init__()
        if nb_hidden == 1:
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim)
                , nn.ReLU()
                , nn.Linear(hidden_dim, actions_dim)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim)
                , nn.ReLU()
                , nn.Linear(hidden_dim, hidden_dim)
                , nn.ReLU()
                , nn.Linear(hidden_dim, actions_dim)
            )

    def forward(self, x):
        return self.net(x)


def make_env():
    return gym.make('CartPole-v1')


def evalRollout(env_eval, QNet):

    state, _ = env_eval.reset()
    over = False
    return_ = 0

    while not over:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = QNet(state_tensor)
        action = torch.argmax(q_values).item()
        next_state, reward, terminated, truncated, _ = env_eval.step(action)
        over = terminated or truncated
        return_ += reward
        state = next_state
    return return_


# def runModel(lr, eps_decay, nb_hidden, hidden_dim):

if __name__ == "__main__":

    start = time.time()

    vec_env_main = SubprocVecEnv([make_env for _ in range(NUM_ENVS)])
    env_eval = gym.make('CartPole-v1')
    actions_dim = vec_env_main.action_space.n
    state_dim = vec_env_main.observation_space.shape[0]

    epsilon = 1
    eps_min = 0.01
    gamma = 1
    n_steps = 10**6 // NUM_ENVS
    eval_freq = 1000 // NUM_ENVS
    lr=0.1
    hidden_dim = 64
    eps_decay = 0.99995


    # we initialise the Q network with random weights:
    QNet = NNet(state_dim, actions_dim, 1, hidden_dim)
    optimiser = optim.Adam(QNet.parameters(), lr=lr)

    
    returns = []
    UTD_ratio = 8
    m = UTD_ratio // NUM_ENVS  # we'll reset the UTD buffer every m steps

    # s = s0:
    states = vec_env_main.reset()

    for i in range(n_steps + 1):
        # this is basically evaluating every 1k steps (but we're using 8 cores):
        if i%eval_freq == 0 and i!=0:
            R_greedy_tau = evalRollout(env_eval, QNet)
            returns.append(R_greedy_tau)
            print(f'Evaluating policy return after {i} steps: {R_greedy_tau}')
            if i == n_steps:  # last eval, no need to keep training after it
                break

        if i%m == 0:
            UTD_buffer = []  # update to data

        state_tensor = torch.tensor(states, dtype=torch.float32)

        # epsilon-greedy select actions a in argmax(Q(s, a)):
        q_values = QNet(state_tensor)
        actions = np.where(
            np.random.rand(NUM_ENVS) < epsilon
            , np.array([vec_env_main.action_space.sample() for _ in range(NUM_ENVS)])
            , torch.argmax(q_values, dim=1).detach().numpy()
        )
        next_states, rewards, dones, _ = vec_env_main.step(actions)

        # now we calculate the update target:
        next_state_tensor = torch.tensor(next_states, dtype=torch.float32)
        next_q_values = QNet(next_state_tensor)
        max_q, _ = torch.max(next_q_values, dim=1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones)
        # if agent is done in any of the next states then the target q are the terminal rewards:
        target_q = rewards + (gamma * max_q * torch.logical_not(dones))

        predicted_q = q_values.gather(
            1, torch.tensor(actions, dtype=torch.int64).view(-1, 1)
        ).squeeze() # we select the q values according to the argmax actions

        # log predictions and targets in buffer:
        UTD_buffer.append((predicted_q, target_q))

        optimiser.zero_grad()  # need to reset grad every time or it accumulates
        loss = nn.MSELoss()
        input, target = zip(*UTD_buffer)
        tens_input = torch.cat(input)
        tens_target = torch.cat(target)
        output = loss(tens_input, tens_target)
        output.backward()
        optimiser.step()

        epsilon = max(eps_min, epsilon * eps_decay)
        states = next_states

    print(f'Training finished in {round(time.time() - start, 2)} seconds.')

    plt.figure(figsize=(12, 8))

    plt.plot(
        np.linspace(0, n_steps * NUM_ENVS, eval_freq * NUM_ENVS)
        , returns
    )
    plt.xlabel('Million steps')
    plt.ylabel('Return')
    plt.title(f'Evolution of greedy Tau return over {n_steps * NUM_ENVS} steps')

    plt.show()
