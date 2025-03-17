import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
# from stable_baselines3.common.vec_env import SubprocVecEnv



class NNet(nn.Module):
    
    def __init__(self, state_dim, actions_dim, hidden_dim):
        super(NNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim)
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
        state_tensor = torch.tensor(state).unsqueeze(0)
        q_values = QNet(state_tensor)
        action = torch.argmax(q_values).item()
        next_state, reward, terminated, truncated, _ = env_eval.step(action)
        over = terminated or truncated
        return_ += reward
        state = next_state
    return return_


if __name__ == "__main__":

    # quick check if running gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using {torch.cuda.get_device_name(0)}")

    num_envs = 8
    env = gym.make('CartPole-v1')
    env_eval = gym.make('CartPole-v1')
    actions_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]
    hidden_dim = 64

    epsilon = 1
    eps_decay = 0.99995
    eps_min = 0.01
    gamma = 1
    lr = 0.01

    # we initialise the Q network with random weights:
    QNet = NNet(state_dim, actions_dim, hidden_dim)
    optimiser = optim.Adam(QNet.parameters(), lr=lr)

    n_steps = 10**6
    eval_freq = 10_000
    returns = [0]
    UTD_ratio = 5
    done = True
    
    for i in range(1, n_steps + 1):
        
        if i%eval_freq == 0:
            R_greedy_tau = evalRollout(env_eval, QNet)
            returns.append(R_greedy_tau)
            print(f'Evaluating policy return after {i} steps: {R_greedy_tau}')
        
        # s = s0:
        if done:
            state, _ = env.reset()
            done = False

        if i%UTD_ratio == 1:
            UTD_buffer = []  # update to data

        state_tensor = torch.tensor(state).unsqueeze(0)

        # epsilon-greedy select action a in argmax(Q(s, a)):
        q_values = QNet(state_tensor)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            # obtain actions probabilities with NNet:
            action = torch.argmax(q_values).item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # now we calculate the update target:
        next_state_tensor = torch.tensor(next_state).unsqueeze(0)
        next_q_values = QNet(next_state_tensor)
        max_q, _ = torch.max(next_q_values, dim=1)
        # if agent is done in next state, then the target q is the terminal reward:
        target_q = reward + (gamma * max_q * (1 - done))
        predicted_q = q_values[0, action].unsqueeze(0)

        # log prediction and target in buffer:
        UTD_buffer.append((predicted_q, target_q))

        if i%UTD_ratio == 0:
            optimiser.zero_grad()  # need to reset grad every time or it accumulates
            loss = nn.MSELoss()
            input, target = zip(*UTD_buffer)
            tens_input = torch.stack(input)
            tens_target = torch.stack(target)
            output = loss(tens_input, tens_target)
            output.backward()
            optimiser.step()
        
        epsilon = max(eps_min, epsilon * eps_decay)
        state = next_state



    plt.figure(figsize=(12, 8))

    plt.plot(
        np.linspace(0, n_steps, eval_freq + 1)
        , returns
    )
    plt.xlabel('Million steps')
    plt.ylabel('Return')
    plt.title(f'Evolution of greedy Tau return over {n_steps}')

    plt.show()