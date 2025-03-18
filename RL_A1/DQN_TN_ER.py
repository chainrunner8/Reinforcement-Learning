import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import time
import random
import sys
from torch import nn, optim
from collections import deque
from stable_baselines3.common.vec_env import SubprocVecEnv


if len(sys.argv) > 1:
    NUM_ENVS = int(sys.argv[1])
else:   
    NUM_ENVS = 8  # my cpu is Ryzen 7 5800h with 8 cores
print(f'Running DQN on {NUM_ENVS} cores.')


class NNet(nn.Module):
    
    def __init__(self, state_dim, actions_dim, nb_hidden, hidden_dim):
        ''' creates a neural network with specified dimensions '''
        super(NNet, self).__init__()
        if nb_hidden == 1:
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim)
                , nn.ReLU()
                , nn.Linear(hidden_dim, actions_dim)
            )
        elif nb_hidden == 2:
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim)
                , nn.ReLU()
                , nn.Linear(hidden_dim, hidden_dim)
                , nn.ReLU()
                , nn.Linear(hidden_dim, actions_dim)
            )
        # i could have written this function to support any number of hidden layers with a
        # for loop, but it's irrelevant to the (only 3) network sizes I want to test.
        else:
            raise RuntimeError('Unsupported number of hidden layers for QNet, specify 1 or 2.')

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, size=10**5):
        self.max_len = size
        self.buffer = deque(maxlen=self.max_len)

    def __len__(self):
        return len(self.buffer)

    def log_experience(self, s, a, r, next_s, done):
        ''' adds an experience tuple to the replay buffer'''
        self.buffer.append((s, a, r, next_s, done))

    def sample(self, batch_size):
        ''' samples with batch_size from the replay buffer'''
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones


def makeEnv():
    ''' makes a CartPole environment for the SubprocVecEnv object'''
    return gym.make('CartPole-v1')


def evalRollout(env_eval, QNet):
    ''' performs a single run until terminated or truncated with current 
    QNet following greedy policy (no random actions) '''
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


def DQN(vec_env_main, env_eval, lr, eps_decay, nb_hidden, hidden_dim, n_steps
        , eval_freq, progress, target_upd_rate, replay_buffer=None, batch_size=None):
    ''' runs the DQN algorithm.
    - to run naive, pass target_upd_rate = 1
    - to run target network, specify a different target_upd_rate 
    (WARNING: it'll be a multiple of NUM_ENVS, 
    so target_upd_rate = 1 updates every 8 steps if NUM_ENVS = 8)
    - to run experience replay, pass a ReplayBuffer() object and a batch_size'''
    
    # agent init:
    actions_dim = vec_env_main.action_space.n
    state_dim = vec_env_main.observation_space.shape[0]
    QNet = NNet(state_dim, actions_dim, int(nb_hidden), int(hidden_dim))
    QNet_target = NNet(state_dim, actions_dim, int(nb_hidden), int(hidden_dim))
    optimiser = optim.Adam(QNet.parameters(), lr=lr)
    
    # training params init:
    epsilon = 1
    eps_min = 0.01
    gamma = 1
    n_steps = int(n_steps) // NUM_ENVS
    eval_freq = int(eval_freq) // NUM_ENVS
    loss = nn.MSELoss()

    # s = s0:
    states = vec_env_main.reset()  # the vec envs reset on their own afterwards
    returns = []

    for i in range(n_steps + 1):
        # this is basically evaluating every 1k steps (but we're using 8 cores):
        if i%eval_freq == 0:
            R_greedy_tau = evalRollout(env_eval, QNet)
            returns.append(R_greedy_tau)
            print(f'{progress} at {i * NUM_ENVS} steps: {R_greedy_tau}')
            if i == n_steps:  # last eval, no need to keep training after it
                break

        state_tensor = torch.tensor(states, dtype=torch.float32)
        # epsilon-greedy select actions a in argmax(Q(s, a)):
        q_values = QNet(state_tensor)
        actions = np.where(
            np.random.rand(NUM_ENVS) < epsilon
            , np.array([vec_env_main.action_space.sample() for _ in range(NUM_ENVS)])
            , torch.argmax(q_values, dim=1).detach().numpy()
        )
        next_states, rewards, dones, _ = vec_env_main.step(actions)

        # target network update:
        if i%target_upd_rate == 0:
            QNet_target.load_state_dict(QNet.state_dict())

        if replay_buffer:
            for i in range(NUM_ENVS):
                replay_buffer.log_experience(
                    states[i], actions[i], rewards[i], next_states[i], dones[i]
                )
            
            if len(replay_buffer) > batch_size:
                states_b, actions_b, rewards_b, next_states_b, dones_b = replay_buffer.sample(batch_size)
                
                # now we calculate the update target:
                next_state_tensor = torch.tensor(np.array(next_states_b), dtype=torch.float32)
                next_q_values = QNet_target(next_state_tensor)
                max_q, _ = torch.max(next_q_values, dim=1)
                rewards_b = torch.tensor(rewards_b, dtype=torch.float32)
                dones_b = torch.tensor(dones_b)
                # if agent is done in any of the next states then the target q are the terminal rewards:
                target_q = rewards_b + (gamma * max_q * torch.logical_not(dones_b))
                target_q = target_q.unsqueeze(1)

                state_tensor_b = torch.tensor(np.array(states_b), dtype=torch.float32)
                predicted_q = QNet(state_tensor_b).gather(
                    1, torch.tensor(actions_b, dtype=torch.int64).view(-1, 1)
                ) # select the q values according to the argmax actions

                # log predictions and targets in buffer:
                optimiser.zero_grad()  # need to reset grad every time or it accumulates
                output = loss(predicted_q, target_q)
                output.backward()
                optimiser.step()

        else:
            next_state_tensor = torch.tensor(next_states, dtype=torch.float32)
            next_q_values = QNet_target(next_state_tensor)
            max_q, _ = torch.max(next_q_values, dim=1)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            dones = torch.tensor(dones)
            # if agent is done in any of the next states then the target q are the terminal rewards:
            target_q = rewards + (gamma * max_q * torch.logical_not(dones))
            target_q = target_q.unsqueeze(1)

            predicted_q = q_values.gather(
                1, torch.tensor(actions, dtype=torch.int64).view(-1, 1)
            ) # select the q values according to the argmax actions

            optimiser.zero_grad()  # need to reset grad every time or it accumulates
            output = loss(predicted_q, target_q)
            output.backward()
            optimiser.step()

        epsilon = max(eps_min, epsilon * eps_decay)
        states = next_states

    return returns


def smooth(array, window):
    ''' smooths out the return curves for plotting '''
    smoothed = np.convolve(array, np.ones(window)/window, mode='valid')
    complete = np.concatenate((array[:window - 1], smoothed))
    return complete



if __name__ == "__main__":
    try: 
        # environment init:
        vec_env_main = SubprocVecEnv([makeEnv for _ in range(NUM_ENVS)])
        env_eval = gym.make('CartPole-v1')

        replay_buffer = ReplayBuffer(size=50_000)

        # training params:
        n_steps=10**6
        eval_freq=1_000
        n_trials = 5
        
        lr = 0.0005
        eps_decay = 0.9999
        layers = 1
        neurons = 64

        params = {
            'Naïve': {'hyperparams': [1, None, None], 'colour': 'darkcyan'}  # naïve
            , 'Only TN': {'hyperparams': [50, None, None], 'colour': 'blueviolet'}  # only TN
            , 'Only ER': {'hyperparams': [1, replay_buffer, 32], 'colour': 'firebrick'}  # only ER
            , 'TN + ER': {'hyperparams': [50, replay_buffer, 32], 'colour': 'dodgerblue'}  # TN + ER
        }
        

        plt.figure(figsize=(12, 8))
        for method, param_vec in params.items():
            total_time = 0
            return_array = np.zeros((n_trials, n_steps // eval_freq + 1))
            for i in range(n_trials):
                start = time.time()
                progress = method + f' trial {i}'
                returns = DQN(
                    vec_env_main
                    , env_eval
                    , lr
                    , eps_decay
                    , layers
                    , neurons
                    , n_steps
                    , eval_freq
                    , progress
                    , *param_vec['hyperparams']
                )
                return_array[i] = returns
                trial_time = time.time() - start
                print(f'Trial number {i+1} completed in {round(trial_time, 2)} seconds.')
                total_time += trial_time
            print(f'{n_trials} trials completed in {round(total_time/60, 2)} minutes.')

            mean_returns = np.mean(return_array, axis=0)
            max_returns = np.max(return_array, axis=0)
            min_returns = np.min(return_array, axis=0)

            window = 20
            # moving average of mean returns:
            smoothed_mean = smooth(mean_returns, window)
            smoothed_max = smooth(max_returns, window)
            smoothed_min = smooth(min_returns, window)
        
            plt.plot(
                np.linspace(0, n_steps, eval_freq//NUM_ENVS * NUM_ENVS + 1), smoothed_mean
                , label=method, color=param_vec['colour']
            )
            # plt.fill_between(
            #     np.linspace(0, n_steps, n_steps // eval_freq + 1), smoothed_min, smoothed_max
            #     , color=param_vec['colour'], alpha=0.1
            # )
        plt.xlabel('Million steps')
        plt.ylabel('Average return')
        plt.title(f'Greedy policy average return over {n_steps} steps: naïve, TN and ER')
        plt.legend()
        plt.savefig(f'naive_TN_ER_4.png')

        # # plot heatmap:
        # mean_arr, stdev_arr = zip(*results)
        # mean_arr = np.array(mean_arr).reshape(-1, 3)
        # annotations = np.array([f'{mean:.1f} ± {stdev:.1f}' for mean, stdev in results]).reshape(-1, 3)

        # plt.figure(figsize=(10, 8))
        # sns.heatmap(
        #     mean_arr, annot=annotations, fmt='s', cmap="viridis", xticklabels=[str(x) for x in epsilon_decays]
        #     , yticklabels= [str(x) for x in learning_rates]
        # )
        # plt.xlabel("Epsilon Decay")
        # plt.ylabel("Learning Rate")
        # plt.title("Mean ± stdev of average return for 9 (learning rate, epsilon decay) pairs")
        # plt.savefig('param_tuning/heatmap.png')

    except Exception as e:
        print(e)
    finally:
        if 'vec_env_main' in locals():
            vec_env_main.close()
