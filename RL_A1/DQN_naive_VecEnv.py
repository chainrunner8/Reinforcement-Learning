import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import time
import itertools
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


def makeEnv():
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


def runDQN(vec_env_main, env_eval, lr, eps_decay, nb_hidden, hidden_dim, UTD_ratio, n_steps, eval_freq):

    # agent init:
    actions_dim = vec_env_main.action_space.n
    state_dim = vec_env_main.observation_space.shape[0]
    QNet = NNet(state_dim, actions_dim, int(nb_hidden), int(hidden_dim))
    optimiser = optim.Adam(QNet.parameters(), lr=lr)
     
    # training params init:
    epsilon = 1
    eps_min = 0.01
    gamma = 1
    n_steps = int(n_steps) // NUM_ENVS
    eval_freq = int(eval_freq) // NUM_ENVS
    UTD_ratio = int(UTD_ratio)
    loss = nn.MSELoss()

    # s = s0:
    states = vec_env_main.reset()
    returns = []

    for i in range(n_steps + 1):
        # this is basically evaluating every 1k steps (but we're using 8 cores):
        if i%eval_freq == 0:
            R_greedy_tau = evalRollout(env_eval, QNet)
            returns.append(R_greedy_tau)
            print(f'Evaluating policy return after {i * NUM_ENVS} steps: {R_greedy_tau}')
            if i == n_steps:  # last eval, no need to keep training after it
                break

        if i%UTD_ratio == 0:  # UTD ratio is a multiple of the number of cores (NUM_ENVS)
            UTD_buffer = []  # note: this is not a replay buffer

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
        ).squeeze() # select the q values according to the argmax actions

        # log predictions and targets in buffer:
        UTD_buffer.append((predicted_q, target_q))

        if len(UTD_buffer) == UTD_ratio:
            optimiser.zero_grad()  # need to reset grad every time or it accumulates
            input, target = zip(*UTD_buffer)
            tens_input = torch.cat(input)
            tens_target = torch.cat(target)
            output = loss(tens_input, tens_target)
            output.backward()
            optimiser.step()

        epsilon = max(eps_min, epsilon * eps_decay)
        states = next_states

    return returns


def smooth(array, window):
    smoothed = np.convolve(array, np.ones(window)/window, mode='valid')
    complete = np.concatenate((array[:window - 1], smoothed))
    return complete



if __name__ == "__main__":
    try: 
        plt.figure(figsize=(12, 8))
        plt.savefig(f'param_tuning/test.png')
        # environment init:
        vec_env_main = SubprocVecEnv([makeEnv for _ in range(NUM_ENVS)])
        env_eval = gym.make('CartPole-v1')

        # training params:
        n_steps=10**6
        eval_freq=1_000
        n_trials = 5
        
        # hyperparam tuning:
        # learning_rates = [0.0001, 0.0005, 0.001]
        # epsilon_decays = [0.9999, 0.99995, 0.99999]
        #     # decay of 0.9999 -> eps min reached after 46k steps
        #     # decay of 0.99999 -> eps min reached after 460k steps
        network_sizes = [(1, 64), (2, 64), (2, 128)]
        UTD_ratios = [1, 5, 10]

        # first we tune the learning rate and epsilon decay via grid search
        # while keeping other hyperparams at medium:
        # combinations = list(itertools.product(learning_rates, epsilon_decays))
        # QNet_params = [[lr, eps_dk, 2, 64, 5] for lr, eps_dk in combinations]
        net_size_params = [[0.0005, 0.9999, layers, neurons, 5] for layers, neurons in network_sizes]
        # UTD_params = [[0.0005, 0.9999, 2, 64, utd] for utd in UTD_ratios]
        QNet_params = net_size_params

        results = []
        k=0  # image counter
        colours = ['darkcyan', 'blueviolet', 'firebrick']

        for param_vec in QNet_params:
            total_time = 0

            return_array = np.zeros((n_trials, n_steps // eval_freq + 1))
            for i in range(n_trials):
                start = time.time()
                returns = runDQN(
                    vec_env_main
                    , env_eval
                    , *param_vec
                    , n_steps
                    , eval_freq)
                return_array[i] = returns
                trial_time = time.time() - start
                print(f'Trial number {i+1} completed in {round(trial_time, 2)} seconds.')
                total_time += trial_time
            print(f'{n_trials} trials completed in {round(total_time/60, 2)} minutes.')
            
            mean_returns = np.mean(return_array, axis=0)
            max_returns = np.max(return_array, axis=0)
            min_returns = np.min(return_array, axis=0)

            results.append((np.mean(mean_returns), np.std(mean_returns)))

            window = 20
            # moving average of mean returns:
            smoothed_mean = smooth(mean_returns, window)
            smoothed_max = smooth(max_returns, window)
            smoothed_min = smooth(min_returns, window)

            plt.figure(figsize=(12, 8))
            plt.plot(
                np.linspace(0, n_steps, n_steps // eval_freq + 1), smoothed_mean
                , label='average return', color='darkcyan'
            )
            plt.fill_between(
                np.linspace(0, n_steps, n_steps // eval_freq + 1), smoothed_min, smoothed_max
                , color='cyan', alpha=0.1
            )
            plt.xlabel('Million steps')
            plt.ylabel('Average return')
            plt.title(f'Greedy policy average return over {n_steps} steps (lr={param_vec[0]}, eps_decay={param_vec[1]})')
            plt.legend()
            plt.savefig(f'param_tuning/network_size.png')
            k+=1

        # plot heatmap:
        mean_arr, stdev_arr = zip(*results)
        mean_arr = np.array(mean_arr).reshape(-1, 3)
        # stdev_arr = list(zip(*(iter(stdev_arr),)*3))
        annotations = np.array([f'{mean:.1f} ± {stdev:.1f}' for mean, stdev in results]).reshape(-1, 3)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            mean_arr, annot=annotations, fmt='s', cmap="viridis", xticklabels=[str(x) for x in epsilon_decays]
            , yticklabels= [str(x) for x in learning_rates]
        )
        plt.xlabel("Epsilon Decay")
        plt.ylabel("Learning Rate")
        plt.title("Mean ± stdev of average return for 9 (learning rate, epsilon decay) pairs")
        plt.savefig('param_tuning/heatmap.png')

    except Exception as e:
        print(e)
    finally:
        if 'vec_env_main' in locals():
            vec_env_main.close()
