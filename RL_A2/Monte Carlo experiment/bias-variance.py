import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import multiprocessing as mp
import pandas as pd
import seaborn as sns
import time
import warnings
import itertools
from dataclasses import dataclass


warnings.simplefilter(action='ignore', category=FutureWarning)
torch.set_num_threads(1)


''' TRAINING CONSTANTS '''
NUM_CORES = 8
N_TRIALS = 20
N_STEPS = 10**6
LOG_INTERVAL = 5_000  # must be >500
LEARNING_RATE = 0.005
GAMMA = 0.999
MC_TRIALS = 1_000


''' CLASSES '''
@dataclass
class TrainOutput:
    actor_net:nn.Sequential
    critic_net:nn.Sequential
    return_list:list


''' FUNCTIONS '''

def QNet(state_dim, action_dim, hidden_dim, n_hidden) -> torch.nn.Sequential:
    layers=[ nn.Linear(state_dim, hidden_dim),
    nn.ReLU()]
    for _ in range(n_hidden-1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim, action_dim))
    return nn.Sequential(*layers)


def action_selection(policy, state, greedy=False):
    state = torch.FloatTensor(state)
    probs = torch.softmax(policy(state), dim=-1)
    if greedy:
        return torch.argmax(probs).item()  # for testing
    else:
        return torch.multinomial(probs, 1).item()


def get_reward(t, T, rewards):
    reward = 0
    for k in range(t,T):
        reward += rewards[k]*GAMMA**(k-t)
    return reward


def eval_rollout(env, Qnet):
    s = env.reset()[0]
    done = False
    cumul_r = 0
    while not done:
        a = action_selection(Qnet, s, greedy=True)
        s_prime, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        cumul_r += r
        s = s_prime
    return cumul_r


def smooth(array, window):
    ''' smooths out the return curves for plotting '''
    smoothed = np.convolve(array, np.ones(window)/window, mode='valid')
    complete = np.concatenate((array[:window - 1], smoothed))
    return complete


def train_A2C(environment, actor, critic, m, n):
    optim_actor = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    optim_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE)
    mse_loss = nn.MSELoss()
    steps_taken = 0
    logged_steps = set()
    returns = []

    while steps_taken < N_STEPS:
        batch_states, batch_actions, batch_qvals = [], [], []

        for _ in range(m):  # sample M trajectories
            if steps_taken > N_STEPS:  # for M>1
                break
            s = environment.reset()[0]
            done =  False
            cur_states, cur_rewards = [], []
            while not done:  # actor walks the trajectory
                a = action_selection(actor, s)
                s_prime, r, terminated, truncated, _  = environment.step(a)
                cur_states.append(s)
                batch_actions.append(a)
                cur_rewards.append(r)
                done = terminated or truncated
                s = s_prime
            T = len(cur_rewards)
            ep_return = sum(cur_rewards)
            steps_taken += ep_return
            # logging:
            step_bin = int((steps_taken // LOG_INTERVAL) * LOG_INTERVAL)
            if step_bin not in logged_steps:
                if m==1 and n==100:
                    print(step_bin)
                logged_steps.add(step_bin)
                cumul_r = eval_rollout(environment, actor)
                returns.append(cumul_r)

            cur_state_tensor = torch.tensor(np.array(cur_states))
            for t in range(T):  # compute Q estimates for each step of the trajectory
                discounted_reward = get_reward(t, min(t+n, T), cur_rewards)
                if t+n < T:
                    qval = discounted_reward + (GAMMA**n) * critic(cur_state_tensor[t+n])
                else:
                    qval = torch.tensor([discounted_reward])
                batch_qvals.append(qval)
            batch_states.extend(cur_states)

        # prepare backprop:
        state_tensor = torch.tensor(np.array(batch_states))
        log_probs = F.log_softmax(actor(state_tensor), dim=-1)
        log_action_probs = torch.stack([log_probs[i, batch_actions[i]] for i in range(len(batch_actions))])
        bootstraps = critic(state_tensor)  # = Vphi(st)
        qvals = torch.stack(batch_qvals)  # = Qn(st, at)
        # An(at, at) = Qn(st, at) - Vphi(st)
        # actor loss:
        optim_actor.zero_grad()
        # you need to detach qvals and bootstraps here because the actor part of the loss isn't supposed to update
        # the parameters of the critic network (according to the algorithm):
        actor_loss = -((qvals.detach() - bootstraps.detach()) * log_action_probs).mean()
        # critic loss:
        optim_critic.zero_grad()
        critic_loss = mse_loss(qvals, bootstraps)
        # critic_loss = ((qvals - bootstraps)**2).sum()
        # backprop:
        (actor_loss + critic_loss).backward()
        optim_actor.step()
        optim_critic.step()

    return TrainOutput(actor_net=actor, critic_net=critic, return_list=returns)


def test_A2C(env, actor, critic):

    returns, advantage_var, econfs = [], [], []
    for _ in range(MC_TRIALS):  # sample Monte Carlo trajectories
        s = env.reset()[0]
        done = False
        states, rewards, qvals = [], [], []
        while not done:  # actor walks the trajectory
            a = action_selection(actor, s, greedy=True)
            s_prime, r, terminated, truncated, _ = env.step(a)
            states.append(s)
            rewards.append(r)
            done = terminated or truncated
            s = s_prime
        T = len(rewards)
        ep_return = sum(rewards)
        returns.append(ep_return)  # logging

        state_tensor = torch.tensor(np.array(states))
        for t in range(T):  # compute full Monte Carlo returns for each step of the trajectory
            qval = get_reward(t, T, rewards)
            qvals.append(qval)

        # compute confidence interval:
        V_phi = critic(state_tensor)  # = Vphi(st)
        mc_returns = torch.tensor(qvals)  # = Qn(st, at)
        advantages = mc_returns - V_phi
        mean_adv = advantages.mean().item()
        std_adv = advantages.std().item()
        std_mean = std_adv/np.sqrt(T)
        conf_int = (mean_adv - 1.96 * std_mean, mean_adv + 1.96 * std_mean)
        econf = conf_int[0] <= 0 <= conf_int[1]
        econfs.append(econf)
        advantage_var.append(std_adv**2)

    return_arr = np.array(returns)
    mean_return = return_arr.mean()
    std_return = return_arr.std()
    mean_econf = sum(econfs) / len(econfs)
    mean_var_adv = sum(advantage_var) / len(advantage_var)

    df_return = pd.DataFrame({
        'mean return': [mean_return]
        , 'std return': [std_return]
        , 'mean econf': [mean_econf]
        , 'mean var advantage': [mean_var_adv]
    })
    return df_return


def train_and_test(param_pair):

    m, n = param_pair
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    train_return_array = np.zeros((N_TRIALS, N_STEPS // LOG_INTERVAL + 1))
    df_test = pd.DataFrame()

    start = time.time()
    for t in range(N_TRIALS):
        actor = QNet(state_dim, action_dim, 64, 1)
        critic = QNet(state_dim, 1, 64, 1)
        train_output = train_A2C(env, actor, critic, m, n)
        test_output = test_A2C(env, train_output.actor_net, train_output.critic_net)
        train_return_array[t] = train_output.return_list
        df_test = pd.concat([df_test, test_output])
        print(f'm = {m}, n = {n} trial {t+1} done.')
    env.close()
    print(f'm = {m}, n = {n} train-test completed in {round((time.time()-start)/60, 1)} minutes.')
    
    # process training results:
    mean_returns = np.mean(train_return_array, axis=0)
    stdev = np.std(train_return_array, axis=0)
    lower_stdev = mean_returns - stdev
    upper_stdev = mean_returns + stdev

    window = 1
    # moving average of mean returns:
    smoothed_mean = smooth(mean_returns, window)
    smoothed_lower = smooth(lower_stdev, window)
    smoothed_upper = smooth(upper_stdev, window)

    df_train = pd.DataFrame({'mean returns': smoothed_mean, 'lstd returns': smoothed_lower, 'ustd returns': smoothed_upper})

    result_dict = {
        param_pair: {
            'train': df_train
            , 'test': df_test.mean()
        }
    }
    return result_dict


def grid_search():
    
    m_vals = [1, 5, 10, 50, 100]
    n_vals = [10, 50, 100, 250, 500]
    combinations = list(itertools.product(m_vals, n_vals))
    
    start = time.time()
    with mp.Pool(processes=NUM_CORES) as pool:
        results = pool.map(train_and_test, combinations)
        print(f'Grid search completed in {round( (time.time() - start)/60 , 1)} minutes.')
    
    flattened = []
    for result in results:
        for (m, n), data in result.items():
            flattened.append({
                'm': m
                , 'n': n
                , 'train mean returns': data['train']['mean returns']
                , 'train lstd returns': data['train']['lstd returns']
                , 'train ustd returns': data['train']['ustd returns']
                , 'test mean return': data['test']['mean return']
                , 'test std return': data['test']['std return']
                , 'test mean econf': data['test']['mean econf']
                , 'test mean var advantage': data['test']['mean var advantage']
            })
    df_results = pd.DataFrame(flattened)

    # return curves:
    x = np.linspace(0, N_STEPS, N_STEPS // LOG_INTERVAL + 1)
    colours = ['sienna', 'fuchsia', 'mediumturquoise', 'navy', 'goldenrod']

    for m in df_results['m'].unique():
        subset = df_results[df_results['m'] == m].copy()
        subset['colour'] = colours
        plt.figure(figsize=(12,8))
        for _, row in subset.iterrows():
            n = row['n']
            mean = row['train mean returns']
            lower = row['train lstd returns']
            upper = row['train ustd returns']
            plt.plot(
                x, mean, label=f'n = {n}', color=row['colour']
            )
            plt.fill_between(
                x, lower, upper, color=row['colour'], alpha=0.1
            )
        plt.ylim((0, 650))
        plt.xlabel('Million steps')
        plt.ylabel('Average return')
        plt.title(f'Average return over {'{:.0e}'.format(N_STEPS)} steps, m = {m}')
        plt.legend(loc='upper right')
        plt.savefig(f'grid_search_m={m}.png')

    # heatmaps:
    pivoted_means = df_results.pivot(index='m', columns='n', values='test mean return')
    pivoted_stdevs = df_results.pivot(index='m', columns='n', values='test std return')
    annotations = np.array([
        f'{mean:.1f} ± {std:.1f}'
        for mean, std in zip(pivoted_means.to_numpy().ravel(), pivoted_stdevs.to_numpy().ravel())
    ]).reshape(pivoted_means.shape)
    plt.figure(figsize=(12,8))
    sns.heatmap(pivoted_means, annot=annotations, fmt='s', cmap='viridis')
    plt.title("Test Mean Return Heatmap")
    plt.savefig(f'return_heatmap.png')

    pivoted = df_results.pivot(index='m', columns='n', values='test mean econf')
    plt.figure(figsize=(12,8))
    sns.heatmap(pivoted, annot=True, cmap='viridis')
    plt.title("Test Empirical Confidence Interval Heatmap")
    plt.savefig(f'econf_heatmap.png')

    pivoted = df_results.pivot(index='m', columns='n', values='test mean var advantage')
    plt.figure(figsize=(12,8))
    sns.heatmap(pivoted, annot=True, cmap='viridis')
    plt.title("Test Mean Advantage Variance Heatmap")
    plt.savefig(f'advantage_var_heatmap.png')


if __name__ == "__main__":
    grid_search()
