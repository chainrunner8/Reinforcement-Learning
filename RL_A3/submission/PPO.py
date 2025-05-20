import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import gymnasium as gym
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import time
from argparse import ArgumentParser
from bufferClass import Buffer


parser = ArgumentParser()
parser.add_argument('--experiment', type=str, help='Possible values: "hyper tuning", "first grid", "second grid", "ablation", "engineering"')
args = parser.parse_args()


torch.set_num_threads(1)

''' CONSTANTS '''
STUDENT_NUMBER = 4645251
NUM_CORES = mp.cpu_count()//2
SEEDS = [STUDENT_NUMBER + 2048*(i+2) for i in range(NUM_CORES)]
TRAIN_EPOCH_STEPS = 2**12
N_STEPS = 10**6
TRAIN_EPOCHS = N_STEPS // TRAIN_EPOCH_STEPS
N_TRIALS = NUM_CORES
TEST_EPISODES = 100
TEST_SAMPLES = 1000
EPSILON = 0.2

class ActorCritic(nn.Module):
    '''
        Creates a two-headed neural network
    '''
    def __init__(self, state_dim, action_dim, hidden_struct):
        super(ActorCritic, self).__init__()
        self.n_hidden, self.hidden_dim = hidden_struct
        self.shared_l1 = nn.Linear(state_dim, self.hidden_dim)
        self.shared_l2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        # actor head:
        self.policy_logits = nn.Linear(self.hidden_dim, action_dim)
        # critic head:
        self.value = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.shared_l1(x))
        if self.n_hidden == 2:
            x = F.relu(self.shared_l2(x))
        logits = self.policy_logits(x)
        value = self.value(x)
        return logits, value


def set_seed(seed, env:gym.Env, env_rollout:gym.Env=None):
    ''' sets random seed '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    if env_rollout:
        env_rollout.action_space.seed(seed)
        env_rollout.observation_space.seed(seed)


def select_action(logits, greedy=False):
    ''' 
        greedy=True is used during rollouts to pick the argmax action
        because during rollouts we're only exploiting, not exploring
    '''
    if greedy:
        probs = torch.softmax(logits, dim=-1)
        return torch.argmax(probs).item()
    else:
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob


def ppo_loss(new_pi, old_pi, advantages, clip:bool):
    ''' computes the policy loss '''
    ratio = torch.exp(new_pi - old_pi)
    if clip:
        clipped_ratio = torch.clamp(ratio, 1-EPSILON, 1+EPSILON)
        L_clip = torch.min(ratio*advantages, clipped_ratio*advantages)
        return -L_clip.mean()
    else:
        return -(ratio*advantages).mean()


def gradient_descent(actor_critic, data, optimiser:optim, batch_size, clip:bool, clip_grad:bool):
    ''' updates the gradient across all mini-batches '''
    for i in range(0, TRAIN_EPOCH_STEPS, batch_size):
        end = i + batch_size
        # prepare batch:
        batch_states = data['states'][i:end]
        batch_actions = data['actions'][i:end]
        batch_returns = data['returns'][i:end]
        batch_advantages = data['advantages'][i:end]
        # assert batch_advantages.dtype == torch.float32, f'batch_advantages dtype: {batch_advantages.dtype}'
        batch_old_logpi = data['old_logpi'][i:end]

        new_logits, new_values = actor_critic(batch_states)
        dist = Categorical(logits=new_logits)
        new_logpi = dist.log_prob(batch_actions)

        optimiser.zero_grad()
        actor_loss = ppo_loss(new_logpi, batch_old_logpi, batch_advantages, clip)
        critic_loss = F.mse_loss(new_values.squeeze(-1), batch_returns)
        loss = actor_loss + critic_loss
        loss.backward()
        if clip_grad:
            clip_grad_norm_(actor_critic.parameters(), max_norm=0.5)
        optimiser.step()

    return actor_critic


def eval_rollout(environment:gym.Env, actor_critic, seed):
    ''' performs one test run on greedy policy '''
    done = False
    cumul_r = 0
    state = environment.reset(seed=seed)[0]
    while not done:
        state = torch.from_numpy(state)
        logits, _ = actor_critic(state)
        action = select_action(logits, greedy=True)
        next_state, _, terminated, truncated, _ = environment.step(action)
        done = terminated or truncated
        cumul_r += 1
        state = next_state
    return cumul_r


def ppo(learning_rate, network_struct, gd_epochs, batch_size, seed, perform_rollout:bool=False, clip:bool=True
        , normalise_advantages:bool=True, normalise_observations:bool=False, clip_gradient:bool=False):
    
    ''' 
        performs an N_STEPS training run of the PPO-Clip algo
        returns either test run data for bootstrapping (grid search) or rollout data 
    '''

    env = gym.make('CartPole-v1')
    if perform_rollout:
        env_rollout = gym.make('CartPole-v1')
        set_seed(seed, env, env_rollout)
    else:
        set_seed(seed, env)
    rollout_returns = [0]
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    actor_critic = ActorCritic(state_dim, action_dim, network_struct)
    optimiser = optim.Adam(actor_critic.parameters(), lr=learning_rate)
    state = env.reset(seed=seed)[0]

    for i in range(TRAIN_EPOCHS):
        buffer = Buffer(size=TRAIN_EPOCH_STEPS, state_dim=state_dim)

        for j in range(TRAIN_EPOCH_STEPS):
            state_t = torch.from_numpy(state)
            logits, value = actor_critic(state_t)
            action, log_prob = select_action(logits)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # log data:
            buffer.log(state, action, reward, log_prob, value)
            # if we're either done or time for training + rollout, we stop:
            if done or j == TRAIN_EPOCH_STEPS - 1:
                last_value = 0 if done else value.item()
                buffer.finish_trajectory(last_value)
                state = env.reset(seed=seed)[0]
                done = False
            else:
                state = next_state

        data = buffer.get_data(normalise_advantages, normalise_observations)
        for _ in range(gd_epochs):
            data = buffer.shuffle(data)
            actor_critic = gradient_descent(actor_critic, data, optimiser, batch_size, clip
                                            , clip_gradient)
        # rollout:
        if perform_rollout:
            ep_return = eval_rollout(env_rollout, actor_critic, seed)
            rollout_returns.append(ep_return)

    if perform_rollout:
        env.close()
        env_rollout.close()
    else:
        for _ in range(TEST_EPISODES):
            ep_return = eval_rollout(env, actor_critic, seed)
            rollout_returns.append(ep_return)
        env.close()
        # bootstrap_samples = np.random.choice(rollout_returns, size=TEST_EPISODES, replace=True)
    return rollout_returns


def smooth_curve(x, window):
    ''' smooths out the return curves for plotting '''
    smoothed = np.convolve(x, np.ones(window)/window, mode='valid')
    complete = np.concatenate((x[:window - 1], smoothed))
    return complete


def process_curve(return_array):
    ''' prepares solid curve and shaded regions '''
    mean_returns = np.mean(return_array, axis=1)
    stdev = np.std(return_array, axis=1)
    lower_stdev = mean_returns - stdev
    upper_stdev = mean_returns + stdev

    window = 10
    # moving average of mean returns:
    smoothed_mean = smooth_curve(mean_returns, window)
    smoothed_lower = smooth_curve(lower_stdev, window)
    smoothed_upper = smooth_curve(upper_stdev, window)

    return {'mean': smoothed_mean, 'lstd': smoothed_lower, 'ustd': smoothed_upper}


def ppo_tuning(lr, net_struct, gd_epochs, batch_size, col, label, rollouts=True, clip=True, norm_adv=True
               , norm_obs=False, clip_grad=False):

    '''
        performs the ppo() function in parallel for each random seed
        plots the return curves
    '''

    start = time.time()
    print(f'Training started for {label} configuration.')
    with mp.Pool(processes=NUM_CORES) as pool:
        results = pool.starmap(
            ppo
            , [[lr, net_struct, gd_epochs, batch_size, seed, rollouts, clip, norm_adv, norm_obs, clip_grad]
                for seed in SEEDS]
        )
    print(f'Training of {label} configuration completed in {round( (time.time() - start)/60 , 1)} minutes.')

    return_array = np.column_stack(results)
    curve = process_curve(return_array)
    mean = curve['mean']
    lower = curve['lstd']
    upper = curve['ustd']
    x = np.linspace(0, N_STEPS, N_STEPS // TRAIN_EPOCH_STEPS + 1)
    plt.plot(
        x, mean, label=label, color=col
    )
    plt.fill_between(
        x, lower, upper, color=col, alpha=0.3
    )


def get_bootstrapped_stats(returns):
    '''
        step 1: return list of 100 returns from 8 trials
        step 2: concatenate these lists
        step 3: sample n=100*NUM_CORES 1000 times
        step 4: calculate the mean and the variance of each sample
        step 5: average them both over the 1000 samples
        step 6: estimate the standard error for both
    '''
    boot_stats = np.zeros((TEST_SAMPLES, 2))
    for i in range(TEST_SAMPLES):
        sample = np.random.choice(returns, size=len(returns), replace=True)
        boot_stats[i] = [sample.mean(), sample.std()]
    boot_mean, boot_std = boot_stats.mean(axis=0).round(2)
    se_boot_mean = round(np.sqrt(np.sum((boot_stats[:, 0] - boot_mean)**2) / (TEST_SAMPLES - 1)), 2)
    se_boot_std = round(np.sqrt(np.sum((boot_stats[:, 1] - boot_std)**2) / (TEST_SAMPLES - 1)), 2)
    return boot_mean, se_boot_mean, boot_std, se_boot_std


def grid_search(grid_dict):
    ''' 
        computes all possible hyperparam pairs and runs ppo() for each
        saves result table as csv
        plots heatmap
    '''

    if grid_dict['pair'] == 'lr_net_struct':
        hyper_1, hyper_2 = 'lr', 'net_struct'
    else:
        hyper_1, hyper_2 = 'gd_epochs', 'batch_size'

    flattened = []
    table_rows = []
    k = 1
    for combination in grid_dict['combinations']:
        start = time.time()
        print(
            f'Training started for pair n°{k}: {hyper_1}={combination[hyper_1]}, {hyper_2}={combination[hyper_2]}'
        )
        with mp.Pool(processes=NUM_CORES) as pool:
            results = pool.starmap(ppo, [combination['params'] + [SEEDS[i]] for i in range(len(SEEDS))])
            returns = list(itertools.chain.from_iterable(results))
            boot_mean, se_boot_mean, boot_std, se_boot_std = get_bootstrapped_stats(returns)
            table_rows.append(dict(
                pair=f'{hyper_1}={combination[hyper_1]}, {hyper_2}={combination[hyper_2]}'
                , mean=boot_mean
                , se_mean=se_boot_mean
                , std=boot_std
                , se_std=se_boot_std
            ))
            flattened.append({
                hyper_1: combination[hyper_1]
                , hyper_2: combination[hyper_2]
                , 'test_mean': boot_mean
                , 'test_std': boot_std
            })
        print(f'Finished pair n°{k} in {round((time.time()-start) / 60, 1)} minutes')
        k += 1
    df_table = pd.DataFrame(table_rows)
    df_table.to_csv(f'{grid_dict['pair']}_table.csv')
    df_results = pd.DataFrame(flattened)

    # heatmaps:
    pivoted_means = df_results.pivot(index=hyper_1, columns=hyper_2, values='test_mean')
    pivoted_stdevs = df_results.pivot(index=hyper_1, columns=hyper_2, values='test_std')
    annotations = np.array([
        f'{mean:.1f} ± {std:.1f}'
        for mean, std in zip(pivoted_means.to_numpy().ravel(), pivoted_stdevs.to_numpy().ravel())
    ]).reshape(pivoted_means.shape)
    plt.figure(figsize=(12,8))
    sns.heatmap(pivoted_means, annot=annotations, fmt='s', cmap='viridis')
    # plt.title("Test Mean Return Heatmap (1000-sample bootstrap)")
    plt.savefig(f'{grid_dict['pair']}_heatmap.png')




'''

FUNCTIONS TO RUN ARTICLE EXPERIMENTS

'''


def first_grid_search():

    learning_rates = [0.00005, 0.0001, 0.00025, 0.0005]
    net_structs = [(1, 32), (1, 64), (2, 32), (2, 64)]
    pairs = list(itertools.product(learning_rates, net_structs))
    first_search_dict = dict(
        pair='lr_net_struct'
        , combinations=[
            {
                'lr': p[0]
                , 'net_struct': p[1]
                , 'params': [p[0], p[1], 10, 64]
            }
            for p in pairs
        ]
    )
    grid_search(first_search_dict)


def second_grid_search():

    gd_epochs = [1, 5, 10, 25]
    batch_sizes = [32, 64, 128, 256]
    pairs = list(itertools.product(gd_epochs, batch_sizes))
    second_search_dict = dict(
        pair='epochs_batch_size'
        , combinations=[
            {
                'gd_epochs': p[0]
                , 'batch_size': p[1]
                , 'params': [0.00025, (1, 32), p[0], p[1]]
            }
            for p in pairs
        ]
    )
    grid_search(second_search_dict)


def ppo_hypertuning():

    plt.figure(figsize=(12,8))

    label = 'lr=.0005, net_struct=(1, 64), gd_epochs=10, batch_size=32'
    ppo_tuning(lr=.0005, net_struct=(1, 64), gd_epochs=10, batch_size=32, col='mediumturquoise', label=label)

    label = 'lr=.0005, net_struct=(1, 64), gd_epochs=10, batch_size=64'
    ppo_tuning(lr=.0005, net_struct=(1, 64), gd_epochs=10, batch_size=64, col='darkorange', label=label)

    label = 'lr=.0005, net_struct=(1, 64), gd_epochs=10, batch_size=128'
    ppo_tuning(lr=.0005, net_struct=(1, 64), gd_epochs=10, batch_size=128, col='indigo', label=label)

    label = 'lr=.0005, net_struct=(1, 32), gd_epochs=10, batch_size=64'
    ppo_tuning(lr=.0005, net_struct=(1, 32), gd_epochs=10, batch_size=64, col='forestgreen', label=label)

    x = np.linspace(0, N_STEPS, N_STEPS // TRAIN_EPOCH_STEPS + 1)
    plt.plot(x, [500]*len(x), linestyle='--', color='black')
    plt.ylim((0, 700))
    plt.xlabel('Steps')
    plt.ylabel('Average return')
    plt.title(f'Average return over {'{:.0e}'.format(N_STEPS)} steps')
    plt.legend(loc='upper right')
    plt.show()


def ppo_ablation():

    plt.figure(figsize=(12,8))

    label = 'Clipping and advantage normalisation'
    ppo_tuning(lr=.0005, net_struct=(1, 64), gd_epochs=10, batch_size=32, col='mediumturquoise', label=label)

    label = 'No clipping'
    ppo_tuning(lr=.0005, net_struct=(1, 64), gd_epochs=10, batch_size=32, col='darkorange', label=label, rollouts=True, clip=False, norm_adv=True)

    label = 'No advantage normalisation'
    ppo_tuning(lr=.0005, net_struct=(1, 64), gd_epochs=10, batch_size=32, col='indigo', label=label, rollouts=True, clip=True, norm_adv=False)

    x = np.linspace(0, N_STEPS, N_STEPS // TRAIN_EPOCH_STEPS + 1)
    plt.plot(x, [500]*len(x), linestyle='--', color='black')
    plt.ylim((0, 700))
    plt.xlabel('Steps')
    plt.ylabel('Average return')
    plt.title(f'Average return over {'{:.0e}'.format(N_STEPS)} steps')
    plt.legend(loc='upper right')
    plt.show()


def ppo_engineering():

    plt.figure(figsize=(12,8))

    label = 'Baseline'
    ppo_tuning(lr=.0005, net_struct=(1, 64), gd_epochs=10, batch_size=32, col='mediumturquoise', label=label)

    label = 'Observation normalisation'
    ppo_tuning(lr=.0005, net_struct=(1, 64), gd_epochs=10, batch_size=32, col='darkorange', label=label, norm_obs=True)

    label = 'Gradient clipping'
    ppo_tuning(lr=.0005, net_struct=(1, 64), gd_epochs=10, batch_size=32, col='indigo', label=label, clip_grad=True)

    x = np.linspace(0, N_STEPS, N_STEPS // TRAIN_EPOCH_STEPS + 1)
    plt.plot(x, [500]*len(x), linestyle='--', color='black')
    plt.ylim((0, 700))
    plt.xlabel('Steps')
    plt.ylabel('Average return')
    plt.title(f'Average return over {'{:.0e}'.format(N_STEPS)} steps')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    if args.experiment == "first grid":
        first_grid_search()
    elif args.experiment == "second grid":
        second_grid_search()
    elif args.experiment == "hyper tuning":
        ppo_hypertuning()
    elif args.experiment == "ablation":  
        ppo_ablation()
    elif args.experiment == "engineering":
        ppo_engineering()
    else:
        raise RuntimeError('Invalid experiment. Must be one of: "hyper tuning", "first grid", "second grid", "ablation", "engineering"')
    
    # winning hyperparam vector: lr=.0005, net_struct=(1, 64), gd_epochs=10, batch_size=32
