import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import multiprocessing as mp
import time
import sys
import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)
torch.set_num_threads(1)


''' ACTOR-CRITIC CONSTANTS '''
M = 1  # number of Monte Carlo trajectories to sample per batch
n = 50  # number of steps for bootstrapping
''' TRAINING CONSTANTS '''
N_TRIALS = 20
N_STEPS = 10**6
LOG_INTERVAL = 2000  # must be >500


''' FUNCTIONS '''

def createPolicy(state_dim, action_dim, hidden_dim, n_hidden) -> torch.nn.Sequential:
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


def get_reward(t, T, gamma, rewards):
    reward = 0
    for k in range(t,T):
        reward += rewards[k]*gamma**(k-t)
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


def reinforce(environment, policy:nn.Sequential, learning_rate, gamma):
    optimiser = optim.Adam(policy.parameters(), lr=learning_rate)
    steps_taken = 0
    logged_steps = set()
    returns = []

    while steps_taken < N_STEPS:
        s = environment.reset()[0]
        done = False
        states = []
        actions = []
        rewards = []
        while not done:
            a = action_selection(policy,s)
            s_prime, r, terminated, truncated, _ = environment.step(a)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            done = terminated or truncated
            s = s_prime

        T = len(rewards)
        episode_return = sum(rewards)
        steps_taken += episode_return
        # logging:
        step_bin = int((steps_taken // LOG_INTERVAL) * LOG_INTERVAL)
        if step_bin not in logged_steps:
            logged_steps.add(step_bin)
            cumul_r = eval_rollout(environment, policy)
            returns.append(cumul_r)

        qvals = torch.zeros(T)
        for t in range(T):
            qvals[t] = get_reward(t, T, gamma, rewards)
        state_tensor = torch.FloatTensor(np.array(states))
        log_probs = F.log_softmax(policy(state_tensor), dim=-1)
        log_action_probs = torch.stack([log_probs[t, actions[t]] for t in range(T)])
        loss = -(qvals * log_action_probs).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    return returns


def AC(environment, actor, critic, learning_rate, gamma, M, n):
    optim_actor = optim.Adam(actor.parameters(), lr=learning_rate)
    optim_critic = optim.Adam(critic.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()
    steps_taken = 0
    logged_steps = set()
    returns = []

    while steps_taken < N_STEPS:
        batch_states, batch_actions, batch_qvals = [], [], []
        for _ in range(M):  # sample M trajectories
            s = environment.reset()[0]
            done = False
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
                logged_steps.add(step_bin)
                cumul_r = eval_rollout(environment, actor)
                returns.append(cumul_r)

            cur_state_tensor = torch.tensor(np.array(cur_states))
            for t in range(T):  # compute Q estimates for each step of the trajectory
                discounted_reward = get_reward(t, min(t+n, T), gamma, cur_rewards)
                if t+n < T:
                    qval = discounted_reward + (gamma**n) * critic(cur_state_tensor[t+n])
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
        # actor loss:
        optim_actor.zero_grad()
        # you need to detach qvals and bootstraps here because the actor part of the loss isn't supposed to update
        # the parameters of the critic network (according to the algorithm):
        actor_loss = -(qvals.detach() * log_action_probs).sum()
        # critic loss:
        optim_critic.zero_grad()
        critic_loss = mse_loss(qvals, bootstraps)
        # critic_loss = ((qvals - bootstraps)**2).sum()
        # backprop:
        (actor_loss + critic_loss).backward()
        optim_actor.step()
        optim_critic.step()

    return returns

def A2C(environment, actor, critic, learning_rate, gamma, M, n, TD_s_a=False):
    optim_actor = optim.Adam(actor.parameters(), lr=learning_rate)
    optim_critic = optim.Adam(critic.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()
    steps_taken = 0
    logged_steps = set()
    returns = []
    grad_vars = []
    grad_norms = []

    while steps_taken < N_STEPS:
        batch_states, batch_actions, batch_qvals = [], [], []

        for i in range(M):  # sample M trajectories
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
                logged_steps.add(step_bin)
                cumul_r = eval_rollout(environment, actor)
                returns.append(cumul_r)
                
                if TD_s_a and step_bin != 0:
                    grad_vars.append(np.std(grad_norms))
                    grad_norms.clear()

            cur_state_tensor = torch.tensor(np.array(cur_states))
            for t in range(T):  # compute Q estimates for each step of the trajectory
                discounted_reward = get_reward(t, min(t+n, T), gamma, cur_rewards)
                if t+n < T:
                    qval = discounted_reward + (gamma**n) * critic(cur_state_tensor[t+n])
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

        # logging for sensitivity analysis:
        grads = torch.cat([
            p.grad.view(-1)
            for p in actor.parameters()
            if p.grad is not None
        ])
        norm = grads.norm().item()
        grad_norms.append(norm)

    if TD_s_a:
        return returns, grad_vars
    else:
        return returns


def train(method:str, n_trials, return_dict):
    '''
        method needs to be one of: 'reinforce', 'ac', 'a2c'.
    '''
    env:gym.Env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    return_array = np.zeros((n_trials, N_STEPS // LOG_INTERVAL + 1))
    start = time.time()

    if method == 'reinforce':
        policy = createPolicy(state_dim, action_dim, 64, 1)
        for i in range(n_trials):
            returns = reinforce(
                env
                , policy
                , learning_rate=0.005
                , gamma=0.999
            )
            return_array[i] = returns
            print(f'REINFORCE trial {i+1} done.')
        print(f'REINFORCE training completed in {round((time.time()-start)/60, 1)} minutes.')
    
    elif method == 'ac':
        actor = createPolicy(state_dim, action_dim, 64, 1)
        critic = createPolicy(state_dim, 1, 64, 1)
        for i in range(n_trials):
            returns = AC(
                env
                , actor
                , critic
                , learning_rate=0.005
                , gamma=0.999
                , M=M
                , n=n
            )
            return_array[i] = returns
            print(f'AC trial {i+1} done.')
        print(f'AC training completed in {round((time.time()-start)/60, 1)} minutes.')

    elif method == 'a2c':
        actor = createPolicy(state_dim, action_dim, 64, 1)
        critic = createPolicy(state_dim, 1, 64, 1)
        for i in range(n_trials):
            returns = A2C(
                env
                , actor
                , critic
                , learning_rate=0.005
                , gamma=0.999
                , M=M
                , n=n
            )
            return_array[i] = returns
            print(f'A2C trial {i+1} done.')
        print(f'A2C training completed in {round((time.time()-start)/60, 1)} minutes.')
    
    else:
        raise RuntimeError(
            f"Unknown method '{method}' passed to train() function. Please pass one of 'reinforce', 'ac', 'a2c'."
        )
    
    env.close()

    mean_returns = np.mean(return_array, axis=0)
    stdev = np.std(return_array, axis=0)
    lower_stdev = mean_returns - stdev
    upper_stdev = mean_returns + stdev

    window = 10
    # moving average of mean returns:
    smoothed_mean = smooth(mean_returns, window)
    smoothed_lower = smooth(lower_stdev, window)
    smoothed_upper = smooth(upper_stdev, window)

    return_dict[method] = {'mean': smoothed_mean, 'lstd': smoothed_lower, 'ustd': smoothed_upper}


def train_A2C(n_boot, n_trials, return_dict):
    env:gym.Env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    return_array = np.zeros((n_trials, N_STEPS // LOG_INTERVAL + 1))
    grad_var_array = np.zeros((n_trials, N_STEPS // LOG_INTERVAL))
    start = time.time()
    actor = createPolicy(state_dim, action_dim, 64, 1)
    critic = createPolicy(state_dim, 1, 64, 1)
    for i in range(n_trials):
        returns, grad_vars = A2C(
            env
            , actor
            , critic
            , learning_rate=0.005
            , gamma=0.999
            , M=M
            , n=n_boot
            , TD_s_a=True
        )
        return_array[i] = returns
        grad_var_array[i] = grad_vars
        print(f'n = {n_boot} trial {i+1} done.')
    print(f'n = {n_boot} training completed in {round((time.time()-start)/60, 1)} minutes.')
    env.close()

    # process results
    mean_returns = np.mean(return_array, axis=0)
    stdev = np.std(return_array, axis=0)
    lower_stdev = mean_returns - stdev
    upper_stdev = mean_returns + stdev

    # sensitivity analysis:
    mean_grad_var = np.mean(grad_var_array, axis=0)

    window = 10
    # moving average of mean returns:
    smoothed_mean = smooth(mean_returns, window)
    smoothed_lower = smooth(lower_stdev, window)
    smoothed_upper = smooth(upper_stdev, window)
    smoothed_grad_var = smooth(mean_grad_var, window)

    return_dict[n_boot] = {
        'returns': {'mean': smoothed_mean, 'lstd': smoothed_lower, 'ustd': smoothed_upper}
        , 'sensitivity': {
            'variances': smoothed_grad_var
        }
    }


def reinforce_AC_A2C():
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    methods = ['reinforce', 'ac', 'a2c']
    print(f'Training started with n_steps = {N_STEPS}, n_trials = {N_TRIALS} for each algo, M = {M}, n = {n}, test run every {LOG_INTERVAL} steps.')
    for method in methods:
        print(f'{method} training started.')
        p = mp.Process(target=train, args=(method, N_TRIALS, return_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    x = np.linspace(0, N_STEPS, N_STEPS // LOG_INTERVAL + 1)
    methods = {
        'REINFORCE': {'returns': return_dict['reinforce'], 'colour': 'dodgerblue'}
        , 'AC': {'returns': return_dict['ac'], 'colour': 'firebrick'}
        , 'A2C': {'returns': return_dict['a2c'], 'colour': 'seagreen'}
    }

    plt.figure(figsize=(12,8))
    for key, val in methods.items():
        plt.plot(
            x, val['returns']['mean'], label=key, color=val['colour']
        )
        plt.fill_between(
            x, val['returns']['lstd'], val['returns']['ustd'], color=val['colour'], alpha=0.1
        )
    plt.ylim((0, 650))
    plt.xlabel('Million steps')
    plt.ylabel('Average return')
    plt.title(f'Average return over {N_STEPS} steps: REINFORCE, AC & A2C')
    plt.legend(loc='upper right')
    plt.savefig(f'final_hopefully.png')


def TD_sensitivity_analysis():

    print(f'Training started with n_steps = {N_STEPS}, n_trials = {N_TRIALS} for each n, M = {M}, test run every {LOG_INTERVAL} steps.')
    n_vals = [1, 5, 10, 25, 50, 100, 250, 500]
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    for n_val in n_vals:
        print(f'n = {n_val} training started.')
        p = mp.Process(target=train_A2C, args=(n_val, N_TRIALS, return_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    x = np.linspace(0, N_STEPS, N_STEPS // LOG_INTERVAL + 1)
    colours = ['sienna', 'fuchsia', 'mediumturquoise', 'navy', 'crimson', 'slategrey', 'goldenrod', 'mediumorchid']
    curves = {
        n: {'data': return_dict[n], 'colour': colours[i]}
        for i, n in enumerate(n_vals) if n in return_dict
    }
    # return curves:
    plt.figure(figsize=(12,8))
    for key, val in curves.items():
        plt.plot(
            x, val['data']['returns']['mean'], label=f'n = {key}', color=val['colour']
        )
        plt.fill_between(
            x, val['data']['returns']['lstd'], val['data']['returns']['ustd'], color=val['colour'], alpha=0.1
        )
    plt.ylim((0, 650))
    plt.xlabel('Million steps')
    plt.ylabel('Average return')
    plt.title(f'Average return over {'{:.0e}'.format(N_STEPS)} steps: n-step bootstrap sensitivity analysis')
    plt.legend(loc='upper right')
    plt.savefig(f'TD_sensitivity.png')

    # sensitivity analysis: variances
    x = np.linspace(LOG_INTERVAL, N_STEPS, N_STEPS // LOG_INTERVAL)
    plt.figure(figsize=(12,8))
    for key, val in curves.items():
        plt.plot(
            x, val['data']['sensitivity']['variances'], label=f'n = {key}', color=val['colour']
        )
    plt.xlim((0, N_STEPS))
    plt.xlabel('Million steps')
    plt.ylabel('Policy gradient variance')
    plt.title(f'Policy gradient variance over {'{:.0e}'.format(N_STEPS)} steps')
    plt.legend(loc='upper right')
    plt.savefig(f'variances.png')


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'q2.4':
            reinforce_AC_A2C()
        elif sys.argv[1] == 'sensitivity':
            TD_sensitivity_analysis()
        else:
            raise RuntimeError('Please pass either "q2.4" for REINFORCE vs AC vs A2C, or "sensitivity" to run the sensitivity analysis on the n parameter. ' \
            'Passing nothing will run question 2.4 REINFORCE vs AC vs A2C.')
    else:
        reinforce_AC_A2C()
