import gymnasium as gym
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random


def createPolicy(state_dim, action_dim, hidden_dim, n_hidden) -> nn.Sequential:
    layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
    for _ in range(n_hidden - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim, action_dim))
    return nn.Sequential(*layers)

def createValueNet(state_dim, hidden_dim, n_hidden) -> nn.Sequential:
    layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
    for _ in range(n_hidden - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim, 1)) 
    return nn.Sequential(*layers)

def action_selection(policy, state, greedy=False):
    state = torch.FloatTensor(state)
    probs = torch.softmax(policy(state), dim=-1)
    if greedy:
        return torch.argmax(probs).item()
    else:
        return torch.multinomial(probs, 1).item()

def compute_n_step_returns(rewards, states, value_net, gamma, n):
    """
    Returns a list (tensor) of n-step returns for each time-step in a single episode.
    """
    T = len(rewards)
    #convert states to tensors once
    state_tensors = [torch.FloatTensor(s) for s in states]
    
    #we'll store the n-step returns for each step
    returns_n = []
    
    for t in range(T):
        G = 0.0
        #Sum up to n rewards (or until end of episode)
        for k in range(n):
            if t + k < T:
                G += (gamma ** k) * rewards[t + k]
            else:
                break
        
        #if we haven't hit the end, add bootstrap from value_net(s_{t+n})
        if t + n < T:
            G += (gamma ** n) * value_net(state_tensors[t + n]).item()
        
        returns_n.append(G)
    
    return torch.tensor(returns_n, dtype=torch.float32)

def actor_critic(environment, policy, value_net, learning_rate, gamma,
                 n_step=20,       #for the n-step returns
                 max_episodes=10000,
                 batch_size=4): #how many episodes per update
    policy_optim = optim.Adam(policy.parameters(), lr=learning_rate)
    value_optim = optim.Adam(value_net.parameters(), lr=learning_rate)
    return_history = deque(maxlen=100)
    
    #buffers to accumulate multiple episodes
    all_states = []
    all_actions = []
    all_nstep_returns = []
    all_log_probs = []

    for episode in range(max_episodes):
        s = environment.reset()[0]
        done = False
        states, actions, rewards, log_probs = [], [], [], []

        while not done:
            a = action_selection(policy, s)
            s_tensor = torch.FloatTensor(s)
            
            probs = torch.softmax(policy(s_tensor), dim=0)
            dist  = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(torch.tensor(a))

            next_s, r, terminated, truncated, _ = environment.step(a)

            states.append(s)
            actions.append(a)
            rewards.append(r)
            log_probs.append(log_prob)

            s = next_s
            done = terminated or truncated

        # 1)Compute n-step returns for this entire episode:
        nstep_returns = compute_n_step_returns(rewards, states, value_net, gamma, n_step)
        
        # 2)Accumulate them into big buffers:
        all_states.extend(states)
        all_actions.extend(actions)
        all_nstep_returns.extend(nstep_returns.tolist())  
        all_log_probs.extend(log_probs)
        
        #logging
        episode_return = sum(rewards)
        return_history.append(episode_return)
        
        #if we've collected enough episodes (M = batch_size), do one update
        if (episode + 1) % batch_size == 0:
            #convert data to Tensors
            state_tensors = [torch.FloatTensor(s) for s in all_states]
            nstep_returns_tensor = torch.FloatTensor(all_nstep_returns)
            
            #evaluate value_net at each state
            values = torch.stack([value_net(s).squeeze() for s in state_tensors])
            
            #actor_loss = - sum( log_pi(a|s) * Q_hat_n(s,a) )
            #(We already have log_probs stored, and Q_hat_n from n-step returns)
            log_probs_tensor = torch.stack(all_log_probs)
            actor_loss = -torch.sum(log_probs_tensor * nstep_returns_tensor.detach())

            #critic loss = MSE( value(s), n-step returns )
            value_loss = F.mse_loss(values, nstep_returns_tensor)
            
            #Combined or separate updates
            policy_optim.zero_grad()
            value_optim.zero_grad()
            (actor_loss + value_loss).backward()
            policy_optim.step()
            value_optim.step()

            #clear buffers for the next batch
            all_states = []
            all_actions = []
            all_nstep_returns = []
            all_log_probs = []

        #check for convergence
        if episode % 10 == 0:
            avg100 = np.mean(return_history)
            print(f"Ep {episode}: Return={episode_return:.1f}, Avg100={avg100:.1f}")
            if avg100 > 450:
                print(f"Converged in {episode} episodes.")
                break
    
    return policy

def main():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = createPolicy(state_dim, action_dim, hidden_dim=64, n_hidden=1)
    value_net = createValueNet(state_dim, hidden_dim=64, n_hidden=1)

    trained_policy = actor_critic(env, policy_net, value_net, learning_rate=0.005, gamma=0.99)

    env.close()

    env = gym.make("CartPole-v1", render_mode="human")
    for _ in range(5):
        s = env.reset()[0]
        done = False
        c = 0
        while not done:
            a = action_selection(trained_policy, s, greedy=True)
            s, _, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            c += 1
        print(f"Test run completed in {c} steps.")
    env.close()

if __name__ == "__main__":
    main()