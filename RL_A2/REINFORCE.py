import gymnasium as gym
import numpy as np
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def createPolicy(state_dim,action_dim,hidden_dim,n_hidden) -> torch.nn.Sequential:
    layers=[ nn.Linear(state_dim,hidden_dim),
    nn.ReLU()]
    for _ in range(n_hidden-1):
        layers.append(nn.Linear(hidden_dim,hidden_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim,action_dim))
    return nn.Sequential(*layers)

def action_selection(policy,state, greedy=False):
    state=torch.FloatTensor(state)
    probs = torch.softmax(policy(state),dim=-1)
    if greedy:
        return torch.argmax(probs).item()  # for testing
    else:
        return torch.multinomial(probs,1).item()

def get_reward(t,T,gamma,rewards):
    reward=0
    for k in range(t,T):
        reward+=rewards[k]*gamma**(k-t)
    return reward

def reinforce(environment,policy:nn.Sequential,learning_rate,epsilon,gamma):
    # gradient_magnitude=epsilon
    optimiser = optim.Adam(policy.parameters(), lr=learning_rate)
    return_history = deque(maxlen=100)  # to check for convergence
    counter = 0
    # while gradient_magnitude>=epsilon:
    while True:
        s=environment.reset()[0]
        done=False
        states=[]
        actions=[]
        rewards=[]
        while not done:
            a=action_selection(policy,s)
            s_prime, r, terminated, truncated, _ =environment.step(a)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            done=terminated or truncated
            s=s_prime
        T=len(rewards)
        episode_return = sum(rewards)
        return_history.append(episode_return)
        # logging:
        if counter%10 == 0:
            print(episode_return)
        if np.mean(return_history) > 450:
            print(f'Converged in {counter} episodes.')
            break
        # gradient ascent:
        losses = torch.zeros(T)
        for t in range(T):
            reward=get_reward(t,T,gamma,rewards)
            state_tensor = torch.FloatTensor(states[t])
            log_probs = F.log_softmax(policy(state_tensor), dim=0)
            act_prob = log_probs[actions[t]]
            losses[t] = -(gamma**t)*reward*act_prob
        loss = losses.mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        # gradient_magnitude=0
        # for param in policy.parameters():
        #     if param.grad is not None:
        #         gradient_magnitude+=torch.norm(param.grad).item()**2
        # gradient_magnitude=np.sqrt(gradient_magnitude)
        counter+=1
    return policy

def main():
    env:gym.Env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy=createPolicy(state_dim,action_dim,64,1)
    trained_policy=reinforce(env,policy,0.005,0.1,0.999)
    env.close()
    env:gym.Env=gym.make("CartPole-v1",render_mode="human")
    for _ in range(10):
        s=env.reset()[0]
        done=False
        c=0
        with torch.no_grad(): 
            while not done:
                a=action_selection(trained_policy,s, greedy=True)
                next_s, _, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                s = next_s
                c+=1
        print(f'Completed simulation in {c} steps.')
    env.close()
    
if __name__ == "__main__":
    main()