import gymnasium as gym
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


''' CONSTANTS '''
M = 1  # number of Monte Carlo trajectories to sample per batch
n = 250  # number of steps for bootstrapping


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
    probs = torch.softmax(policy(state),dim=-1)
    if greedy:
        return torch.argmax(probs).item()  # for testing
    else:
        return torch.multinomial(probs,1).item()


def get_reward(t, T, gamma, rewards):
    reward = 0
    for k in range(t,T):
        reward += rewards[k]*gamma**(k-t)
    return reward


def A2C(environment, actor, critic, learning_rate, gamma, M, n):
    optim_actor = optim.Adam(actor.parameters(), lr=learning_rate)
    optim_critic = optim.Adam(critic.parameters(), lr=learning_rate)
    return_history = deque(maxlen=100)  # to check for convergence
    done_episodes = 0
    steps_taken = 0

    while True:
        batch_states, batch_actions, batch_qvals = [], [], []
        for i in range(M):  # sample M trajectories
            s=environment.reset()[0]
            done=False
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
            cur_state_tensor = torch.tensor(np.array(cur_states))
            for t in range(T):  # compute Q estimates for each step of the trajectory
                discounted_reward = get_reward(t, min(t+n, T), gamma, cur_rewards)
                if t+n < T:
                    qval = discounted_reward + (gamma**n) * critic(cur_state_tensor[t+n])
                else:
                    qval = torch.tensor([discounted_reward])
                batch_qvals.append(qval)
            batch_states.extend(cur_states)
            # convergence check:
            done_episodes += 1
            ep_return = sum(cur_rewards)
            return_history.append(ep_return)
            steps_taken += ep_return
            if np.mean(return_history) > 450:
                print(f'Converged in {done_episodes} episodes, {int(steps_taken)} steps.')
                return actor
            # logging:
            if done_episodes%10 == 0:
                print(ep_return)
            
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
        actor_loss = -((qvals.detach() - bootstraps.detach()) * log_action_probs).sum()
        # critic loss:
        optim_critic.zero_grad()
        critic_loss = ((qvals - bootstraps)**2).sum()
        # backprop:
        (actor_loss + critic_loss).backward()
        optim_actor.step()
        optim_critic.step()


def main(M, n):
    env:gym.Env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    actor = createPolicy(state_dim, action_dim, 64, 1)
    critic = createPolicy(state_dim, 1, 64, 1)
    trained_actor = A2C(
        env
        , actor
        , critic
        , learning_rate=0.005
        , gamma=0.999
        , M=M
        , n=n
    )
    env.close()
    env:gym.Env=gym.make("CartPole-v1",render_mode="human")
    for _ in range(10):
        s=env.reset()[0]
        done=False
        c=0
        with torch.no_grad(): 
            while not done:
                a=action_selection(trained_actor, s, greedy=True)
                next_s, _, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                s = next_s
                c+=1
        print(f'Completed simulation in {c} steps.')
    env.close()
    
if __name__ == "__main__":
    main(M, n)
