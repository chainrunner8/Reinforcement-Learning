import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from classes import GridEnv, Agent


grid = GridEnv(3)
n_episodes = 200
n_runs = 100
learning_rates = [1, 1.1, 1.2, 1.5, 2] # [0.01, 0.05, 0.1, 0.5, 1]
df_sa_lr = pd.DataFrame()

for lr in learning_rates:

    n_runs_rewards = np.zeros((n_runs, n_episodes))

    for i in range(n_runs):

        agent = Agent(lr=lr, gamma=0.99, n_actions=4, eps_start=1, eps_end=0.01, eps_dec=0.995)
        run_rewards = []

        for j in range(n_episodes):

            state = grid.reset()
            done = False
            cumul_reward = 0

            while not done:
                action = agent.choose_actions(state)
                next_state, reward, done = grid.step(action)
                agent.learn(state, action, reward, next_state)
                state = next_state
                cumul_reward += reward
            
            run_rewards.append(cumul_reward)
            # print(f"Episode {j + 1}: epsilon{agent.epsilon:.3f}")
        n_runs_rewards[i] = run_rewards
        print(f"Run nb {i + 1} done")
    
    averaged_rewards = np.mean(n_runs_rewards, axis=0)
    df_sa_lr[f'lr={lr}'] = averaged_rewards
    # onto next lr...

print("Training complete!")


plt.figure(figsize=(12, 8))

for lr in learning_rates:
    plt.plot(
        list(range(1, n_episodes + 1)), 
        df_sa_lr[f'lr={lr}'],
        label=f'lr={lr}'
    )
plt.xlabel('Episodes')
plt.ylabel('Avg. cumulative reward')
plt.title('Sensitivity analysis of average cumulative reward by episode for different learning rates')
plt.legend()
plt.grid()

plt.show()

