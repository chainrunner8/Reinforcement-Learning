import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from classes import GridEnv, Agent


n_episodes = 500
n_runs = 100

learning_rates = [1.5, 2, 2.5] # [0.01, 0.05, 0.1, 0.5, 1]
epsilons = [0.02, 0.04, 0.07, 0.1, 0.3]
grid_sizes = [3, 5, 10, 15, 20]

df_sensitivity = pd.DataFrame()

# for lr in learning_rates:
for size in grid_sizes:

    grid = GridEnv(size)

    n_runs_rewards = np.zeros((n_runs, n_episodes))

    for i in range(n_runs):

        agent = Agent(lr=0.1, gamma=0.99, n_actions=4, eps_start=0.1, eps_end=0.05, eps_dec=0.999)
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

        # print(f"Q = {int(agent.Q.get(((2,1), 3), 0.0))}")
        n_runs_rewards[i] = run_rewards
        print(f"Run nb {i + 1} done")

    averaged_rewards = np.mean(n_runs_rewards, axis=0)
    df_sensitivity[f'size={size}x{size}'] = averaged_rewards
        # onto next eps...

print("Training complete!")


plt.figure(figsize=(12, 8))

for size in grid_sizes:
    plt.plot(
        list(range(1, n_episodes + 1)), 
        df_sensitivity[f'size={size}x{size}'],
        label=f'size={size}x{size}'
    )
plt.xlabel('Episodes')
plt.ylabel('Avg. cumulative reward')
plt.title('Sensitivity analysis of average cumulative reward by episode for different grid sizes')
plt.legend()
plt.grid()

plt.show()

