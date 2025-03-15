import gymnasium as gym
from stable_baselines3 import PPO

n_episodes = 100
env = gym.make('CartPole-v1', render_mode="rgb_array")

model = PPO('MlpPolicy', env, verbose=1, device='cpu')
print(model.policy)
model.learn(total_timesteps=10_000)

for n in range(n_episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action, _states = model.predict(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
        total_reward += reward
        steps += 1

    print(f'Episode {n + 1} finished with reward: {total_reward}; {steps} steps.')
