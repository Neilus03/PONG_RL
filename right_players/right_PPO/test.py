from stable_baselines3 import PPO
from env import make_env
from stable_baselines3.common.evaluation import evaluate_policy

env = make_env() 

model = PPO.load("ppo_pong_right_side", env=env)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10) # 10 episodes

# Enjoy trained agent
obs = env.reset()
model = PPO.load("ppo_pong_right_side", env=env, render_mode='human')
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)