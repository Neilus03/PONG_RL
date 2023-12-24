
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

import config
from PPO_RIGHT_agent import CustomCNN
from utils import CustomRewardLogger
from env import make_env


# Check for CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env = make_env()


# Define the custom policy kwargs with the custom CNN and device
policy_kwargs = {
    'features_extractor_class': CustomCNN,
    'features_extractor_kwargs': {'features_dim': 512},
    'normalize_images': False
}


# Define the path to save the best model
save_path = "./models/pong_right"

# Define the TensorBoard log directory
tensorboard_log_dir = "./tensorboard_logs/pong_right"



if config.pretrained:
    # Load the pretrained agent
    model = PPO.load("./ppo_pong_right_side_v1.zip", env=env, device=device, tensorboard_log=tensorboard_log_dir)
else:
    # Instantiate the agent with TensorBoard logging
    model = PPO(
        "CnnPolicy",
        env,
        verbose=2,
        policy_kwargs=policy_kwargs,
        device=device,
        tensorboard_log=tensorboard_log_dir
    )

# Instantiate the callback
reward_logger = CustomRewardLogger()

# Train the agent with the custom callback
model.learn(total_timesteps=10000000, callback=reward_logger)
model.save("ppo_pong_right_side_v1")

