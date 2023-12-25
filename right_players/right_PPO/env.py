import gymnasium as gym
import supersuit as ss
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env():
    '''
    Description:
        Creates a new environment with the following wrappers applied:
            - ss.color_reduction_v0(env, mode='B') # Reduces the color of frames to black and white
            - ss.resize_v1(env, x_size=84, y_size=84) # Resize the observation space to 84x84 
            - ss.frame_stack_v1(env, 4) # Stack 4 frames together 
            - ss.dtype_v0(env, dtype='float32') # Change the data type of observations to float32
            - ss.normalize_obs_v0(env, env_min=0, env_max=1) # Normalize the observation space to [0, 1]
            - DummyVecEnv([lambda: env]) # Vectorize the environment
    Args:
        None
    Returns:
        env: The wrapped environment
    '''
    
    # Load the base environment
    env = gym.make('ALE/Pong-v5')

    # Apply the specified wrappers
    env = ss.color_reduction_v0(env, mode='B')  # Reduces the color of frames to black and white
    env = ss.resize_v1(env, x_size=84, y_size=84)  # Resize the observation space
    env = ss.frame_stack_v1(env, 4)  # Stack 4 frames together
    env = ss.dtype_v0(env, dtype='float32')  # Change the data type of observations
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)  # Normalize the observation space
    env = DummyVecEnv([lambda: env]) # Vectorize the environment
    
    return env
