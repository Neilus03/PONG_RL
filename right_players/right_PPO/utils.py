
from stable_baselines3.common.callbacks import BaseCallback

class CustomRewardLogger(BaseCallback):
    '''
    Description:
        Custom callback for logging the total episode reward during training
    Args:
        verbose (int): Verbosity level 0: not output 1: info 2: debug
    Returns:
        None
    '''
    def __init__(self, verbose=2):
        super(CustomRewardLogger, self).__init__(verbose)
        self.episode_rewards = [] # Initialize the list of episode rewards
        self.total_reward = 0 # Initialize the total reward counter

    def _on_step(self) -> bool: #This is a callback function that is called at each step of the training process by the learn() function
        # Accumulate reward
        self.total_reward += self.locals['rewards'][0]  # The reward for the current step is stored in the 'rewards' key of the locals dictionary

        # Check if the episode is done
        if self.locals['dones'][0]: 
            # Log the total episode reward
            self.logger.record('reward/episode_reward', self.total_reward) # The logger object is stored in the 'logger' key of the locals dictionary
            self.episode_rewards.append(self.total_reward) # Append the total episode reward to the list of episode rewards
            # Reset the total reward for the next episode
            self.total_reward = 0  # Reset the total reward counter    
        return True # Return True to continue training the agent
