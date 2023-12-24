
from stable_baselines3.common.callbacks import BaseCallback

class CustomRewardLogger(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomRewardLogger, self).__init__(verbose)
        self.episode_rewards = []
        self.total_reward = 0

    def _on_step(self) -> bool:
        # Accumulate reward
        self.total_reward += self.locals['rewards'][0]

        # Check if the episode is done
        if self.locals['dones'][0]:
            # Log the total episode reward
            self.logger.record('reward/episode_reward', self.total_reward)
            self.episode_rewards.append(self.total_reward)
            # Reset the total reward for the next episode
            self.total_reward = 0
        return True
