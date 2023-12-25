from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch


class cUSTOMcNN(BaseFeaturesExtractor):
    '''
    Description:
        Custom CNN feature extractor for the PPO agent, we will make it more complex than sb3's original 
        CNN feature extractor by adding more convolutional layers
    Args:
        BaseFeaturesExtractor (class): The base feature extractor class
        observation_space (gym.spaces.Box): The observation space of the environment
        features_dim (int): The dimension of the extracted features
    Returns:
        None
    '''
    def __init__(self, observation_space, features_dim=512):
        super(cUSTOMcNN, self).__init__(observation_space, features_dim)
        
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 32, kernel_size=3, stride=1, padding=1), # 32x84x84
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 64x84x84
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 128x84x84
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # 256x84x84
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # 512x84x84 
            nn.ReLU(),
            nn.Flatten() # 512*84*84 = 3,612,672
        )
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1] # 512*84*84 = 3,612,672
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU()) # 512*84*84 -> 512 = 512 #This is the final feature map
    
    def forward(self, observations):
        return self.linear(self.cnn(observations))
