import gym
import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class GroverSearchNetwork(BaseFeaturesExtractor):

	def __init__(self, observation_space: gym.spaces.MultiBinary, features_dim: int = 0):
		super().__init__(observation_space, features_dim)
		input_dim = observation_space.n
		self.main = nn.Sequential(
			nn.Linear(input_dim, features_dim),
			nn.ReLU(),
			nn.Linear(features_dim, features_dim),
			nn.Tanh()
		)
	
	def forward(self, observations: torch.Tensor):
		return self.main(observations)