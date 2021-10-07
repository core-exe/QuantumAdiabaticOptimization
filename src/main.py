from environment import GroverSearchEnv
from model import GroverSearchNetwork
from stable_baselines3.td3 import TD3

env = GroverSearchEnv(n_qubit=5, cutoff=6, time_final=32, time_step=0.2)

policy_kwargs = dict(
	features_extractor_class=GroverSearchNetwork,
    features_extractor_kwargs=dict(features_dim=env.n_qubit)
)
model = TD3('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(10000)