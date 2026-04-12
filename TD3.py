import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import robosuite as suite

env = suite.make(
	"Lift",
	robots="Panda",
	has_renderer=True,
	has_offscreen_renderer=False,
	use_camera_obs=False,
	reward_shaping=True,
	control_freq=20,
)

obs = env.reset()
print()
print(env.action_dim)
print(obs)
#for _ in range(500):
#	action = env.action_spec[0] * 0
#	obs, reward, done, info = env.step(action)
#	env.render()
#
#env.close()
