import gymnasium as gym
from agents.reinforce import Reinforce

env_id = "CartPole-v1"
env = gym.make(env_id, render_mode="rgb_array")

agent = Reinforce(env, n_training_episodes=100, max_steps=1000, hidden_size=16, gamma=1.0, lr=1e-2)
agent.train()
#agent.test("envs/cartpole_trained_videos")

