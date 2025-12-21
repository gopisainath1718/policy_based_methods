import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from collections import deque
from networks.cartpole_mlp import Actor

class Reinforce():
    def __init__(self,
        env: gym.Env,
        n_training_episodes: int,
        max_steps: int,
        gamma: float,
        hidden_size: int, 
        lr: float,
    ):
        #initialize.
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        self.env = env
        self.n_training_episodes = n_training_episodes
        self.max_steps = max_steps
        self.gamma = gamma

        #device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #network
        self.actor = Actor(obs_dim, hidden_size, action_dim).to(self.device)

        #optimizer
        self.optimizer = optim.Adam(self.actor.parameters(), lr = lr)
        
        #train/test
        self.is_test = False

        #transition
        self.transition = []

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.actor(state)
        m = Categorical(probs)
        action = m.sample()
        
        if not self.is_test:
            self.transition = [state, m.log_prob(action)]
        return action.item()
    
    def step(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            self.transition.extend([next_state, reward, done])
        
        return next_state, reward, done

    def update_policy(self, transitions):
        returns = deque(maxlen = self.max_steps)
        discounted_return = 0
        for _, _, _, reward, done in transitions[::-1]:
            if done:
                discounted_return = 0
            discounted_return = reward + self.gamma * discounted_return
            returns.appendleft(discounted_return)
        
        eps = np.finfo(np.float32).eps.item()
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        log_probs = [t[1] for t in transitions]
        policy_loss = []
        for log_prob, disc_return in zip(log_probs, returns):
            policy_loss.append(disc_return * -log_prob)
        
        policy_loss = torch.cat(policy_loss).sum()
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        return policy_loss.item()

    def train(self):
        self.is_test = False
        scores = []
        policy_losses = []

        for episode in range(self.n_training_episodes):
            transitions = []
            score = 0
            state, _ = self.env.reset()

            for t in range(self.max_steps):
                action = self.get_action(state)
                next_state, reward, done = self.step(action)
                score += reward
                transitions.append(tuple(self.transition))
                state = next_state
                
                if done:
                    break
            policy_losses.append(self.update_policy(transitions))
            scores.append(score)

            if episode % 10 == 0:
                self._plot(episode, scores, policy_losses)

    def test(self, video_folder):
        self.is_test = True

        tmp_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

        state, _ = self.env.reset()
        done = False
        score = 0

        while not done:
            action = self.get_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        print("score: ", score)
        self.env.close()

        self.env = tmp_env

    def _plot(self, episode, scores, policy_losses):
        # reuse the same window named "Training" to prevent memory leaks
        plt.figure("Training", figsize=(12, 5))
        plt.clf()

        # --- Subplot 1: Scores ---
        plt.subplot(1, 2, 1)
        plt.title(f"Episode {episode} - Score")
        plt.plot(scores, label='Score', color='blue')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # --- Subplot 2: Losses ---
        plt.subplot(1, 2, 2)
        plt.title("Policy Loss")
        
        # Detach tensors if necessary so matplotlib can plot them
        clean_losses = [
            x.item() if isinstance(x, torch.Tensor) else x 
            for x in policy_losses
        ]
        
        plt.plot(clean_losses, label='Loss', color='red')
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Render
        plt.tight_layout()
        plt.pause(0.001)  # Brief pause to allow the plot to render
        
if __name__ == '__main__':

    env_id = "CartPole-v1"
    env = gym.make(env_id, render_mode="rgb_array")

    agent = Reinforce(env, n_training_episodes=10, max_steps=1000, hidden_size=16, gamma=1.0, lr=1e-2)
    agent.train()
    #agent.test("reinforce")
