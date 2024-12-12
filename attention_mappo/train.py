import torch
import numpy as np
import os
import pygame as p
from config.config import Config
from gym_wrapper.environment import InfernoTamerGym
from attention_mappo.ppo import MAPPO
import random
import matplotlib.pyplot as plt


class MetricsTracker:
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []

    def add_episode_metrics(self, reward, length):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)

    def add_losses(self, policy_loss, value_loss, entropy_loss):
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropy_losses.append(entropy_loss)

    def plot_metrics(self):
        plt.figure(figsize=(14, 8))

        # Plot for Episode Rewards
        plt.subplot(2, 2, 1)
        plt.plot(self.episode_rewards, label="Episode Rewards")
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.title("Average Episode Rewards")
        plt.legend()

        # Plot for Episode Lengths
        plt.subplot(2, 2, 2)
        plt.plot(self.episode_lengths, label="Episode Lengths")
        plt.xlabel("Episodes")
        plt.ylabel("Timesteps")
        plt.title("Episode Lengths")
        plt.legend()

        # Plot for Policy and Value Losses
        plt.subplot(2, 2, 3)
        plt.plot(self.policy_losses, label="Policy Loss")
        plt.plot(self.value_losses, label="Value Loss")
        plt.xlabel("Episodes")
        plt.ylabel("Loss")
        plt.title("Policy and Value Loss")
        plt.legend()

        # Plot for Entropy Loss and Speed Loss
        plt.subplot(2, 2, 4)
        plt.plot(self.entropy_losses, label="Entropy Loss")
        plt.xlabel("Episodes")
        plt.ylabel("Loss")
        plt.title("Entropy Loss")
        plt.legend()

        # Adjust layout for better spacing
        plt.tight_layout()
        plt.show()

def save_model(ppo, episode, model_dir='models'):
    os.makedirs(model_dir, exist_ok=True)
    for i, (actor, critic) in enumerate(zip(ppo.actors, ppo.critics)):
        # Save the Actor network
        torch.save(actor.state_dict(), f"{model_dir}/agent_{i}_actor_episode_{episode}.pth")
        # Save the Critic network
        torch.save(critic.state_dict(), f"{model_dir}/agent_{i}_critic_episode_{episode}.pth")


def train_ppo_ctde(env, ppo, num_episodes=1000, epsilon=0.1, save_interval=10, model_dir='models'):
    tracker = MetricsTracker()

    for episode in range(num_episodes):
        states = []
        actions = []
        individual_rewards = []
        global_rewards = []
        next_states = []
        dones = []
        old_log_probs = []
        global_states = []

        # Reset environment and retrieve initial state
        env_state, state, partial_states = env.reset()
        episode_reward = 0
        episode_length = 0

        action_probs = np.zeros(Config.NUM_OF_PLANES, dtype=int)
        one_hot_actions = np.zeros((Config.NUM_OF_PLANES, Config.NUM_ACTIONS), dtype=int)
        log_probs = np.zeros(Config.NUM_OF_PLANES, dtype=float)

        for timestep in range(Config.MAX_MOVE):
            # Collect actions for all agents
            for agent_id in range(Config.NUM_OF_PLANES):
                # Get local state for the agent
                agent_state = torch.tensor(partial_states[agent_id], dtype=torch.float32).unsqueeze(0)

                # Get action probabilities from the target actor
                with torch.no_grad():  # Ensure no gradients are computed
                    action_prob = ppo.target_actors[agent_id](agent_state.unsqueeze(0))

                if np.random.rand() < epsilon:  # Add Dirichlet noise
                    dirichlet_alpha = 0.3  # Dirichlet parameter
                    dirichlet_noise = np.random.dirichlet([dirichlet_alpha] * Config.NUM_ACTIONS)
                    noisy_action_prob = action_prob.detach().cpu().numpy() + dirichlet_noise
                    noisy_action_prob = noisy_action_prob / noisy_action_prob.sum()  # Normalize
                    action_dist = torch.distributions.Categorical(probs=torch.tensor(noisy_action_prob))
                    action = action_dist.sample().item()
                else:
                    action_dist = torch.distributions.Categorical(probs=action_prob)
                    action = action_dist.sample().item()  # Exploration

                # Compute log probability of the selected action
                log_prob = torch.log(action_prob[0, action])

                # Convert action to one-hot encoding
                one_hot_action = np.zeros(action_prob.shape[1])
                one_hot_action[action] = 1

                # Save action data in the numpy arrays
                action_probs[agent_id] = action
                one_hot_actions[agent_id] = one_hot_action
                log_probs[agent_id] = log_prob.item()

            # Perform the environment step
            next_state_representation, next_state, next_partial_states, individual_reward, global_reward, done, _ = env.step_inference(env_state, one_hot_actions)

            # Update episode metrics
            episode_reward += global_reward
            episode_length += 1

            # Store transition data
            states.append(partial_states)
            actions.append(action_probs)
            individual_rewards.append(individual_reward)
            global_rewards.append(global_reward)
            next_states.append(next_state)
            dones.append(done)
            old_log_probs.append(log_probs)
            global_states.append(state)  # Centralized critic uses global state

            env_state = next_state_representation
            partial_states = next_partial_states

            if done:
                break

        # **Zip and shuffle transitions**
        transitions = list(zip(states, actions, individual_rewards, global_rewards, next_states, dones, old_log_probs, global_states))
        random.shuffle(transitions)  # Shuffle in place

        # **Unzip shuffled transitions**
        states, actions, individual_rewards, global_rewards, next_states, dones, old_log_probs, global_states = zip(*transitions)

        # Train the PPO algorithm with CTDE approach
        policy_loss, value_loss, entropy_loss = ppo.train(
            states, actions, individual_rewards, global_rewards, next_states, dones, old_log_probs, global_states
        )

        # Update tracker with episode and loss metrics
        tracker.add_episode_metrics(episode_reward, episode_length)
        tracker.add_losses(policy_loss, value_loss, entropy_loss)

        # Save the models at the specified interval
        if episode % save_interval == 0:
            save_model(ppo, episode, model_dir)

        # Print metrics every 10 episodes
        if episode % 10 == 0:
            print(
                f"Episode {episode}: Reward = {episode_reward}, Length = {episode_length}, Policy Loss = {policy_loss:.4f}, Value Loss = {value_loss:.4f}, Entropy Loss = {entropy_loss:.4f}")

    # Plot metrics after training
    tracker.plot_metrics()


if __name__ == "__main__":
    p.init()
    env = InfernoTamerGym()
    env.make()
    mappo = MAPPO(Config.GRID_NUMBER * Config.GRID_NUMBER, Config.NUM_ACTIONS, Config.HIDDEN_DIM, Config.NUM_OF_PLANES)
    train_ppo_ctde(env, mappo)