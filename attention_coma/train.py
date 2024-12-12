import torch
import numpy as np
import os
import pygame as p
from config.config import Config
from gym_wrapper.environment import InfernoTamerGym
from attention_coma.coma import MultiAgentCOMA  # Import the COMA agents
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


def save_model(coma, episode, model_dir='models'):
    os.makedirs(model_dir, exist_ok=True)

    # Save the actor networks for each agent
    for i, actor in enumerate(coma.actors):
        actor_path = f"{model_dir}/agent_{i}_actor_episode_{episode}.pth"
        torch.save(actor.state_dict(), actor_path)

    # Save the centralized critic network
    critic_path = f"{model_dir}/centralized_critic_episode_{episode}.pth"
    torch.save(coma.critic.state_dict(), critic_path)


def train_coma(env, coma, num_episodes=1000, epsilon=0.1, save_interval=10, model_dir='models'):
    tracker = MetricsTracker()

    for episode in range(num_episodes):
        partial_states_ = []
        actions = []
        next_partial_states_ = []
        global_rewards = []
        next_global_states = []
        dones = []
        global_states = []

        # Reset environment and retrieve initial state
        env_state, state, partial_states = env.reset()
        episode_reward = 0
        episode_length = 0

        action_probs = np.zeros(Config.NUM_OF_PLANES, dtype=int)
        one_hot_actions = np.zeros((Config.NUM_OF_PLANES, Config.NUM_ACTIONS), dtype=int)

        for timestep in range(Config.MAX_MOVE):
            # Collect actions for all agents
            for agent_id in range(Config.NUM_OF_PLANES):
                # Get local state for the agent
                agent_state = torch.tensor(partial_states[agent_id], dtype=torch.float32).unsqueeze(0)
                agent_action_mask = MultiAgentCOMA.generate_action_mask(partial_states[agent_id])
                # Select an action using COMA's policy network
                with torch.no_grad():  # No gradients during action selection
                    action, action_prob = coma.select_action(agent_state, agent_id, agent_action_mask)

                # Convert action to one-hot encoding
                one_hot_action = np.zeros(Config.NUM_ACTIONS)
                one_hot_action[action] = 1

                # Save action data in the numpy arrays
                action_probs[agent_id] = action
                one_hot_actions[agent_id] = one_hot_action

            # Perform the environment step
            next_state_representation, next_state, next_partial_states, individual_reward, global_reward, done, _ = env.step_inference(env_state, one_hot_actions)

            # Update episode metrics
            episode_reward += global_reward
            episode_length += 1

            # Store transition data
            partial_states_.append(partial_states)
            actions.append(action_probs)
            global_rewards.append(global_reward)
            next_partial_states_.append(next_partial_states)
            dones.append(done)
            next_global_states.append(next_state)
            global_states.append(state)  # Centralized critic uses global state

            env_state = next_state_representation
            partial_states = next_partial_states

            if done:
                break

        # **Zip and shuffle transitions**
        transitions = list(zip(partial_states_, actions, next_partial_states_, next_global_states, global_rewards, global_states, dones))
        random.shuffle(transitions)  # Shuffle in place

        # **Unzip shuffled transitions**
        partial_states_, actions, next_partial_states_, next_global_states, global_rewards, global_states, dones = zip(*transitions)

        # Train the COMA algorithm
        policy_loss, value_loss = coma.train(
            global_states, partial_states_, actions, global_rewards, next_global_states, dones
        )

        # Update tracker with episode and loss metrics
        tracker.add_episode_metrics(episode_reward, episode_length)
        tracker.add_losses(policy_loss, value_loss, 0.0)

        # Save the models at the specified interval
        if episode % save_interval == 0:
            save_model(coma, episode, model_dir)

        # Print metrics every 10 episodes
        if episode % 10 == 0:
            print(
                f"Episode {episode}: Reward = {episode_reward}, Length = {episode_length}, Policy Loss = {policy_loss:.4f}, Value Loss = {value_loss:.4f}")

    # Plot metrics after training
    tracker.plot_metrics()


if __name__ == "__main__":
    p.init()
    env = InfernoTamerGym()
    env.make()
    coma_agent = MultiAgentCOMA(7, Config.NUM_ACTIONS, Config.NUM_OF_PLANES, Config.HIDDEN_DIM)
    train_coma(env, coma_agent)