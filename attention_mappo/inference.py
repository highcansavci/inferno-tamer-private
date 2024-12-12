import torch
import numpy as np
import pygame as p
from attention_mappo.ppo import RecurrentActor
from config.config import Config
from gym_wrapper.environment import InfernoTamerGym


def mappo_inference(env, actors, num_episodes=10):
    """
    Perform inference using a trained MAPPO model in the given environment.

    :param env: The multi-agent environment.
    :param mappo: The trained MAPPO instance containing actor networks.
    :param num_episodes: Number of episodes to run inference.
    """
    for episode in range(num_episodes):
        # Reset the environment
        env_state, global_state, partial_states = env.reset()
        done = False
        episode_reward = 0
        step_count = 0

        print(f"Starting Episode {episode + 1}")

        while not done:
            # Store actions for all agents
            actions = []
            one_hot_actions = np.zeros((Config.NUM_OF_PLANES, Config.NUM_ACTIONS), dtype=int)

            # Predict actions for all agents
            for agent_id in range(Config.NUM_OF_PLANES):
                # Get the local state of the agent
                agent_state = torch.tensor(partial_states[agent_id], dtype=torch.float32).unsqueeze(0)

                # Get action probabilities from the actor
                with torch.no_grad():
                    action_prob = actors[agent_id](agent_state.unsqueeze(0))

                # Apply action masking
                action_prob = action_prob / (action_prob.sum() + 1e-5)  # Normalize

                # Sample the action
                action_dist = torch.distributions.Categorical(probs=action_prob)
                action = action_dist.sample().item()

                # Convert action to one-hot encoding
                one_hot_action = np.zeros(action_prob.shape[1])
                one_hot_action[action] = 1

                # Save the action for the environment step
                actions.append(action)
                one_hot_actions[agent_id] = one_hot_action

            # Step in the environment
            next_state_representation, next_global_state, next_partial_states, reward, _, done, _ = env.step_inference(
                env_state, one_hot_actions
            )

            # Update episode metrics
            episode_reward += sum(reward)
            step_count += 1

            # Update the environment state
            env_state = next_state_representation
            partial_states = next_partial_states

        print(f"Episode {episode + 1} ended. Total Reward: {episode_reward}, Steps: {step_count}")


if __name__ == "__main__":
    actors = {}
    p.init()
    env = InfernoTamerGym()
    env.make()
    for i in range(Config.NUM_OF_PLANES):
        model = RecurrentActor(1, 16, 3, Config.NUM_ACTIONS, grid_number=7, num_heads=8)
        model.load_state_dict(torch.load(f"models/agent_{i}_actor_episode_990.pth", weights_only=True))
        actors[i] = model
    mappo_inference(env, actors)