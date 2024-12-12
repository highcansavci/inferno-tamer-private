import random
import sys

from config.config import Config

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
import pygame as p
from model.environment import Environment


class Controller:
    def __init__(self, model: Environment, view):
        self.model = model
        self.view = view

    def reset(self):
        environment, state, partial_states = self.model.reset()
        self.model.move_count = 0
        self.view.draw_game(environment)
        return environment, state, partial_states

    def reset_optimized(self):
        environment, state, partial_states = self.model.reset()
        self.model.move_count = 0
        return environment, state, partial_states

    def step_inference(self, state, actions):
        next_state_representation, next_state, partial_states, individual_rewards, global_reward, done, _ = self.model.step(state, actions)
        self.view.draw_game(next_state_representation)
        self.view.clock.tick(Config.MAX_FPS)
        p.display.flip()
        return next_state_representation, next_state, partial_states, individual_rewards, global_reward, done, {}

    def step(self, state, actions):
        return self.model.step(state, actions)

    def random_action(self):
        """
        Generate random one-hot encoded actions for all planes.
        """
        actions = []
        for _ in self.model.get_all_planes():
            # Randomly choose an action (0 to 10)
            action_index = random.randint(0, Config.NUM_ACTIONS - 1)

            # Create one-hot encoding for the action
            one_hot_action = [0] * 10
            one_hot_action[action_index] = 1

            actions.append(one_hot_action)

        return actions
