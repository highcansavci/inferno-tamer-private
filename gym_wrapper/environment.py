import copy
import time

import pygame as p
import sys

from config.config import Config
from controller.controller import Controller
from model.environment import Environment
from view.screen import Screen

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")



class InfernoTamerGym:
    def __init__(self):
        self.environment = None
        self.screen = None
        self.controller = None
        self.state_space = None
        self.action_space = None

    def make(self):
        self.environment = Environment(grid_number=Config.GRID_NUMBER,
                                       num_of_planes=Config.NUM_OF_PLANES,
                                       num_of_fires=Config.NUM_OF_FIRES,
                                       max_intensity=Config.MAX_INTENSITY,
                                       spread_timestep=Config.SPREAD_TIMESTEP,
                                       intensity_timestep=Config.INTENSITY_TIMESTEP,
                                       max_stack=Config.MAX_STACK)
        self.screen = Screen(environment=self.environment)
        self.controller = Controller(model=self.environment, view=self.screen)
        self.state_space = StateSpace(self.controller)
        self.action_space = ActionSpace(self.controller)

    def make_optimized(self):
        self.environment = Environment(grid_number=Config.GRID_NUMBER,
                                       num_of_planes=Config.NUM_OF_PLANES,
                                       num_of_fires=Config.NUM_OF_FIRES,
                                       max_intensity=Config.MAX_INTENSITY,
                                       spread_timestep=Config.SPREAD_TIMESTEP,
                                       intensity_timestep=Config.INTENSITY_TIMESTEP,
                                       max_stack=Config.MAX_STACK)
        self.controller = Controller(self.environment, None)
        self.state_space = StateSpace(self.controller)
        self.action_space = ActionSpace(self.controller)

    def copy(self, environment):
        self.environment = copy.deepcopy(environment)
        self.screen = Screen(self.environment)
        self.controller = Controller(self.environment, self.screen)
        self.state_space = StateSpace(self.controller)
        self.action_space = ActionSpace(self.controller)

    def copy_optimized(self, environment):
        self.environment = copy.deepcopy(environment)
        self.controller = Controller(self.environment, None)
        self.state_space = StateSpace(self.controller)
        self.action_space = ActionSpace(self.controller)

    def close(self):
        del self.action_space
        del self.state_space
        del self.controller
        del self.screen
        del self.environment
        self.state_space = None
        self.action_space = None
        self.controller = None
        self.screen = None
        self.environment = None

    def reset(self):
        return self.controller.reset()

    def reset_optimized(self):
        return self.controller.reset_optimized()

    def step(self, state, actions):
        return self.controller.step(state, actions)

    def step_inference(self, state, actions):
        return self.controller.step_inference(state, actions)

class ActionSpace:
    def __init__(self, controller):
        self.action_size = Config.NUM_ACTIONS
        self.controller = controller

    def sample(self):
        return self.controller.random_action()


class StateSpace:
    def __init__(self, controller):
        self.controller = controller
        self.shape = (Config.GRID_NUMBER, Config.GRID_NUMBER)

    def get_state(self):
        return self.controller.model.get_environment()


if __name__ == '__main__':
    p.init()
    env = InfernoTamerGym()
    env.make()
    state, _, _, _ = env.reset()
    env.screen.draw_game(state)
    score = 0
    done = False

    while not done:
        actions = env.action_space.sample()
        next_state_representation, communication, next_state, partial_states, individual_rewards, global_reward, done, _ = env.step_inference(state, actions)
        state = next_state_representation
        score += global_reward
        print(f"Score: {score}")
        time.sleep(1)

    env.close()