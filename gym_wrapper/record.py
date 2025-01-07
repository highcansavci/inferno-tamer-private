import time
from functools import partial

import vidmaker
import pygame as p
import sys

from gym_wrapper.environment import InfernoTamerGym

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")


class VideoRecorder:
    def __init__(self, env: InfernoTamerGym, video_path):
        self.env = env
        self.video_recorder = vidmaker.Video(video_path, late_export=True)

    def record(self):
        self.video_recorder.update(p.surfarray.pixels3d(self.env.screen.screen).swapaxes(0, 1), inverted=False)

    def export(self):
        self.video_recorder.export(verbose=True)


if __name__ == '__main__':
    p.init()
    env_ = InfernoTamerGym()
    record_env = VideoRecorder(env_, "random_action.mp4")
    env_.make()
    state, _, _, _ = env_.reset()
    env_.screen.draw_game(state)
    record_env.record()
    score = 0
    done = False

    while not done:
        actions = env_.action_space.sample()
        next_state_representation, communication, next_state, partial_states, individual_rewards, global_reward, done, _ = env_.step_inference(state, actions)
        state = next_state_representation
        score += global_reward
        print(f"Score: {score}")
        record_env.record()
        time.sleep(1)

    record_env.export()
    env_.close()