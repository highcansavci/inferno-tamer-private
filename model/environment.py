import numpy as np
import random
import sys

from config.config import Config

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")

from model.fire import Fire
from model.plane import Plane


class Environment:
    def __init__(self, grid_number, num_of_planes, num_of_fires, max_intensity, spread_timestep, intensity_timestep, max_stack):
        assert grid_number > 0 and num_of_planes > 0 and num_of_fires > 0
        self.grid_number = grid_number
        self.num_of_planes = num_of_planes
        self.num_of_fires = num_of_fires
        self.max_intensity = max_intensity
        self.spread_timestep = spread_timestep
        self.intensity_timestep = intensity_timestep
        self.max_stack = max_stack
        self.environment = None
        self.fires = []
        self.steps_taken = 0
        self.total_reward = 0
        self.type = ("environment", "map")
        self.running = True
        self.move_count = 0

    def reset(self):
        self.steps_taken = 0
        self.total_reward = 0
        self.environment = [[None for _ in range(self.grid_number)] for _ in range(self.grid_number)]
        self.fires = []
        self.move_count = 0

        temp_num_planes = 0
        temp_num_fires = 0

        while temp_num_fires < self.num_of_fires or temp_num_planes < self.num_of_planes:
            # Place fires
            if temp_num_fires < self.num_of_fires:
                rand_x = random.randint(0, self.grid_number - 1)
                rand_y = random.randint(0, self.grid_number - 1)
                rand_intensity = random.randint(1, self.max_intensity)
                if self.environment[rand_x][rand_y] is None:
                    fire = Fire(
                        fire_id=temp_num_fires,
                        x=rand_x,
                        y=rand_y,
                        intensity=rand_intensity,
                        max_intensity=self.max_intensity,
                        spread_timestep=self.spread_timestep,
                        intensity_timestep=self.intensity_timestep,
                        grid_number=self.grid_number,
                        environment=self.environment
                    )
                    self.environment[rand_x][rand_y] = fire
                    self.fires.append(fire)
                    temp_num_fires += 1

            # Place planes
            if temp_num_planes < self.num_of_planes:
                rand_x = random.randint(0, self.grid_number - 1)
                rand_y = random.randint(0, self.grid_number - 1)
                rand_level = random.randint(1, self.max_stack)
                if self.environment[rand_x][rand_y] is None:
                    plane = Plane(
                        agent_id=temp_num_planes,
                        stack=self.max_stack,
                        max_stack=self.max_stack,
                        x=rand_x,
                        y=rand_y,
                        level=rand_level,
                        grid_number=self.grid_number,
                        environment=self.environment
                    )
                    self.environment[rand_x][rand_y] = plane
                    temp_num_planes += 1

        # Get partial states for all planes
        partial_states = []
        for plane in self.get_all_planes():
            partial_state = plane.get_partial_state(self.environment)  # Retrieve partial state for each plane
            partial_states.append(partial_state)
        # Concatenate partial states along a new axis
        partial_states = np.stack(partial_states,
                                  axis=0)  # Shape: (num_of_planes, view_radius * 2 + 1, view_radius * 2 + 1)

        return self.environment, self.compute_communication_matrix(self.environment), self.get_environment_state(), partial_states

    def update_fires(self):
        """Handle fire spread, intensity increase."""

        for fire in self.fires[:]:
            fire.update_timers()

            # Increase intensity if intensity timer expires
            if fire.intensity_timer <= 0:
                fire.increase_intensity()
                fire.reset_intensity_timer(self.intensity_timestep)

            # Spread fire if spread timer expires
            if fire.spread_timer <= 0:
                self.fires.extend(fire.spread_fire())
                fire.reset_spread_timer(self.spread_timestep)

    def update_planes(self, actions):
        """
        Update the state of each plane based on provided one-hot encoded actions.
        :param actions: Dictionary mapping plane IDs to one-hot encoded actions.
        """
        for plane in self.get_all_planes():
            action = actions[plane.agent_id]
            if sum(action) != 1:
                raise ValueError(f"Invalid action for plane {plane.agent_id}: {action}")

            action_index = list(action).index(1)  # Decode the one-hot encoding

            # Actions
            if 0 <= action_index <= 7:  # Movement actions
                directions = [
                    (-1, -1), (-1, 0), (-1, 1),  # Up-left, Up, Up-right
                    (0, -1), (0, 1),  # Left, Right
                    (1, -1), (1, 0), (1, 1)  # Down-left, Down, Down-right
                ]
                dx, dy = directions[action_index]

                new_x, new_y = plane.x + dx, plane.y + dy
                if (new_x, new_y) in plane.get_available_moves():
                    self.environment[plane.x][plane.y] = None  # Clear the current position
                    plane.move(dx, dy)
                    self.environment[plane.x][plane.y] = plane  # Update to the new position

            elif action_index == 8:  # Extinguish fire
                # Check for adjacent fires and extinguish if possible
                for direction in plane.extinguish_fire_check():
                    target_row, target_col = plane.x + direction[0], plane.y + direction[1]
                    if self.is_within_bounds(target_row, target_col):
                        fire = self.environment[target_row][target_col]
                        if isinstance(fire, Fire):
                            plane.extinguish_fire(fire)

            # Replenish water if at grid edge
            plane.replenish_stack()

    def step(self, state, actions):
        """
        Perform a single step in the environment based on the given state and actions.

        :param state: Current state of the environment.
        :param actions: List of one-hot encoded actions for all planes.
        :return: next_state_representation, next_state, partial_states, rewards, done, info
        """
        if len(actions) != self.num_of_planes:
            raise ValueError("Number of actions must match the number of planes.")

        # Set the environment state to the provided state
        self.set_environment_state(state)

        # Map actions to plane IDs
        actions_dict = {plane.agent_id: actions[i] for i, plane in enumerate(self.get_all_planes())}

        # Update planes based on actions
        self.update_planes(actions_dict)

        # Update fires
        self.update_fires()

        # Get the next state (global representation)
        next_state = self.get_environment_state()

        # Get partial states for all planes
        partial_states = []
        for plane in self.get_all_planes():
            partial_state = plane.get_partial_state(self.environment)  # Retrieve partial state for each plane
            partial_states.append(partial_state)
        # Concatenate partial states along a new axis
        partial_states = np.stack(partial_states,
                                  axis=0)  # Shape: (num_of_planes, view_radius * 2 + 1, view_radius * 2 + 1)

        # Calculate rewards
        individual_rewards, global_reward = self.calculate_rewards(actions_dict)

        for fire in self.fires:
            if fire.intensity <= 0:
                self.environment[fire.x][fire.y] = None
                self.fires.remove(fire)

        # Check if the episode is done
        done = (
                len(self.fires) == 0  # No fires left
                or self.move_count > Config.MAX_MOVE  # Maximum moves reached
                or all(
            self.environment[row][col] is not None
            for row in range(self.grid_number)
            for col in range(self.grid_number)
        )  # No empty cells left
        )

        # Increment the move count
        self.move_count += 1

        return self.environment, self.compute_communication_matrix(self.environment), next_state, partial_states, individual_rewards, global_reward, done, {}

    def calculate_rewards(self, actions_dict):
        """
        Reward shaping function for CTDE.
        :param actions_dict: Dictionary mapping plane IDs to one-hot encoded actions.
        :return: Dictionary mapping plane IDs to individual rewards and the global reward.
        """
        global_reward = 0
        extinguish_reward = 0
        individual_rewards = {plane.agent_id: 0 for plane in self.get_all_planes()}

        # Reward for extinguishing fires
        extinguished_fires = [fire for fire in self.fires if fire.intensity <= 0]
        for fire in extinguished_fires:
            extinguish_reward += fire.max_intensity * Config.EXTINGUISH_REWARD

        global_reward += extinguish_reward

        # Assign extinguishing reward based on proximity to extinguished fires
        for plane in self.get_all_planes():
            for fire in self.fires:
                if fire.intensity <= 0 and plane.distance_to(fire) < Config.PROXIMITY_THRESHOLD:
                    # Scale by proximity and total planes
                    proximity_scale = Config.EXTINGUISH_REWARD * fire.max_intensity / plane.distance_to(fire)
                    individual_rewards[plane.agent_id] += proximity_scale

        # Reward for moving closer to fires
        for plane in self.get_all_planes():
            closest_fire = min(self.fires, key=lambda fire: plane.distance_to(fire), default=None)
            if closest_fire:
                dist = plane.distance_to(closest_fire)
                # Reward scales inversely with distance
                proximity_reward = max(Config.CLOSE_TO_FIRE_REWARD / (dist + 1), Config.MIN_PROXIMITY_REWARD)
                individual_rewards[plane.agent_id] += proximity_reward
                global_reward += proximity_reward

        # Penalize planes running out of water
        for plane in self.get_all_planes():
            if plane.stack == 0:
                penalty = Config.EMPTY_STACK_PENALTY
                individual_rewards[plane.agent_id] -= penalty
                global_reward -= penalty / len(self.get_all_planes())

        # Penalize unnecessary moves (noop)
        noop_count = sum(1 for action in actions_dict.values() if list(action).index(1) == 9)
        noop_penalty = noop_count * Config.NOOP_PENALTY
        global_reward -= noop_penalty / len(self.get_all_planes())

        # Distribute noop penalty among planes that performed noop
        for plane_id, action in actions_dict.items():
            if list(action).index(1) == 9:
                individual_rewards[plane_id] -= Config.NOOP_PENALTY

        # Penalize longer move counts (speed penalty)
        speed_penalty = self.move_count * Config.SPEED_PENALTY_SCALE / len(self.get_all_planes())
        global_reward -= speed_penalty

        # Distribute speed penalty proportionally among planes
        for plane_id in individual_rewards.keys():
            individual_rewards[plane_id] -= speed_penalty / len(self.get_all_planes())

        individual_rewards = np.array([individual_rewards[agent_id] for agent_id in range(Config.NUM_OF_PLANES)])
        return individual_rewards, global_reward

    def set_environment_state(self, state):
        for row in range(self.grid_number):
            for col in range(self.grid_number):
                self.environment[row][col] = state[row][col]

    def get_all_planes(self):
        return [cell for row in self.environment for cell in row if isinstance(cell, Plane)]

    def get_environment_state(self):
        state = np.zeros((self.grid_number, self.grid_number))
        for row in range(self.grid_number):
            for col in range(self.grid_number):
                cell = self.environment[row][col]
                if isinstance(cell, Fire):
                    state[row][col] = cell.intensity
                elif isinstance(cell, Plane):
                    state[row][col] = -1
        return state

    def render(self):
        for row in self.environment:
            row_repr = ""
            for cell in row:
                if cell is None:
                    row_repr += " . "
                elif isinstance(cell, Fire):
                    row_repr += f" F{cell.intensity} "
                elif isinstance(cell, Plane):
                    row_repr += f" P{cell.agent_id} "
            print(row_repr)

    def is_within_bounds(self, row, col):
        return 0 <= row < self.grid_number and 0 <= col < self.grid_number

    def compute_communication_matrix(self, environment, view_radius=6):
        """
        Compute the communication matrix based on the agents' partial states.
        Parameters:
            environment: 2D grid environment
            view_radius: Radius of visibility for each agent
        Returns:
            A 2D communication matrix.
        """
        # Initialize a global visibility matrix with -4 (invisible)
        communication_matrix = np.full((Config.GRID_NUMBER, Config.GRID_NUMBER), -4)

        for plane in self.get_all_planes():
            # Get the agent's partial state
            partial_state = plane.get_partial_state(environment, view_radius)

            # Map the partial state back to the global grid
            plane_row, plane_col = plane.x, plane.y
            row_start = max(0, plane_row - view_radius)
            row_end = min(Config.GRID_NUMBER, plane_row + view_radius + 1)
            col_start = max(0, plane_col - view_radius)
            col_end = min(Config.GRID_NUMBER, plane_col + view_radius + 1)

            for local_row, row in enumerate(range(row_start, row_end)):
                for local_col, col in enumerate(range(col_start, col_end)):
                    if partial_state[local_row][local_col] != 0:
                        # Update the communication matrix with the visible state
                        communication_matrix[row][col] = partial_state[local_row][local_col]

        return communication_matrix