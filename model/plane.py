import math
import numpy as np
from model.fire import Fire


class Plane:
    def __init__(self, agent_id, stack, max_stack, x, y, level, grid_number, environment):
        self.agent_id = agent_id
        self.stack = stack
        self.min_stack = 1
        self.max_stack = max_stack
        self.x = x
        self.y = y
        self.reward = 0
        self.level = level
        self.grid_number = grid_number
        self.environment = environment
        self.type = ()

    def move(self, dx, dy):
        self.x += dx
        self.y += dy
        return self.x, self.y

    def distance_to(self, fire):
        """
        Calculate the Euclidean distance between the plane and a fire.
        :param fire: The Fire object to calculate the distance to.
        :return: The Euclidean distance between the plane and the fire.
        """
        dx = self.x - fire.x
        dy = self.y - fire.y
        return math.sqrt(dx ** 2 + dy ** 2)

    def get_available_moves(self):
        moves = []
        directions = [(-1, 1), (1, 1), (-1, -1), (1, -1), (-1, 0), (1, 0), (0, 1), (0, -1)]

        for dir_ in directions:
            target_row = self.x + dir_[0]
            target_column = self.y + dir_[1]

            if 0 <= target_row < self.grid_number and 0 <= target_column < self.grid_number:
                grid = self.environment[target_row][target_column]
                if grid is not None:
                    continue
                moves.append((target_row, target_column))
        return moves

    def extinguish_fire_check(self):
        moves = []
        directions = [(-1, 1), (1, 1), (-1, -1), (1, -1), (-1, 0), (1, 0), (0, 1), (0, -1)]

        for dir_ in directions:
            target_row = self.x + dir_[0]
            target_column = self.y + dir_[1]

            if 0 <= target_row < self.grid_number and 0 <= target_column < self.grid_number:
                grid = self.environment[target_row][target_column]
                if not isinstance(grid, Fire):
                    continue
                moves.append((dir_[0], dir_[1]))
        return moves

    def replenish_stack(self):
        if self.x in (0, self.grid_number - 1) and self.y in (0, self.grid_number - 1):
            self.stack = self.max_stack

    def extinguish_fire(self, fire):
        if self.stack >= self.level:
            self.stack -= self.level
            self.reward += fire.intensity
            fire.intensity -= self.level
            return fire.intensity <= 0
        return False

    def determine_type(self):
        if self.stack == 1:
            self.type = "plane", "low_stack"
        elif self.stack <= self.max_stack - 1:
            self.type = "plane", "moderate_stack"
        else:
            self.type = "plane", "high_stack"

    def get_type(self):
        return self.type

    def get_partial_state(self, environment, view_radius=6):
        """
        Create a partial state centered around the given plane.
        Parameters:
            plane: Plane object
            environment: Target environment
            view_radius: Radius of the neighborhood to consider for partial state
        Returns:
            A partial state array for the given plane.
        """
        # Get plane position
        plane_row, plane_col = self.x, self.y

        # Define the bounds of the local view
        row_start = max(0, plane_row - view_radius)
        row_end = min(self.grid_number, plane_row + view_radius + 1)
        col_start = max(0, plane_col - view_radius)
        col_end = min(self.grid_number, plane_col + view_radius + 1)

        # Initialize partial state with zeros
        partial_state = np.full((2 * view_radius + 1, 2 * view_radius + 1), -1)

        # Fill the partial state with data from the global state
        for row in range(row_start, row_end):
            for col in range(col_start, col_end):
                # Map global coordinates to partial state coordinates
                local_row = row - plane_row + view_radius
                local_col = col - plane_col + view_radius

                # Mark corners as 4 if visible
                if environment[row][col] is None and (row, col) in [(0, 0), (0, self.grid_number - 1), (self.grid_number - 1, 0),
                                  (self.grid_number - 1, self.grid_number - 1)]:
                    partial_state[local_row][local_col] = 4
                else:
                    cell = environment[row][col]
                    if cell is None:  # Empty cells or areas with no data
                        partial_state[local_row][local_col] = 0
                    elif isinstance(cell, Fire):
                        partial_state[local_row][local_col] = cell.intensity
                    elif isinstance(cell, Plane) and cell != self:  # Avoid marking the current plane
                        partial_state[local_row][local_col] = -1
                    elif isinstance(cell, Plane) and cell == self:  # Mark the current plane for action masking
                        partial_state[local_row][local_col] = -3 if self.stack <= self.min_stack else -2

        return partial_state

