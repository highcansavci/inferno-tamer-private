import random
import sys


class Fire:
    def __init__(self, fire_id, intensity, max_intensity, intensity_timestep, spread_timestep, x, y, grid_number, environment):
        self.fire_id = fire_id
        self.intensity = intensity
        self.max_intensity = max_intensity
        self.intensity_timestep = intensity_timestep
        self.spread_timestep = spread_timestep
        self.spread_timer = self.spread_timestep
        self.intensity_timer = self.intensity_timestep
        self.x = x
        self.y = y
        self.grid_number = grid_number
        self.environment = environment
        self.type = ()

    def update_timers(self):
        self.spread_timer -= 1
        self.intensity_timer -= 1

    def reset_spread_timer(self, spread_interval):
        self.spread_timer = spread_interval

    def reset_intensity_timer(self, intensity_interval):
        self.intensity_timer = intensity_interval

    def increase_intensity(self):
        self.intensity += 1

    def spread_fire(self):
        emerged_fires = []
        x, y = self.x, self.y
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Check neighbors (up, down, left, right)
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_number and 0 <= ny < self.grid_number:  # Within bounds
                if self.environment[nx][ny] is None:  # Not already burning
                    fire_id = random.randint(0, sys.maxsize - 1)
                    fire = Fire(fire_id=fire_id,
                                spread_timestep=self.spread_timestep,
                                environment=self.environment,
                                intensity_timestep=self.intensity_timestep,
                                intensity=1,
                                max_intensity=self.max_intensity,
                                grid_number=self.grid_number,
                                x=nx,
                                y=ny)
                    self.environment[nx][ny] = fire
                    emerged_fires.append(fire)
        return emerged_fires

    def determine_type(self):
        if self.intensity == 1:
            self.type = "fire", "low_intensity"
        elif self.intensity <= self.max_intensity - 1:
            self.type = "fire", "moderate_intensity"
        else:
            self.type = "fire", "high_intensity"

    def get_type(self):
        return self.type