import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
import pygame as p
from config.config import Config

class ImagePrototype:
    IMAGES = dict()
    IMAGES[("environment", "map")] = p.transform.scale(p.image.load("../assets/images/output_with_grid.png"), (Config.ENVIRONMENT_SIZE, Config.ENVIRONMENT_SIZE))
    IMAGES[("fire", "high_intensity")] = p.transform.scale(p.image.load("../assets/images/fire_high_intensity.png"), (Config.GRID_SIZE, Config.GRID_SIZE))
    IMAGES[("fire", "moderate_intensity")] = p.transform.scale(p.image.load("../assets/images/fire_moderate_intensity.png"), (Config.GRID_SIZE, Config.GRID_SIZE))
    IMAGES[("fire", "low_intensity")] = p.transform.scale(p.image.load("../assets/images/fire_low_intensity.png"), (Config.GRID_SIZE, Config.GRID_SIZE))
    IMAGES[("plane", "high_stack")] = p.transform.scale(p.image.load("../assets/images/plane_high_stack.png"), (Config.GRID_SIZE, Config.GRID_SIZE))
    IMAGES[("plane", "moderate_stack")] = p.transform.scale(p.image.load("../assets/images/plane_moderate_stack.png"), (Config.GRID_SIZE, Config.GRID_SIZE))
    IMAGES[("plane", "low_stack")] = p.transform.scale(p.image.load("../assets/images/plane_low_stack.png"), (Config.GRID_SIZE, Config.GRID_SIZE))

class Image:
    def __init__(self, type_, image_path):
        self.type_ = type_
        self.image_path = image_path

    def get_type(self):
        return self.type_

    def get_image(self):
        return ImagePrototype.IMAGES[self.type_]