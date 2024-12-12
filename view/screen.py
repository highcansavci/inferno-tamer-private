import pygame as p
import pygame.display
import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")

from config.config import Config
from view.image_prototype import ImagePrototype


class Screen:
    def __init__(self, environment):
        self.model = environment
        self.screen = p.display.set_mode((Config.ENVIRONMENT_SIZE, Config.ENVIRONMENT_SIZE))
        self.clock = p.time.Clock()

    def draw_game(self, environment):
        self.draw_board(("environment", "map"))
        self.draw_images(environment)

    def draw_board(self, type_):
        image = ImagePrototype.IMAGES[type_]
        self.screen.blit(image, p.Rect(0, 0, Config.ENVIRONMENT_SIZE, Config.ENVIRONMENT_SIZE))

    def draw_images(self, environment):
        for i in range(Config.GRID_NUMBER):
            for j in range(Config.GRID_NUMBER):
                piece = environment[i][j]
                if piece is not None:
                    piece.determine_type()
                    image = ImagePrototype.IMAGES[piece.get_type()]
                    self.screen.blit(image, p.Rect(j * Config.GRID_SIZE, i * Config.GRID_SIZE,
                                                   Config.GRID_SIZE, Config.GRID_SIZE))
        pygame.display.update()