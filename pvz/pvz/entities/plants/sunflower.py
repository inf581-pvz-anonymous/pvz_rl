from .plant import Plant
from ... import config

SUNFLOWER_COST = 50
SUNFLOWER_COOLDOWN = 5
SUNFLOWER_MAX_HP = 300
SUNFLOWER_PRODUCTION = 25
SUNFLOWER_INITIAL_COOLDOWN = 5
SUNFLOWER_PRODUCTION_COOLDOWN = 5

class Sunflower(Plant):

    # Entity
    MAX_HP = SUNFLOWER_MAX_HP

    # Plant
    COOLDOWN = SUNFLOWER_COOLDOWN
    COST = SUNFLOWER_COST

    # Sunflower
    PRODUCTION = SUNFLOWER_PRODUCTION
    PRODUCTION_COOLDOWN = SUNFLOWER_PRODUCTION_COOLDOWN
    INITIAL_COOLDOWN = SUNFLOWER_INITIAL_COOLDOWN

    def __init__(self, lane, pos):
        super().__init__(lane, pos)
        self.production_cooldown = self.INITIAL_COOLDOWN * config.FPS - 1
    
    def step(self, scene):
        if self.production_cooldown <= 0:
            scene.sun += self.PRODUCTION
            self.production_cooldown = self.PRODUCTION_COOLDOWN * config.FPS - 1
        else:
            self.production_cooldown -= 1
