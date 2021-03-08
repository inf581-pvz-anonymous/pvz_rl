from .zombie_spawner import ZombieSpawner
from .zombie import Zombie
from ... import config
import random

INITIAL_OFFSET = 5
SPAWN_INTERVAL = 4

class BasicZombieSpawner(ZombieSpawner):

    def __init__(self):
        self._timer = INITIAL_OFFSET * config.FPS - 1
        
    def spawn(self, scene):
        if self._timer <= 0:
            lane = random.choice(range(config.N_LANES))
            scene.add_zombie(Zombie(lane))
            self._timer = SPAWN_INTERVAL * config.FPS - 1
        else:
            self._timer -= 1