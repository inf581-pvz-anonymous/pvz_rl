from .zombie_spawner import ZombieSpawner
from .zombie import Zombie
from .zombie_cone import Zombie_cone
from .zombie_bucket import Zombie_bucket
from .zombie_flag import Zombie_flag
from ... import config
import random

INITIAL_OFFSET = 10
SPAWN_INTERVAL = 8

class WaveZombieSpawner(ZombieSpawner):

    def __init__(self):
        self._timer = INITIAL_OFFSET * config.FPS - 1
        self._wave_timer= 10*INITIAL_OFFSET * config.FPS - 1
        self.p=0.05
    def spawn(self, scene):
        if self._timer <= 0 and self._wave_timer>0 :
            lane = random.choice(range(config.N_LANES))
            s=random.random() 
            if(s<self.p):
                scene.add_zombie(Zombie_bucket(lane))
            elif(s<3*self.p):
                scene.add_zombie(Zombie_cone(lane))
            else:
                scene.add_zombie(Zombie(lane))
            self._timer = SPAWN_INTERVAL * config.FPS - 1
        else:
            if(self._wave_timer>0):
                self._timer -= 1
                self._wave_timer -=1
            else:
                scene.add_zombie(Zombie_flag(0))
                for lane in range(config.N_LANES):
                    s=random.random() 
                    if(s<self.p):
                        scene.add_zombie(Zombie_bucket(lane))
                    elif(s<3*self.p):
                        scene.add_zombie(Zombie_cone(lane))
                    else:
                        scene.add_zombie(Zombie(lane))
                self._wave_timer = 20 * SPAWN_INTERVAL * config.FPS - 1
                self._timer = 10 * INITIAL_OFFSET * config.FPS - 1
                self.p=min(self.p*2,1)