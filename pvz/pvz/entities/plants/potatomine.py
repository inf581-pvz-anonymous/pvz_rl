from .plant import Plant
from ... import config

POTATOMINE_COST = 25
POTATOMINE_COOLDOWN = 20
POTATOMINE_MAX_HP = 300
POTATOMINE_ATTACK_COOLDOWN = 14

class Potatomine(Plant):

    # Entity
    MAX_HP = POTATOMINE_MAX_HP

    # Plant
    COOLDOWN = POTATOMINE_COOLDOWN
    COST = POTATOMINE_COST

    #Potatomine
    ATTACK_COOLDOWN = POTATOMINE_ATTACK_COOLDOWN

    def __init__(self, lane, pos):
        super().__init__(lane, pos)
        self.attack_cooldown = self.ATTACK_COOLDOWN * config.FPS - 1
        self.used=0

    
    def step(self, scene):
        if self.attack_cooldown <= 0:
            for zombie in scene.zombies:
                if(zombie.pos==self.pos and zombie.lane==self.lane):
                    self.used=1
                    for zombie in scene.zombies:
                        if (zombie.pos==self.pos) and (zombie.lane==self.lane):
                            zombie.hp=0
                    break
            if(self.used==1):
                self.hp=0
        else:
            self.attack_cooldown -= 1
                    
           


