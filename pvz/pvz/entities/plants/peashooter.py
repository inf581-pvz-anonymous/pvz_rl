from .plant import Plant
from ..projectile.pea import Pea
from ... import config

PEASHOOTER_COST = 100
PEASHOOTER_COOLDOWN = 5
PEASHOOTER_MAX_HP = 300
PEASHOOTER_ATTACK = 20
PEASHOOTER_ATTACK_COOLDOWN = 1.5 # Seconds between attacks
PEASHOOTER_PROJECTILE_SPEED = 5 # Cells per second

class Peashooter(Plant):

    # Entity
    MAX_HP = PEASHOOTER_MAX_HP

    # Plant
    COOLDOWN = PEASHOOTER_COOLDOWN
    COST = PEASHOOTER_COST

    # Peashooter
    ATTACK = PEASHOOTER_ATTACK
    ATTACK_COOLDOWN = PEASHOOTER_ATTACK_COOLDOWN
    PROJECTILE_SPEED = PEASHOOTER_PROJECTILE_SPEED

    def __init__(self, lane, pos):
        super().__init__(lane, pos)
        self.attack_cooldown = self.ATTACK_COOLDOWN * config.FPS - 1
        self.projectiles = []

    
    def step(self, scene):
        if self.attack_cooldown <= 0:
            if scene.grid.is_attacked(self.lane):
                scene.projectiles.append(Pea(self.PROJECTILE_SPEED, self.ATTACK, self.lane, self.pos))
                self.attack_cooldown = self.ATTACK_COOLDOWN * config.FPS - 1
        else:
            self.attack_cooldown -= 1

    """def update_projectiles(self, scene):
        kept_projectiles = []
        for projectile in self.projectiles:
            projectile.step()
            alive = True
            for zombie in scene.zombies:
                if projectile.hit(zombie):
                    zombie.hp -= self.ATTACK
                    alive = False
                    break
            if alive and (not projectile.is_out()):
                kept_projectiles.append(projectile)
        self.projectiles = kept_projectiles"""
            

