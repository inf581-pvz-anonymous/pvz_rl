from ... import config
from .projectile import Projectile

class Pea(Projectile):

    def __init__(self, speed, attack, lane, pos):
        super().__init__(speed, lane, pos)

        self._attack = attack

    def _attack_zombies(self, zombies): # Pea hits the first zombie and dies
        # Zombie that is the most on the left
        zombie_hit = min(zombies, key = lambda z: (z.pos, z._offset)) 
        zombie_hit.hp -= self._attack
        self.hp = 0 # The projectile dies