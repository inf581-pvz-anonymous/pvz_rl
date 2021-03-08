from ... import config
from ..entity import Entity

RENDERING_OFFSET = 0.8

class Projectile(Entity):

    MAX_HP = 1

    def __init__(self, speed, lane, pos):
        super().__init__(lane)
        self._speed = speed
        self._pos = pos
        self._offset = 0. # Between 0 and 1
        
        # Used to find intermediary hits since we need the segment between two intermediary positions
        self._previous_pos = pos
        self._previous_offset = 0.

        self._render_start = True
        self._render_start_pos = pos

    def _move_one_step(self): # Move the projectile right for one frame
        self._previous_pos = self._pos
        self._previous_offset = self._offset
        
        self._offset += self._speed / config.FPS
        self._pos += int(self._offset)
        self._offset -= int(self._offset)

        if self._render_start:
            if (self._pos, self._offset) >= (self._render_start_pos, RENDERING_OFFSET):
                self._render_start = False

    def _is_out(self): # If it exited the grid
        return self._pos>=config.LANE_LENGTH

    def _hit(self, zombie): # If a zombie is on its path or not
        if zombie.lane == self.lane:
            if  (self._previous_pos, self._previous_offset * zombie.WALKING_SPEED * config.FPS) <= (zombie.pos, zombie._offset) and \
                (self._pos, self._offset * zombie.WALKING_SPEED * config.FPS) >= (zombie.pos, zombie._offset): # Lexicographic order
                return True
        return False
    
    def _attack_zombies(self, zombies): # What does the projectile do when it meets zombies that are in its path
        pass

    def step(self, scene):
        self._move_one_step()  # Projectile move
        zombies_hit = [] # Zombies that are in hit range
        for zombie in scene.zombies:
            if self._hit(zombie):
                zombies_hit.append(zombie)
        
        if zombies_hit:
            self._attack_zombies(zombies_hit)
        if self._is_out():
            self.hp = 0 # The projectile dies

    def _render(self):
        return not self._render_start

    