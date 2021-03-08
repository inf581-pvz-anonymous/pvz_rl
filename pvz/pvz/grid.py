from . import config
import numpy as np

class Grid:

    def __init__(self):
        # Used to check the presence of plants
        self._grid = np.zeros((config.N_LANES, config.LANE_LENGTH), dtype = bool)
        
        # Number of zombies on the lane
        self._lanes = np.zeros(config.N_LANES, dtype = int)

        # Presence of mowers
        self._mowers = np.full(config.N_LANES, config.MOWERS, dtype = bool)

    def add_obj(self, lane, pos):
        self._grid[lane, pos] = True
    
    def remove_obj(self, lane, pos):
        self._grid[lane, pos] = False
    
    def is_empty(self, lane, pos):
        return not self._grid[lane,pos]
    
    def is_full(self):
        return not bool(len(np.nonzero(np.logical_not(self._grid))[0]))

    def empty_cells(self):
        return np.nonzero(np.logical_not(self._grid))

    def zombie_entrance(self, lane):
        self._lanes[lane] += 1

    def zombie_death(self, lane):
        self._lanes[lane] -= 1
    
    def is_attacked(self, lane):
        return bool(self._lanes[lane])

    def is_mower(self, lane):
        return self._mowers[lane]
    
    def remove_mower(self, lane):
        self._mowers[lane] = False