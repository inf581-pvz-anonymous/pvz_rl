from . import config

class Move:
    """A move consists in planting a plant on the grid"""
    
    def __init__(self, plant_name, lane, pos):
        self.plant_name = plant_name

        assert(lane>=0)
        assert(lane<config.N_LANES)
        assert(pos>=0)
        assert(pos<config.LANE_LENGTH)
        self.lane = lane
        self.pos = pos

    def is_valid(self, scene):
        assert (self.plant_name in scene.plant_deck) # Else the doesn't even exists
        
        valid = True
        valid &= scene.plant_cooldowns[self.plant_name] <= 0 # Cooldown refreshed
        valid &= scene.grid.is_empty(self.lane, self.pos) # Cell is empty of all plants
        valid &= scene.sun >= scene.plant_deck[self.plant_name].COST # Enough sun to buy

        return valid

    def apply_move(self, scene):
        scene.plants.append(scene.plant_deck[self.plant_name](self.lane, self.pos))
        scene.grid.add_obj(self.lane, self.pos)
        scene.plant_cooldowns[self.plant_name] = scene.plant_deck[self.plant_name].COOLDOWN * config.FPS - 1 # Reset cooldown
        scene.sun -= scene.plant_deck[self.plant_name].COST