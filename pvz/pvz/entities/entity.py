from .. import config

class Entity():

    MAX_HP = None

    def __init__(self, lane):
        self.hp = self.MAX_HP
        
        assert(lane>=0)
        assert(lane<config.N_LANES)
        self.lane = lane

    def step(self, scene):
        raise NotImplementedError

    def __bool__(self):
        return self.hp>0