from ..entity import Entity
from ... import config

class Plant(Entity):

    COOLDOWN = None
    COST = None

    def __init__(self, lane, pos):
        """
        lane: lane where the plant is
        pos: position of the plant on the lane, 0 is on the left
        """

        super().__init__(lane)

        assert(pos>=0)
        assert(pos<config.LANE_LENGTH)
        self.pos = pos