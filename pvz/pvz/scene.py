# Time are in seconds

from . import config
from .grid import Grid
import numpy as np
from .entities.projectile.mower import Mower

class Scene:
    def __init__(self, plant_deck, zombie_spawner):
        self.plants = []
        self.zombies = []

        self.projectiles = []

        self.sun = config.INITIAL_SUN_AMOUNT

        self.plant_deck = plant_deck # Dictionnary linking names to classes containing the plants the player can use
        self.plant_cooldowns = {plant: 0 for plant in plant_deck} # All plants are available at the beginning of the level

        self.grid = Grid()

        self._zombie_spawner = zombie_spawner # Used to spawn the zombies we want for the level
        self._timer = config.NATURAL_SUN_PRODUCTION_COOLDOWN * config.FPS - 1 # Natural production of sun

        self._chrono = 0

        self.score = 0 # score
        self.lives = 1 # hp of the player (lives = 0: lossed battle)

        self._render_info = [{"zombies": [[] for _ in range(config.N_LANES)], "plants": [[] for _ in range(config.N_LANES)], 
                            "projectiles": [[] for _ in range(config.N_LANES)], "sun": self.sun,
                            "score": self.score, "cooldowns": {name: 0 for name in self.plant_cooldowns}, "time":0}]


    def step(self):
        for plant in self.plants:
            plant.step(self)
        for zombie in self.zombies:
            zombie.step(self)
        for projectile in self.projectiles:
            projectile.step(self)
        
        self._chrono += 1
        self.score = config.SURVIVAL * int((self._chrono + 1) % (config.FPS * config.SURVIVAL_STEP) == 0) + self.grid._mowers.sum()

        self._zombie_spawner.spawn(self)
        self._remove_dead_objects()
        self._timed_events()

        self._timer -= 1

        self._render_info.append(self._render_get_info())

    def add_zombie(self, zombie):
        self.zombies.append(zombie)
        self.grid.zombie_entrance(zombie.lane)

    def zombie_reach(self, lane):
        """ A zombie reached the end of a given lane """
        if self.grid.is_mower(lane):
            self.grid.remove_mower(lane)
            self.projectiles.append(Mower(lane))
        else:
            self.lives -= 1

    def _remove_dead_objects(self):
        alive_plants = []
        for plant in self.plants:
            if plant: # If alive
                alive_plants.append(plant)
                self.score += config.SCORE_ALIVE_PLANT
            else:
                self.grid.remove_obj(plant.lane, plant.pos)
        self.plants = alive_plants

        alive_zombies = []
        for zombie in self.zombies:
            if zombie: # If alive
                alive_zombies.append(zombie)
            else:
                self.grid.zombie_death(zombie.lane)
                self.score += zombie.SCORE
        self.zombies = alive_zombies

        alive_projectiles = []
        for projectile in self.projectiles:
            if projectile: # If alive
                alive_projectiles.append(projectile)
        self.projectiles = alive_projectiles

    def _timed_events(self):
        for plant in self.plant_cooldowns:
            self.plant_cooldowns[plant] = max(0, self.plant_cooldowns[plant] - 1)
        
        if self._timer <= 0:
            self.sun += config.NATURAL_SUN_PRODUCTION
            self._timer = config.NATURAL_SUN_PRODUCTION_COOLDOWN * config.FPS - 1
    
    def _render_get_info(self):
        info = {"zombies": [[] for _ in range(config.N_LANES)], "plants": [[] for _ in range(config.N_LANES)], 
                "projectiles": [[] for _ in range(config.N_LANES)], "sun": self.sun,
                "score": self.score, "cooldowns": {name: int(self.plant_cooldowns[name]/config.FPS)+1 for name in self.plant_cooldowns},
                "time": int(self._chrono/config.FPS)}
        for zombie in self.zombies:
            info["zombies"][zombie.lane].append((zombie.__class__.__name__, zombie.pos, zombie.get_offset()))
        for projectile in self.projectiles:
            if projectile._render():
                info["projectiles"][projectile.lane].append((projectile.__class__.__name__, projectile._pos, projectile._offset))
        for plant in self.plants:
            info["plants"][plant.lane].append((plant.__class__.__name__, plant.pos))
        return info

    def move_available(self): # Return true if a player can make a move
        if not self.grid.is_full():
            for plant_name in self.plant_deck:
                if (self.plant_cooldowns[plant_name]<=0) and (self.plant_deck[plant_name].COST <= self.sun):
                    return True
        return False

    def get_available_moves(self):
        empty_cells = self.grid.empty_cells()
        available_plants = [self.plant_deck[plant_name] for plant_name in self.plant_deck if (self.sun>=self.plant_deck[plant_name].COST) and (self.plant_cooldowns[plant_name] <= 0)]
        return (empty_cells, available_plants)

    def __str__(self):
        grid = np.full((config.N_LANES, config.LANE_LENGTH, 3), ['______', '_', '_'], dtype=object)
        for plant in self.plants:
            grid[plant.lane, plant.pos][0] = plant.__class__.__name__[0] + ":" + str(plant.hp).zfill(4)

        zombies_info = "\n"
        for zombie in self.zombies:
            zombies_info = zombies_info + str(zombie) + "\n"
            grid[zombie.lane, zombie.pos][2] = "Z"

        for projectile in self.projectiles:
            if projectile.__class__.__name__=="Mower":
                grid[projectile.lane, projectile._pos][1] =  "M"
            elif projectile.__class__.__name__=="Pea":
                grid[projectile.lane, projectile._pos][1] =  "o"

        grid_string = ""
        for lane in range(config.N_LANES):
            if self.grid.is_mower(lane):
                grid_string += "M"
            else:
                grid_string += "_"
            for pos in range(config.LANE_LENGTH):
                grid_string += " " + "_".join(grid[lane, pos]) + " "
            grid_string += "\n"


        return ("\nZombies" + zombies_info + "\nPlants :\n" + grid_string + "\nCooldowns:\n"
                    + str(self.plant_cooldowns) + "\nSun\n" + str(self.sun) + "\nLives" + str(self.lives) + "\nScore" + str(self.score) + "\nChrono" + str(self._chrono))