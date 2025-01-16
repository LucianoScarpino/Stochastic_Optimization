import gymnasium as gym


class ShopFloor(gym.Env):
    def __init__(self, prb_instance):
        self.prb_instance = prb_instance

    def simulate_scheduling(self, schedule):
        obj_func = 0
        return obj_func