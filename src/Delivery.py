import numpy as np
import random
import torch
import json


class Delivery:

    def __init__(self, data, config, seed=42):

        self.seed = seed
        self.set_seed()
        self.config = self.load_config(config)

    def set_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

    def load_config(self, config_path):
        config_dict = json.loads(open(config_path, "r").read())
        return config_dict
