import numpy as np
import random
import json


class DataProcessor:

    def __init__(self, raw_data, config, seed=42):

        self.seed = seed
        self.set_seed()
        self.config = self.load_config(config)

    def set_seed(self):
        np.random.seed(self.seed)
        random.seed(self.seed)

    def load_config(self, config_path):
        config_dict = json.loads(open(config_path, "r").read())
        return config_dict
