import torch
from torch.utils.data import Dataset
import json
import pandas as pd
import numpy as np
import random
from datetime import datetime

class CricketDataset(Dataset):
    def __init__(self, config_path, seed=0):
        self.seed = seed
        self.set_seed()

        self.config = self.load_config(config_path)

        self.data = self.load_data()

        self.sequnce_length = self.config['sequence_length']

        self.features, self.labels = self.create_features_labels()  # implement this

    def set_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config

    def load_data(self):

        # read match files
        international_matches = pd.read_csv(
            self.config["data"]["international_path"])
        domestic_matches = pd.read_csv(self.config["data"]["domestic_path"])

        # convert dates to datetime format
        international_matches['match_date'] = pd.to_datetime(international_matches['match_date'])
        domestic_matches['match_date'] = pd.to_datetime(domestic_matches['match_date'])

        start_date = datetime.strptime(self.config['data']['start_date'], '%Y/%m/%d')
        end_date = datetime.strptime(self.config['data']['end_date'], '%Y/%m/%d')

        # combine match files and enforce date restrictions
        all_matches = pd.concat(
            [international_matches, domestic_matches]).reset_index(drop=True)
        all_matches = all_matches[(all_matches['match_date'] >= start_date) & (
            all_matches['match_date'] <= end_date)]

        match_data_columns = self.config['data']['match_data_columns']
        delivery_data_columns = self.config['data']['delivery_data_columns']

        # create ball by ball dataset
        overall_dataset = pd.DataFrame(
            columns=match_data_columns + delivery_data_columns)

        for index, match in all_matches.iterrows():
            delivery_data = pd.read_csv(
                self.config['data']['deliveries_path'] + match['match_id'] + '.csv')

            # add match data to delivery data
            for column in match_data_columns:
                delivery_data[column] = match[column]

            delivery_data = delivery_data[match_data_columns +
                                          delivery_data_columns]

            overall_dataset = pd.concat([overall_dataset, delivery_data])

        overall_dataset = overall_dataset.reset_index(drop=True)

        return overall_dataset

    def create_features_labels(self):
        df = pd.DataFrame(columns=self.data.columns)

        for index, row in self.data.iterrows():
            new_row = {}
            for column_name in self.config['data']['previous_delivery_columns']:
                new_row[column_name] = row[column_name]
            for column_name in self.config['data']['current_delivery_columns']:
                new_row[column_name] = self.data.iloc[index + 1][column_name]

            next_delivery_striker = self.data.iloc[index +
                                                   1]['batter_on_strike_name']
            next_delivery_non_striker = self.data.iloc[index +
                                                       1]['batter_off_strike_name']
            next_delivery_bowler = self.data.iloc[index +
                                                  1]['bowler_on_strike_name']

            if next_delivery_striker == row['batter_on_strike_name']:
                new_row['batter_on_strike_runs'] = row['batter_on_strike_runs']
                new_row['batter_on_strike_balls'] = row['batter_on_strike_balls']
            elif next_delivery_striker == row['batter_off_strike_name']:
                new_row['batter_on_strike_runs'] = row['batter_off_strike_runs']
                new_row['batter_on_strike_balls'] = row['batter_off_strike_balls']
            else:
                new_row['batter_on_strike_runs'] = 0
                new_row['batter_on_strike_balls'] = 0

            if next_delivery_non_striker == row['batter_on_strike_name']:
                new_row['batter_off_strike_runs'] = row['batter_on_strike_runs']
                new_row['batter_off_strike_balls'] = row['batter_on_strike_balls']
            elif next_delivery_non_striker == row['batter_off_strike_name']:
                new_row['batter_off_strike_runs'] = row['batter_off_strike_runs']
                new_row['batter_off_strike_balls'] = row['batter_off_strike_balls']
            else:
                new_row['batter_off_strike_runs'] = 0
                new_row['batter_off_strike_balls'] = 0

            if next_delivery_bowler == row['bowler_on_strike_name']:
                new_row['bowler_on_strike_runs'] = row['bowler_on_strike_runs']
                new_row['bowler_on_strike_balls'] = row['bowler_on_strike_balls']
                new_row['bowler_on_strike_wickets'] = row['bowler_on_strike_wickets']
            else:
                pass

    def __getitem__(self, idx):
        sequence_indices = self.sequences[idx]

        


if __name__ == "__main__":
    config_path = "../models/config_draft.json"

    CricketDataset(config_path)