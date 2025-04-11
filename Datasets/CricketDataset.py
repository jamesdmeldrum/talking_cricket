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
        """
        Loads and processes cricket match data from multiple CSV files.
        This function reads international and domestic cricket match data from 
        specified file paths, filters the data based on a date range, and combines 
        it with delivery-level data to create a comprehensive dataset.
        Returns:
            pd.DataFrame: A DataFrame containing combined match and delivery data 
            with specified columns.
        Raises:
            FileNotFoundError: If any of the specified CSV files are not found.
            KeyError: If required keys are missing in the configuration dictionary.
        Notes:
            - The function expects the configuration dictionary (`self.config`) to 
              contain the following keys:
                - `data.international_path`: Path to the international matches CSV file.
                - `data.domestic_path`: Path to the domestic matches CSV file.
                - `data.deliveries_path`: Directory path containing delivery data CSV files.
                - `data.start_date`: Start date for filtering matches (inclusive).
                - `data.end_date`: End date for filtering matches (inclusive).
                - `data.match_data_columns`: List of columns to extract from match data.
                - `data.delivery_data_columns`: List of columns to extract from delivery data.
            - Delivery data files are expected to be named using the match ID 
              (e.g., `<match_id>.csv`) and stored in the specified deliveries path.
        """

        # read match files
        international_matches = pd.read_csv(
            self.config["data"]["international_path"])
        domestic_matches = pd.read_csv(self.config["data"]["domestic_path"])

        # convert dates to datetime format
        international_matches['match_date'] = pd.to_datetime(
            international_matches['match_date'])
        domestic_matches['match_date'] = pd.to_datetime(
            domestic_matches['match_date'])

        start_date = datetime.strptime(
            self.config['data']['start_date'], '%Y/%m/%d')
        end_date = datetime.strptime(
            self.config['data']['end_date'], '%Y/%m/%d')

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

        overall_dataset = overall_dataset.sort_values(by=["match_id", "balls_remaining", "current_runs"], how=[
                                                      "ascending", "descending", "ascending"]).reset_index(drop=True)

        return overall_dataset

    def create_features_labels(self):
        """
        Generates a new DataFrame with features and labels for machine learning 
        by processing the existing cricket dataset.
        This function iterates through the dataset row by row and creates a new 
        DataFrame where each row represents a delivery with features from the 
        previous delivery, current delivery, and the outcome of the next delivery.
        The function also calculates and updates statistics for batters and bowlers 
        based on their performance in the dataset up to the current delivery.
        Returns:
            pd.DataFrame: A new DataFrame containing the processed features and labels.
        Helper Functions:
            fetch_bowler_stats(bowler_name, row_index):
                Fetches the most recent statistics (runs, balls, and wickets) 
                for a given bowler up to the specified row index.
        Notes:
            - The function assumes that the dataset (`self.data`) is sorted in 
              chronological order of deliveries.
            - The configuration for column names is provided in `self.config`.
        Raises:
            KeyError: If the required columns specified in `self.config` are 
                      not present in the dataset.
            IndexError: If the function attempts to access a row beyond the 
                        bounds of the dataset.
        Example:
            Assuming `self.data` is a DataFrame containing cricket match data 
            and `self.config` is a dictionary with the required column mappings:
            >>> features, labels = create_features_labels()
        """

        def fetch_bowler_stats(bowler_name, row_index):
            recent_dat = self.data.iloc[:row_index]

            bowler_data = recent_dat[recent_dat['bowler_on_strike_name']
                                     == bowler_name]

            if len(bowler_data) == 0:
                return 0, 0, 0
            else:
                most_recent_row = bowler_data.iloc[-1]
                return most_recent_row['bowler_on_strike_runs'], most_recent_row['bowler_on_strike_balls'], most_recent_row['bowler_on_strike_wickets']

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
                new_row['bowler_on_strike_runs'], new_row['bowler_on_strike_balls'], new_row['bowler_on_strike_wickets'] = fetch_bowler_stats(
                    next_delivery_bowler, index)

            for column_name in self.config['outcome_columns']:
                new_row[column_name] = self.data.iloc[index + 1][column_name]

            df = df.append(new_row, ignore_index=True)

        features = df.drop(columns=self.config['outcome_columns'])
        labels = df[self.config['outcome_columns']]

        return features, labels

    def __getitem__(self, idx):
        sequence_indices = self.sequences[idx]


if __name__ == "__main__":
    config_path = "../models/config_draft.json"

    CricketDataset(config_path)
