import pandas as pd
import numpy as np

def preprocess_dataframes(dfs):

    unique_user_ids = pd.concat([df['UserId'] for df in dfs]).unique()
    uncode_unique_user_ids = pd.Categorical(unique_user_ids).codes

    unique_location_ids = pd.concat([df['h3'] for df in dfs]).unique()
    uncode_unique_location_ids = pd.Categorical(unique_location_ids).codes



    for df in dfs:
        df['UserId'] = pd.Categorical(df['UserId'], categories=unique_user_ids)
        df['h3'] = pd.Categorical(df['h3'], categories=unique_location_ids)

    for df in dfs:
        df['UserId'] = df['UserId'].cat.codes
        df['h3'] = df['h3'].cat.codes
        df['UserId'] = pd.Categorical(df['UserId']).codes
        df['VenueType'] = pd.Categorical(df['VenueType']).codes
        df['day'] = pd.Categorical(df['day'], categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).codes

    return dfs, unique_user_ids, unique_location_ids
