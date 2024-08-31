import os
import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm
import pyarrow.feather as feather
import h3
from torch.utils.data import DataLoader, TensorDataset
import torch
from logging_utils import logger, load_config
import numpy as np

def split_data(num_of_day,data_folder_path,data_name):
    if os.path.exists(data_folder_path+data_name+'-train.tsv') and os.path.exists(data_folder_path+data_name+'-test.tsv'):
        logger.info(f'{data_name} data already split')
        return
    else:
        logger.info(f'Splitting {data_name} data')
        data=pd.read_csv(data_folder_path+data_name+'.tsv', sep='\t')
        data['date'] = pd.to_datetime(data['CheckinTime']).dt.date
        start_date = data['date'].min()
        end_train_date = start_date + pd.Timedelta(days=num_of_day)
        train_data = data[data['date'] <= end_train_date]
        test_data = data[data['date'] > end_train_date]
        train_data.drop(columns=['date'], inplace=True)
        test_data.drop(columns=['date'], inplace=True)
        train_data.reset_index(drop=True, inplace=True)
        test_data.reset_index(drop=True, inplace=True)
        train_data.to_csv(data_folder_path+data_name+'-train.tsv', sep='\t')
        test_data.to_csv(data_folder_path+data_name+'-test.tsv', sep='\t')
        logger.info(f'{data_name} data split completed')
    return





def convertNOLA(latitude, longitude):
    from pyproj import Transformer
    transformer = Transformer.from_crs("epsg:26782", "epsg:4326")
    return transformer.transform(latitude, longitude)

def haversine(lat1, lon1, lat2, lon2):
    R = 63710000  # Earth radius in kilometers

    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c

def calculate_move_distance(df):
    df = df.sort_values(by=['UserId', 'CheckinTime']).reset_index(drop=True)

    df['prev_lat'] = df.groupby('UserId')['Latitude'].shift(1)
    df['prev_lon'] = df.groupby('UserId')['Longitude'].shift(1)
    
    df['move_distance_m'] = haversine(df['prev_lat'], df['prev_lon'], df['Latitude'], df['Longitude'])
    df['move_distance_m']= df['move_distance_m'].fillna(0)
    return df
def calculate_centroid(df):
    centroid_lat = df['Latitude'].mean()
    centroid_lon = df['Longitude'].mean()
    centroid_h3=h3.geo_to_h3(centroid_lat, centroid_lon,10)
    return centroid_h3

def read_files(data_folder_path , output_feather,data_name):
    if os.path.exists(output_feather+data_name+'-train-processed.feather') and os.path.exists(output_feather+data_name+'-train-processed.feather'):
        train_data = pd.read_feather(output_feather+data_name+'-train-processed.feather')
        test_data = pd.read_feather(output_feather+data_name+'-test-processed.feather')
        logger.info(f'{data_name} data loaded from feather file')

    else:
        logger.info(f'begin {data_name} data loaded from tsv file')

        train_data = pd.read_csv(data_folder_path+data_name+'-train.tsv', sep='\t')
        test_data = pd.read_csv(data_folder_path+data_name+'-test.tsv', sep='\t')
        
        train_data['Latitude'], train_data['Longitude'] = convertNOLA(train_data['X'], train_data['Y'])
        test_data['Latitude'], test_data['Longitude'] = convertNOLA(test_data['X'], test_data['Y'])
        
        train_data.drop(columns=['X', 'Y'], inplace=True)
        test_data.drop(columns=['X', 'Y'], inplace=True)
        
        train_data = calculate_move_distance(train_data)
        test_data = calculate_move_distance(test_data)
        
        train_data['day'] = pd.DatetimeIndex(train_data['CheckinTime']).day_name()
        train_data['hour'] = pd.DatetimeIndex(train_data['CheckinTime']).hour
        
        test_data['day'] = pd.DatetimeIndex(test_data['CheckinTime']).day_name()
        test_data['hour'] = pd.DatetimeIndex(test_data['CheckinTime']).hour
        
        train_data['duration'] = 1
        test_data["duration"]=1
        
        center_point=calculate_centroid(train_data)
        
        train_data['h3']=train_data.apply(lambda x: h3.geo_to_h3(x['Latitude'], x['Longitude'], 10), axis=1)
        test_data['h3']=test_data.apply(lambda x: h3.geo_to_h3(x['Latitude'], x['Longitude'], 10), axis=1)
        logger.info(f'{data_name} generated h3')

        train_data["h3_IJ"]= train_data["h3"].apply(lambda x: h3.experimental_h3_to_local_ij(center_point, x))
        test_data["h3_IJ"]= test_data["h3"].apply(lambda x: h3.experimental_h3_to_local_ij(center_point, x))
        logger.info(f'{data_name} generated IJ')

        
        train_data[['h3_I', 'h3_J']] = pd.DataFrame(train_data['h3_IJ'].tolist(), index=train_data.index)
        train_data.drop(columns=['h3_IJ'], inplace=True)
        
        test_data[['h3_I', 'h3_J']] = pd.DataFrame(test_data['h3_IJ'].tolist(), index=test_data.index)
        test_data.drop(columns=['h3_IJ'], inplace=True)
        
        if not os.path.exists(output_feather):
            os.makedirs(output_feather)


        feather.write_feather(train_data, output_feather+data_name+'-train-processed.feather')
        feather.write_feather(test_data, output_feather+data_name+'-test-processed.feather')
    
        logger.info(f'{data_name} data loaded from tsv file')
    return train_data, test_data






config = load_config('config.ini')
data_folder_path = config['DataParameters']['data_folder_path']
output_feather = config['DataParameters']['output_feather']
data_name = config['Task']['data_name']



def create_dataloader(data, batch_size=8192):
    user_ids = torch.tensor(data['UserId'].values, dtype=torch.long)
    hours = torch.tensor(data['hour'].values, dtype=torch.long)
    days = torch.tensor(data['day'].values, dtype=torch.long) 
    move_distance = torch.tensor(data['move_distance_m'].values, dtype=torch.float)
    ratings = torch.tensor(data['VenueType'].values, dtype=torch.long)
    dataset = TensorDataset(user_ids,  hours,days,move_distance,ratings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader