import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from dataloader import create_dataloader, split_data,read_files
from model import NeuralMatrixFactorization,train_model
from prepare_user import preprocess_dataframes
from logging_utils import load_config, logger
from evaluation import plot_precision_recall_curve, plot_roc_curve

config = load_config('config.ini')

print(config.sections())

data_name = config['Task']['data_name']
datapath = config['DataParameters']['data_folder_path']
out_path = config['DataParameters']['output_feather']
device = config['TrainingParameters']['device']
num_epochs = config.getint('TrainingParameters','num_epochs')
lr = config.getfloat('TrainingParameters','lr')
batch_size = config.getint('TrainingParameters','batch_size')
embedding_dim = config.getint('ModelParameters','embedding_dim')
hidden_dim = config.getint('ModelParameters','hidden_dim')
true_path = config['DataParameters']['true_path']

logger.info(f'num_epochs: {num_epochs}')
logger.info(f'lr: {lr}')
logger.info(f'batch_size: {batch_size}')
logger.info(f'embedding_dim: {embedding_dim}')
logger.info(f'hidden_dim: {hidden_dim}')
logger.info(f'device: {device}')
logger.info('Config loaded successfully')

_= split_data(450,datapath, data_name)
train_X, test_X = read_files(datapath, out_path, data_name)


processed_dfs, unique_user_ids, unique_location_ids = preprocess_dataframes([train_X, test_X])

train_X=processed_dfs[0]
test_X=processed_dfs[1]


train_data, test_data = train_test_split(train_X, test_size=0.3, random_state=42)


num_hours = 24  # Assuming hours range from 0 to 23
num_days = 7    # Assuming days range from 0 to 6
num_users = unique_user_ids.shape[0]
num_POI = unique_location_ids.shape[0]
num_types=4



model = NeuralMatrixFactorization(num_users, num_days, num_hours, embedding_dim, hidden_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_dataloader = create_dataloader(train_data,batch_size=batch_size)
logger.info('Training the model')
train_model(model, train_dataloader, optimizer, criterion, epochs=num_epochs,device=device)
logger.info('Training complete')

# Evaluate the model on the test set
test_dataloader = create_dataloader(test_data,batch_size=batch_size)
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for user_ids,  hours,days,move_distance,  ratings in test_dataloader:
        user_ids,  hours,days,move_distance,  ratings = user_ids.to(device),  hours.to(device),days.to(device),move_distance.to(device), ratings.to(device)
        outputs = model(user_ids,  hours,days,move_distance)
        _, predicted = torch.max(outputs.data, 1)
        total += ratings.size(0)
        correct += (predicted == ratings).sum().item()
    logger.info(f'Sample test ACC: {100 * correct / total}%')

torch.save(model, f'{data_name}_emb{embedding_dim}_ep{num_epochs}_batch{batch_size}.pth')



# Evaluate the model on the full test set

pred_loader = create_dataloader(test_X, batch_size=8192)
logger.info('Evaluating the model on the full test set')
test_dataloader = create_dataloader(test_X,batch_size=batch_size)
model.eval()
original_user_ids = []
original_POI_ids = []
model_outputs = []
original_ratings = []  
with torch.no_grad():
    correct = 0
    total = 0
    for user_ids,  hours,days,move_distance,  ratings in pred_loader:
        user_ids,  hours,days,move_distance,  ratings = user_ids.to(device),  hours.to(device),days.to(device),move_distance.to(device), ratings.to(device)
        outputs = model(user_ids,  hours,days,move_distance)
        _, predicted = torch.max(outputs.data, 1)
        original_user_ids.extend(user_ids.cpu().numpy())
        model_outputs.extend(predicted.cpu().numpy().ravel())
        original_ratings.extend(ratings.cpu().numpy())
        total += ratings.size(0)
        correct += (predicted == ratings).sum().item()
    logger.info(f'On FULL Testset ACC: {100 * correct / total}%')

pred_data = {
        'original_user_ids': original_user_ids,
        'model_outputs': model_outputs,
        "original_ratings": original_ratings
    }
pred_data = pd.DataFrame(pred_data)

df_filtered = pred_data[pred_data['model_outputs'] != pred_data['original_ratings']]
value_counts = df_filtered['original_user_ids'].value_counts()
value_counts_df = value_counts.reset_index()
value_counts_df.columns = ['original_user_ids', 'count']
gt= np.load(true_path)
gt= set(gt)
count_in_ccc = sum(value in gt for value in value_counts_df["original_user_ids"][:150])

logger.info(f'The top 150 hit: {count_in_ccc}')

logger.info('Evaluation complete')