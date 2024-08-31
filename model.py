import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import pandas as pd
from logging_utils import load_config, logger

class NeuralMatrixFactorization(nn.Module):
    def __init__(self, num_users, num_days, num_hours, embedding_dim, hidden_dim):
        super(NeuralMatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.day_embedding = nn.Embedding(num_days, embedding_dim)
        self.hour_embedding = nn.Embedding(num_hours, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 3 + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 128)
        self.fc3 = nn.Linear(128, 4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self,user_ids, hours,days,move_distance):
        user_embedded = self.user_embedding(user_ids)
        day_embedded = self.day_embedding(days)
        hour_embedded = self.hour_embedding(hours)
        
        features = torch.cat([user_embedded, day_embedded, hour_embedded, move_distance.unsqueeze(1)], dim=1)
        x = self.relu(self.fc1(features))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    



def train_model(model, train_loader, optimizer, criterion, epochs=5,device='cpu'):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for user_ids,  hours,days,move_distance,  ratings in train_loader:
            user_ids,  hours,days,move_distance,  ratings = user_ids.to(device),  hours.to(device),days.to(device),move_distance.to(device),  ratings.to(device)
            optimizer.zero_grad()
            outputs = model(user_ids,  hours,days,move_distance)
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        #print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss/len(train_loader)}')
        logger.info(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}')



def test_model(model, test_dataloader, device='cpu'):
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
        logger.info(f'Accuracy on test data: {100 * correct / total}%')



