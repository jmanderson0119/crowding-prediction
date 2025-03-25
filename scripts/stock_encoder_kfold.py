import os
import torch
import yaml
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from contrastive_loss import ContrastiveLoss
from siamese_network import SiameseNetwork
from stock_pairs import StockPairs
from datetime import datetime

# load hyperparameters
with open("../encoder_hyperparameters.yaml", "r") as f: config = yaml.safe_load(f)

input_size = config["model"]["input_size"]
hidden_size = config["model"]["hidden_size"]
batch_first = config["model"]["batch_first"]
batch_size = config["training"]["batch_size"]
learning_rate = config["training"]["learning_rate"]
num_epochs = config["training"]["num_epochs"]
k_folds = config["training"]["folds"]
margin = config["training"]["margin"]

# load data
train_dataset = StockPairs("../data/processed_data/training_pairs.csv")

# init K-fold cross-validation
kfold = KFold(n_splits=k_folds, shuffle=True)

# create log directory
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f"../runs/kfold_experiment/{timestamp}"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

# train/val loop
for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):
    print(f"Fold {fold + 1} of {k_folds}")
    
    # init model, loss, and optimizer
    model = SiameseNetwork(input_size, hidden_size, batch_first)
    criterion = ContrastiveLoss(margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # move to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # init data loaders
    train_subsampler = Subset(train_dataset, train_ids)
    val_subsampler = Subset(train_dataset, val_ids)
    
    train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)
    
    # train loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        train_negative_distances = []
        train_positive_distances = []
        
        for batch_idx, (stock1_batch, stock2_batch, label_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            
            stock1_batch = stock1_batch.to(device)
            stock2_batch = stock2_batch.to(device)
            label_batch = label_batch.to(device)
            
            distances = model(stock1_batch, stock2_batch)
            loss = criterion(distances, label_batch)
            
            negative_mask = (label_batch == 1)
            train_negative_distances.extend(distances[negative_mask].cpu().detach().numpy())
            
            positive_mask = (label_batch == 0)
            train_positive_distances.extend(distances[positive_mask].cpu().detach().numpy())
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_negative_distance = np.mean(train_negative_distances) if train_negative_distances else 0.0
        avg_train_positive_distance = np.mean(train_positive_distances) if train_positive_distances else 0.0
        
        # val loop
        model.eval()
        val_loss = 0.0
        val_negative_distances = []
        val_positive_distances = []
        
        with torch.no_grad():
            for stock1_batch, stock2_batch, label_batch in val_loader:
                stock1_batch = stock1_batch.to(device)
                stock2_batch = stock2_batch.to(device)
                label_batch = label_batch.to(device)
                
                distances = model(stock1_batch, stock2_batch)
                loss = criterion(distances, label_batch)
                
                negative_mask = (label_batch == 1)
                val_negative_distances.extend(distances[negative_mask].cpu().detach().numpy())
                
                positive_mask = (label_batch == 0)
                val_positive_distances.extend(distances[positive_mask].cpu().detach().numpy())
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_negative_distance = np.mean(val_negative_distances) if val_negative_distances else 0.0
        avg_val_positive_distance = np.mean(val_positive_distances) if val_positive_distances else 0.0
        
        print(f"Fold {fold + 1} - Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.12f}, "
              f"Val Loss: {avg_val_loss:.12f}, "
              f"Train Avg Neg Distance: {avg_train_negative_distance:.6f}, "
              f"Train Avg Pos Distance: {avg_train_positive_distance:.6f}, "
              f"Val Avg Neg Distance: {avg_val_negative_distance:.6f}, "
              f"Val Avg Pos Distance: {avg_val_positive_distance:.6f}")
        
        # tensor board logs
        writer.add_scalar(f'Fold {fold + 1}/Train Loss', avg_train_loss, epoch)
        writer.add_scalar(f'Fold {fold + 1}/Val Loss', avg_val_loss, epoch)
        writer.add_scalar(f'Fold {fold + 1}/Train Avg Neg Distance', avg_train_negative_distance, epoch)
        writer.add_scalar(f'Fold {fold + 1}/Train Avg Pos Distance', avg_train_positive_distance, epoch)
        writer.add_scalar(f'Fold {fold + 1}/Val Avg Neg Distance', avg_val_negative_distance, epoch)
        writer.add_scalar(f'Fold {fold + 1}/Val Avg Pos Distance', avg_val_positive_distance, epoch)

writer.close()
