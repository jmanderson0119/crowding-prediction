import os
import torch
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from contrastive_loss import ContrastiveLoss
from siamese_network import SiameseNetwork
from stock_pairs import StockPairs
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# load hyperparameters
with open("../encoder_hyperparameters.yaml", "r") as f: config = yaml.safe_load(f)

input_size = config["model"]["input_size"]
hidden_size = config["model"]["hidden_size"]
batch_first = config["model"]["batch_first"]
batch_size = config["training"]["batch_size"]
learning_rate = config["training"]["learning_rate"]
num_epochs = config["training"]["num_epochs"]
margin = config["training"]["margin"]

# performance metrics paths
os.makedirs("../saved_models/", exist_ok=True)

# load dataset
train_dataset = StockPairs("../data/processed_data/training_pairs.csv")
test_dataset = StockPairs("../data/processed_data/test_pairs.csv")

# init model, loss, and optimizer
model = SiameseNetwork(input_size, hidden_size, batch_first)
criterion = ContrastiveLoss(margin)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# move to gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# init data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# performance tracking
train_losses = []
test_losses = []
train_negative_distances = []
train_positive_distances = []
test_negative_distances = []
test_positive_distances = []

# create log directory
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f"../runs/full_experiment/{timestamp}"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

# train/eval loop
for epoch in range(num_epochs):
    
    # train loop
    model.train()
    epoch_loss = 0.0
    train_negative_distances_epoch = []
    train_positive_distances_epoch = []
    
    for batch_idx, (stock1_batch, stock2_batch, label_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        
        stock1_batch = stock1_batch.to(device)
        stock2_batch = stock2_batch.to(device)
        label_batch = label_batch.to(device)

        distances = model(stock1_batch, stock2_batch)
        loss = criterion(distances, label_batch)
        
        negative_mask = (label_batch == 1)
        train_negative_distances_epoch.extend(distances[negative_mask].cpu().detach().numpy())
        
        positive_mask = (label_batch == 0)
        train_positive_distances_epoch.extend(distances[positive_mask].cpu().detach().numpy())
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_train_loss = float(epoch_loss / len(train_loader))
    train_losses.append(avg_train_loss)
    
    # average distances
    avg_train_negative_distance = float(sum(train_negative_distances_epoch) / len(train_negative_distances_epoch)) if train_negative_distances_epoch else 0.0
    avg_train_positive_distance = float(sum(train_positive_distances_epoch) / len(train_positive_distances_epoch)) if train_positive_distances_epoch else 0.0
    train_negative_distances.append(avg_train_negative_distance)
    train_positive_distances.append(avg_train_positive_distance)
    
    # val loop
    model.eval()
    test_loss = 0.0
    test_negative_distances_epoch = []
    test_positive_distances_epoch = []
    
    with torch.no_grad():
        for stock1_batch, stock2_batch, label_batch in test_loader:
            stock1_batch = stock1_batch.to(device)
            stock2_batch = stock2_batch.to(device)
            label_batch = label_batch.to(device)
            
            distances = model(stock1_batch, stock2_batch)
            loss = criterion(distances, label_batch)
            
            negative_mask = (label_batch == 1)
            test_negative_distances_epoch.extend(distances[negative_mask].cpu().detach().numpy())
            
            positive_mask = (label_batch == 0)
            test_positive_distances_epoch.extend(distances[positive_mask].cpu().detach().numpy())
            
            test_loss += loss.item()
    
    avg_test_loss = float(test_loss / len(test_loader))
    test_losses.append(avg_test_loss)
    
    avg_test_negative_distance = float(sum(test_negative_distances_epoch) / len(test_negative_distances_epoch)) if test_negative_distances_epoch else 0.0
    avg_test_positive_distance = float(sum(test_positive_distances_epoch) / len(test_positive_distances_epoch)) if test_positive_distances_epoch else 0.0
    test_negative_distances.append(avg_test_negative_distance)
    test_positive_distances.append(avg_test_positive_distance)
    
    print(f"Epoch {epoch + 1}/{num_epochs}, "
          f"Train Loss: {avg_train_loss:.12f}, "
          f"Test Loss: {avg_test_loss:.12f}, "
          f"Train Avg Neg Distance: {avg_train_negative_distance:.6f}, "
          f"Train Avg Pos Distance: {avg_train_positive_distance:.6f}, "
          f"Test Avg Neg Distance: {avg_test_negative_distance:.6f}, "
          f"Test Avg Pos Distance: {avg_test_positive_distance:.6f}")
    
    # tensor board logs
    writer.add_scalar('Train Loss', avg_train_loss, epoch)
    writer.add_scalar('Test Loss', avg_test_loss, epoch)
    writer.add_scalar('Train Avg Neg Distance', avg_train_negative_distance, epoch)
    writer.add_scalar('Train Avg Pos Distance', avg_train_positive_distance, epoch)
    writer.add_scalar('Test Avg Neg Distance', avg_test_negative_distance, epoch)
    writer.add_scalar('Test Avg Pos Distance', avg_test_positive_distance, epoch)

torch.save(model.state_dict(), log_dir)

writer.close()
