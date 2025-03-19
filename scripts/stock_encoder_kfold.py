import os
import json
import torch
import yaml
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from contrastive_loss import ContrastiveLoss
from siamese_network import SiameseNetwork
from stock_pairs import StockPairs

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

# storage of performance metrics
os.makedirs("../performance_metrics/kfold", exist_ok=True)
os.makedirs("../plots/kfold", exist_ok=True)

# load data
train_dataset = StockPairs("../data/processed_data/training_pairs.csv")

# init K-fold cross-validation
kfold = KFold(n_splits=k_folds, shuffle=True)

# performance tracking
fold_metrics = []
all_train_losses = []
all_val_losses = []
all_train_negative_distances = []
all_train_positive_distances = []
all_val_negative_distances = []
all_val_positive_distances = []

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
    
    # fold performance tracking
    fold_train_losses = []
    fold_val_losses = []
    fold_train_negative_distances = []
    fold_train_positive_distances = []
    fold_val_negative_distances = []
    fold_val_positive_distances = []
    
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
        
        avg_train_loss = float(epoch_loss / len(train_loader))
        fold_train_losses.append(avg_train_loss)
        
        # average distances
        avg_train_negative_distance = float(sum(train_negative_distances) / len(train_negative_distances)) if train_negative_distances else 0.0
        avg_train_positive_distance = float(sum(train_positive_distances) / len(train_positive_distances)) if train_positive_distances else 0.0
        fold_train_negative_distances.append(avg_train_negative_distance)
        fold_train_positive_distances.append(avg_train_positive_distance)
        
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
        
        avg_val_loss = float(val_loss / len(val_loader))
        fold_val_losses.append(avg_val_loss)
        
        avg_val_negative_distance = float(sum(val_negative_distances) / len(val_negative_distances)) if val_negative_distances else 0.0
        avg_val_positive_distance = float(sum(val_positive_distances) / len(val_positive_distances)) if val_positive_distances else 0.0
        fold_val_negative_distances.append(avg_val_negative_distance)
        fold_val_positive_distances.append(avg_val_positive_distance)
        
        print(f"Fold {fold + 1} - Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.12f}, "
              f"Val Loss: {avg_val_loss:.12f}, "
              f"Train Avg Neg Distance: {avg_train_negative_distance:.6f}, "
              f"Train Avg Pos Distance: {avg_train_positive_distance:.6f}, "
              f"Val Avg Neg Distance: {avg_val_negative_distance:.6f}, "
              f"Val Avg Pos Distance: {avg_val_positive_distance:.6f}")
    
    # save fold metrics
    all_train_losses.append(fold_train_losses)
    all_val_losses.append(fold_val_losses)
    all_train_negative_distances.append(fold_train_negative_distances)
    all_train_positive_distances.append(fold_train_positive_distances)
    all_val_negative_distances.append(fold_val_negative_distances)
    all_val_positive_distances.append(fold_val_positive_distances)
    
    # plot and save fold results
    plt.figure()
    plt.plot(fold_train_losses, label='Training Loss')
    plt.plot(fold_val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Fold {fold + 1} Training and Validation Loss')
    plt.legend()
    plt.savefig(f'../plots/kfold/fold_{fold + 1}_loss_curve.png')
    plt.close()
    
    fold_metrics.append({
        "fold": fold + 1,
        "train_losses": fold_train_losses,
        "val_losses": fold_val_losses,
        "train_negative_distances": fold_train_negative_distances,
        "train_positive_distances": fold_train_positive_distances,
        "val_negative_distances": fold_val_negative_distances,
        "val_positive_distances": fold_val_positive_distances
    })

# average metrics across all folds
avg_train_losses = [sum(epoch_losses)/k_folds for epoch_losses in zip(*all_train_losses)]
avg_val_losses = [sum(epoch_losses)/k_folds for epoch_losses in zip(*all_val_losses)]
avg_train_negative_distances = [sum(epoch_distances)/k_folds for epoch_distances in zip(*all_train_negative_distances)]
avg_train_positive_distances = [sum(epoch_distances)/k_folds for epoch_distances in zip(*all_train_positive_distances)]
avg_val_negative_distances = [sum(epoch_distances)/k_folds for epoch_distances in zip(*all_val_negative_distances)]
avg_val_positive_distances = [sum(epoch_distances)/k_folds for epoch_distances in zip(*all_val_positive_distances)]

# plot average results
plt.figure()
plt.plot(avg_train_losses, label='Average Training Loss')
plt.plot(avg_val_losses, label='Average Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Average Training and Validation Loss')
plt.legend()
plt.savefig(f'../plots/kfold/average_loss_curve.png')
plt.close()

# save performance metrics
average_metrics = {
    "average_train_losses": avg_train_losses,
    "average_val_losses": avg_val_losses,
    "average_train_negative_distances": avg_train_negative_distances,
    "average_train_positive_distances": avg_train_positive_distances,
    "average_val_negative_distances": avg_val_negative_distances,
    "average_val_positive_distances": avg_val_positive_distances
}

metrics = {
    "fold_metrics": fold_metrics,
    "average_metrics": average_metrics
}

with open("../performance_metrics/kfold/cv_metrics.json", "w") as f: json.dump(metrics, f, indent=4)