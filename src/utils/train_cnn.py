import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import sys

# Ensure the root directory is in the path
sys.path.append(os.getcwd())

from src.models.cnn_policy import TraceCNN

def accuracy(model, obs, actions):
    # return the fraction of the number of model's chosen matching the teacher's chosen action over total number of actions
    with torch.no_grad():
        pi_logits, _ = model(obs)

        chosen = torch.argmax(pi_logits, dim=1)

        correct = chosen == actions
        return correct.float().mean().item()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Load data
    data_path = "src/Training_Data/training_data_l1.pt"
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    print("Loading data...")
    episodes = torch.load(data_path, weights_only=False)
    random.seed(37)
    random.shuffle(episodes)

    # split the dataset into 80% of training data and 20% of validation data by determining the number of training episodes
    training_amount = int((len(episodes) * 80)/100)
    
    # Flatten episodes into a list of transitions, include only the first 80% of the episodes
    train_transitions = []
    for ep in episodes[:training_amount]:
        train_transitions.extend(ep)
        
    print(f"Total transitions for training loaded: {len(train_transitions)}")
    val_transitions = []
    for ep in episodes[training_amount:]:
        val_transitions.extend(ep)

    print(f"Total transitions for validations loaded: {len(val_transitions)}")

    # Prepare tensors
    # obs is already a tensor (9, 10, 10). We cat them along dim 0 to make a tensor (N, 9, 10, 10).
    obs_batch = torch.stack([t['obs'] for t in train_transitions], dim=0).to(device)
    val_obs = torch.stack([v['obs'] for v in val_transitions], dim=0).to(device)
    # target actions for cross entropy loss
    action_batch = torch.tensor([t['action'] for t in train_transitions], dtype=torch.long).to(device)
    val_actions = torch.tensor([v['action'] for v in val_transitions], dtype=torch.long).to(device)
    # target values for MSE loss
    value_batch = torch.tensor([t['value'] for t in train_transitions], dtype=torch.float32).unsqueeze(1).to(device)
    
    # Initialize Model, Loss, Optimizer
    model = TraceCNN(input_channels=9, num_actions=5).to(device)
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    batch_size = 128
    num_epochs = 30
    dataset_size = obs_batch.size(0)
    
    print("Starting training...")
    for epoch in range(num_epochs):
        # Shuffle indices
        indices = torch.randperm(dataset_size)
        
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_total_loss = 0.0
        
        for i in range(0, dataset_size, batch_size):
            batch_idx = indices[i:i+batch_size]
            
            b_obs = obs_batch[batch_idx]
            b_actions = action_batch[batch_idx]
            b_values = value_batch[batch_idx]
            
            # Forward pass
            pi_logits, v_pred = model(b_obs)
            
            # Compute losses
            loss_pi = policy_criterion(pi_logits, b_actions)
            loss_v = value_criterion(v_pred, b_values)
            
            # Combine loss (weight value loss by 0.5)
            loss = loss_pi + 0.5 * loss_v
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_policy_loss += loss_pi.item()
            epoch_value_loss += loss_v.item()
            epoch_total_loss += loss.item()
        train_acc = accuracy(model,obs_batch, action_batch)
        val_acc = accuracy(model, val_obs, val_actions)
            
        # Logging
        num_batches = (dataset_size + batch_size - 1) // batch_size
        print(f"Epoch {epoch+1}/{num_epochs} - "
            f"training accuracy: {train_acc:.3f}, validation accuracy: {val_acc:.3f}\n"
              f"Total Loss: {epoch_total_loss/num_batches:.4f}\n"
              f"(Policy: {epoch_policy_loss/num_batches:.4f}, Value: {epoch_value_loss/num_batches:.4f})\n")
        
    # Save the model
    os.makedirs("src/models", exist_ok=True)
    save_path = "src/models/validation_split_test.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")

if __name__ == "__main__":
    train()
