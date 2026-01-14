"""
Neural Network model for chess position evaluation using PyTorch.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional
import os


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers."""
    
    def __init__(self, filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters)
    
    def forward(self, x):
        shortcut = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + shortcut
        x = F.relu(x)
        return x


class ChessNet(nn.Module):
    """Neural network architecture for chess."""
    
    def __init__(self, policy_size: int = 4096, num_residual_blocks: int = 10):
        super().__init__()
        
        # Initial convolution (input: 14 channels for piece planes)
        self.conv_input = nn.Conv2d(14, 256, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(256)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(256) for _ in range(num_residual_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(256, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, policy_size)
        
        # Value head
        self.value_conv = nn.Conv2d(256, 32, 1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        # Initial convolution
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = F.softmax(self.policy_fc(policy), dim=1)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value


class ChessNeuralNetwork:
    """Neural network for chess policy and value prediction."""
    
    # Policy output size: 64 * 64 = 4096 possible from-to combinations
    POLICY_SIZE = 4096
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the neural network.
        
        Args:
            model_path: Optional path to load pre-trained model.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = ChessNet(self.POLICY_SIZE).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Predict policy and value for a board state.
        
        Args:
            state: Board state tensor of shape (8, 8, 14).
            
        Returns:
            Tuple of (policy array, value float).
        """
        self.model.eval()
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            # Change from (H, W, C) to (C, H, W) format for PyTorch
            state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0)
            state_tensor = state_tensor.to(self.device)
            
            policy, value = self.model(state_tensor)
            
            return policy[0].cpu().numpy(), float(value[0][0].cpu())
    
    def train(self, states: np.ndarray, policies: np.ndarray, 
              values: np.ndarray, epochs: int = 10, 
              batch_size: int = 32) -> dict:
        """Train the neural network.
        
        Args:
            states: Array of board states (N, 8, 8, 14).
            policies: Array of target policies (N, 4096).
            values: Array of target values (N, 1).
            epochs: Number of training epochs.
            batch_size: Training batch size.
            
        Returns:
            Training history dictionary.
        """
        self.model.train()
        
        # Convert to PyTorch tensors (change to channels-first format)
        states_tensor = torch.FloatTensor(states).permute(0, 3, 1, 2)
        policies_tensor = torch.FloatTensor(policies)
        values_tensor = torch.FloatTensor(values)
        
        # Create dataset and dataloader
        dataset = TensorDataset(states_tensor, policies_tensor, values_tensor)
        
        # Split into train/validation (90/10)
        val_size = int(0.1 * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        history = {'loss': [], 'val_loss': [], 'policy_loss': [], 'value_loss': []}
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_policy_loss = 0.0
            train_value_loss = 0.0
            train_total_loss = 0.0
            
            for batch_states, batch_policies, batch_values in train_loader:
                batch_states = batch_states.to(self.device)
                batch_policies = batch_policies.to(self.device)
                batch_values = batch_values.to(self.device)
                
                self.optimizer.zero_grad()
                
                pred_policies, pred_values = self.model(batch_states)
                
                # Policy loss: cross-entropy
                policy_loss = -torch.sum(batch_policies * torch.log(pred_policies + 1e-8)) / batch_states.size(0)
                
                # Value loss: MSE
                value_loss = F.mse_loss(pred_values, batch_values)
                
                # Combined loss
                total_loss = policy_loss + value_loss
                
                total_loss.backward()
                self.optimizer.step()
                
                train_policy_loss += policy_loss.item()
                train_value_loss += value_loss.item()
                train_total_loss += total_loss.item()
            
            # Average training losses
            n_batches = len(train_loader)
            train_policy_loss /= n_batches
            train_value_loss /= n_batches
            train_total_loss /= n_batches
            
            # Validation phase
            self.model.eval()
            val_total_loss = 0.0
            
            with torch.no_grad():
                for batch_states, batch_policies, batch_values in val_loader:
                    batch_states = batch_states.to(self.device)
                    batch_policies = batch_policies.to(self.device)
                    batch_values = batch_values.to(self.device)
                    
                    pred_policies, pred_values = self.model(batch_states)
                    
                    policy_loss = -torch.sum(batch_policies * torch.log(pred_policies + 1e-8)) / batch_states.size(0)
                    value_loss = F.mse_loss(pred_values, batch_values)
                    
                    val_total_loss += (policy_loss + value_loss).item()
            
            val_total_loss /= len(val_loader) if len(val_loader) > 0 else 1
            
            # Record history
            history['loss'].append(train_total_loss)
            history['val_loss'].append(val_total_loss)
            history['policy_loss'].append(train_policy_loss)
            history['value_loss'].append(train_value_loss)
            
            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"loss: {train_total_loss:.4f} - "
                  f"policy_loss: {train_policy_loss:.4f} - "
                  f"value_loss: {train_value_loss:.4f} - "
                  f"val_loss: {val_total_loss:.4f}")
        
        return history
    
    def save(self, path: str) -> None:
        """Save model to file.
        
        Args:
            path: File path to save model.
        """
        # Change extension to .pt for PyTorch
        if path.endswith('.keras'):
            path = path.replace('.keras', '.pt')
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model from file.
        
        Args:
            path: File path to load model from.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")
    
    def summary(self) -> None:
        """Print model summary."""
        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
