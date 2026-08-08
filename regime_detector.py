import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RegimeDetectorCNN(nn.Module):
    """
    1D-CNN to classify market regimes based on a window of price/indicator ticks.
    Regimes: 0 = Trending Up, 1 = Trending Down, 2 = Ranging/Volatile, 3 = Quiet
    """
    def __init__(self, sequence_length=50, num_features=4, num_classes=4):
        super(RegimeDetectorCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Calculate flattened size
        flattened_size = (sequence_length // 4) * 32
        
        self.fc1 = nn.Linear(flattened_size, 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, num_features, sequence_length)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def train_regime_model(model_dir="model_registry"):
    os.makedirs(model_dir, exist_ok=True)
    
    # Mock data generation
    # In reality, we'd label windows of tick data (e.g., bid, ask, spread, volume) based on future returns or ATR.
    batch_size = 32
    seq_len = 50
    num_features = 4
    num_classes = 4
    
    model = RegimeDetectorCNN(sequence_length=seq_len, num_features=num_features, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training 1D-CNN Market Regime Detector (Mock Data)...")
    model.train()
    for epoch in range(10):
        # Generate random batch
        X = torch.randn(batch_size, num_features, seq_len)
        y = torch.randint(0, num_classes, (batch_size,))
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
    print(f"Final mock training loss: {loss.item():.4f}")
    
    # Save the model
    # Convert to ONNX for easy inference in C++ or other environments if needed
    model_path = os.path.join(model_dir, "regime_v1.onnx")
    dummy_input = torch.randn(1, num_features, seq_len)
    torch.onnx.export(model, dummy_input, model_path, 
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
                      
    print(f"Model exported to {model_path}")

def async_regime_loop(server_url="http://localhost:8001/params"):
    import requests
    import time
    
    print("Starting async regime detection loop...")
    while True:
        try:
            # Mock regime inference
            # 0 = Trending Up, 1 = Trending Down, 2 = Ranging, 3 = Quiet
            import random
            inferred_regime = random.randint(0, 3)
            
            # Send to SHM param server
            requests.put(server_url, json={"regime_id": inferred_regime})
        except Exception as e:
            print(f"Regime error: {e}")
            
        time.sleep(1.0) # 1s cadence for regime

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer"], default="train")
    args = parser.parse_args()
    
    if args.mode == "train":
        train_regime_model()
    else:
        async_regime_loop()
