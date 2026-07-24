import os
import sys
import subprocess

import os
import sys
import subprocess
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Use pre-processed parquet file from Feature Pipeline
BASE_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "goldmine_features_1s.parquet")
if not os.path.exists(DATA_PATH):
    # Kaggle dataset path adjustment
    DATA_PATH = "../input/goldmine-features/goldmine_features_1s.parquet"


class GoldmineTradingEnv(gym.Env):
    """
    Custom Gym Environment for Goldmine HFT Parameter Optimization.
    The agent outputs the optimal risk parameters dynamically based on market state.
    """
    def __init__(self, df):
        super(GoldmineTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.current_step = 0
        self.max_steps = len(self.df) - 1
        
        # Observation space based on precomputed features
        # [rsi_14, atr_14, z_score_50, spread_bps]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        
        # Action space: The critical parameters for the C++ expert_engine
        # [risk_pct, tp_multiplier, sl_multiplier, z_score_threshold]
        # Scaled to [-1, 1] for PPO, then mapped to actual bounds in _map_actions
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        
        self.current_balance = 100000.0 # $100k starting capital
        
    def _map_actions(self, action):
        # Map [-1, 1] neural network outputs to physical trading boundaries
        risk_pct = np.clip((action[0] + 1) * 2.5, 0.1, 5.0) # 0.1% to 5.0%
        tp_mult = np.clip((action[1] + 1) * 2.5, 0.5, 5.0)  # 0.5 to 5.0
        sl_mult = np.clip((action[2] + 1) * 2.0, 0.5, 4.0)  # 0.5 to 4.0
        z_thresh = np.clip((action[3] + 1) * 1.5, 0.5, 3.0) # 0.5 to 3.0
        return risk_pct, tp_mult, sl_mult, z_thresh

    def step(self, action):
        risk_pct, tp_mult, sl_mult, z_thresh = self._map_actions(action)
        
        # Get current market state features
        row = self.df.iloc[self.current_step]
        z_score = row['z_score_50']
        atr = row['atr_14']
        spread = row['spread_bps'] / 10000.0
        price = row['close']
        
        reward = 0.0
        
        if abs(z_score) > z_thresh:
            # Trade triggered by the C++ engine simulation
            position_size_usd = self.current_balance * (risk_pct / 100.0)
            qty = position_size_usd / price
            
            # Simulate outcome (probabilistic fill based on spread and ATR)
            slippage = price * spread
            tp_dist = atr * tp_mult
            sl_dist = atr * sl_mult
            
            # Edge expectation based on z-score strength
            win_prob = 0.55 + (abs(z_score) - z_thresh) * 0.05
            win_prob = np.clip(win_prob, 0.1, 0.8)
            
            if np.random.rand() < win_prob:
                reward = (tp_dist - slippage) * qty
            else:
                reward = -(sl_dist + slippage) * qty
                
            self.current_balance += reward
            
        # Heavy penalty for total loss (Bankruptcy)
        if self.current_balance <= 0:
            reward = -100000
            done = True
        else:
            done = self.current_step >= self.max_steps
            
        self.current_step += 1
        
        if done:
            obs = np.zeros(4, dtype=np.float32)
        else:
            next_row = self.df.iloc[self.current_step]
            obs = np.array([
                next_row['rsi_14'],
                next_row['atr_14'],
                next_row['z_score_50'],
                next_row['spread_bps']
            ], dtype=np.float32)
            
        info = {
            "balance": self.current_balance,
            "risk_pct": risk_pct,
            "tp_mult": tp_mult,
            "sl_mult": sl_mult,
            "z_thresh": z_thresh
        }
        
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_balance = 100000.0
        row = self.df.iloc[0]
        obs = np.array([
            row['rsi_14'],
            row['atr_14'],
            row['z_score_50'],
            row['spread_bps']
        ], dtype=np.float32)
        return obs, {}

class CustomNetwork(BaseFeaturesExtractor):
    """
    Multiple deep layers to extract robust feature representations.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super(CustomNetwork, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(n_input_channels, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)

if __name__ == "__main__":
    if os.path.exists("/kaggle/working"):
        print("[*] Kaggle environment detected. Running training...")
        print("[*] Loading highly compressed Parquet feature dataset...")
        
        # Split train/eval
        df = pd.read_parquet(DATA_PATH)
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size]
        eval_df = df.iloc[train_size:]
        
        train_env = GoldmineTradingEnv(train_df)
        eval_env = GoldmineTradingEnv(eval_df)
        
        policy_kwargs = dict(
            features_extractor_class=CustomNetwork,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=[dict(pi=[128, 64], vf=[128, 64])] 
        )
        
        device = "cpu"
        print(f"[*] Initializing Multiple Layer PPO Model on {device.upper()}...")
        
        model = PPO("MlpPolicy", train_env, policy_kwargs=policy_kwargs, 
                    learning_rate=3e-4, n_steps=2048, batch_size=256,
                    n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, 
                    ent_coef=0.01, verbose=1, device=device)
        
        os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
        os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
        
        eval_callback = EvalCallback(eval_env, best_model_save_path=os.path.join(BASE_DIR, 'models'),
                                     log_path=os.path.join(BASE_DIR, 'logs'), eval_freq=10000,
                                     deterministic=True, render=False)
        
        print("[*] Starting 45-Minute Kaggle Training Loop (1.5 Million Timesteps)...")
        model.learn(total_timesteps=1500000, callback=eval_callback)
        
        print("[+] Training Complete. Saving final dynamic model weights...")
        model.save(os.path.join(BASE_DIR, "models", "goldmine_ppo_final"))
        
        obs, _ = eval_env.reset()
        action, _states = model.predict(obs, deterministic=True)
        risk, tp, sl, z = eval_env._map_actions(action)
        
        best_params = {
            "risk_pct": float(risk),
            "tp_multiplier": float(tp),
            "sl_multiplier": float(sl),
            "z_score_threshold": float(z)
        }
        
        with open(os.path.join(BASE_DIR, "models", "best_parameters.json"), "w") as f:
            json.dump(best_params, f, indent=4)
            
        print(f"[+] Optimal Baseline Parameters Discovered: {best_params}")
    else:
        import mmap
        import struct
        import time
        
        print("[*] Local environment detected. Starting Parameter Server...")
        param_path = os.path.join(BASE_DIR, "models", "models", "best_parameters.json")
        if not os.path.exists(param_path):
            param_path = os.path.join(BASE_DIR, "models", "best_parameters.json")
            
        if not os.path.exists(param_path):
            print("[-] best_parameters.json not found! Cannot start parameter server.")
            sys.exit(1)
            
        with open(param_path, "r") as f:
            params = json.load(f)
            
        print(f"[+] Loaded parameters: {params}")
        
        shm_path = "/dev/shm/goldmine_param_shm"
        if not os.path.exists(shm_path):
            with open(shm_path, "wb") as f:
                f.write(b'\x00' * 448)
                
        with open(shm_path, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 448, mmap.MAP_SHARED, mmap.PROT_WRITE)
            version = 1
            print("[+] Updating /dev/shm/goldmine_param_shm continuously...")
            try:
                while True:
                    mm.seek(0)
                    mm.write(struct.pack("Q", version))
                    mm.seek(64)
                    mm.write(struct.pack("d", params.get("risk_pct", 2.0)))
                    mm.seek(128)
                    mm.write(struct.pack("d", params.get("tp_multiplier", 0.35)))
                    mm.seek(192)
                    mm.write(struct.pack("d", params.get("sl_multiplier", 1.5)))
                    mm.seek(256)
                    mm.write(struct.pack("d", params.get("z_score_threshold", 0.5)))
                    mm.seek(320)
                    mm.write(struct.pack("I", 6000))
                    mm.seek(384)
                    mm.write(struct.pack("B", 0))
                    
                    version += 1
                    time.sleep(1.0)
            except KeyboardInterrupt:
                print("[*] Parameter Server stopped.")
