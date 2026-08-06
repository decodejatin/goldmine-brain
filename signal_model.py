import os
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

def prepare_data(db_path='../build/trades.db'):
    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        # We need features and outcomes
        # A real implementation would join trades with a tick/indicator snapshot.
        # For now, we simulate features based on the existing trade log.
        df = pd.read_sql_query("SELECT * FROM trades", conn)
        conn.close()
    except Exception as e:
        print(f"Failed to load data: {e}")
        return None, None
        
    if len(df) < 50:
        print("Not enough data to train XGBoost signal model. Need at least 50 trades.")
        return None, None
        
    # Feature engineering (Mocked for demonstration, would use actual indicator snapshots)
    # df['conviction'] is already present
    # We will derive mock features from price/qty to make it run
    df['is_profitable'] = (df['pnl_usd'] - df['fees'] > 0).astype(int)
    
    # Mock features
    X = df[['conviction', 'qty', 'duration_ticks']].copy()
    
    # Add noise to simulate real indicator values if they are missing
    X['mock_rsi'] = np.random.uniform(30, 70, size=len(X))
    X['mock_zscore'] = np.random.uniform(-3, 3, size=len(X))
    X['mock_atr'] = np.random.uniform(0.5, 3.0, size=len(X))
    
    y = df['is_profitable']
    
    return X, y

def train_signal_model(model_dir="model_registry"):
    os.makedirs(model_dir, exist_ok=True)
    
    X, y = prepare_data()
    if X is None: return
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        max_depth=4,
        learning_rate=0.05,
        n_estimators=100
    )
    
    print("Training XGBoost Signal Confidence Model...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs)
    print(f"Validation Accuracy: {acc:.4f} | Log Loss: {ll:.4f}")
    
    model_path = os.path.join(model_dir, "signal_v1.json")
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

def predict_confidence(features, model_path="model_registry/signal_v1.json"):
    if not os.path.exists(model_path):
        return 0.5 # Default confidence if no model
        
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    # features should be a pandas DataFrame with matching columns
    probs = model.predict_proba(features)[:, 1]
    return probs

def async_inference_loop(model_path="model_registry/signal_v1.json", server_url="http://localhost:8001/params"):
    import requests
    import time
    from feature_pipeline import FeaturePipeline
    
    fp = FeaturePipeline()
    print("Starting async signal inference loop...")
    while True:
        try:
            # Mock getting live features
            features = fp.get_signal_features(current_conviction=0.6, current_qty=1.0)
            prob = predict_confidence(features, model_path)[0]
            
            # Map [0, 1] to bps (basis points) [0, 10000]
            prob_bps = int(prob * 10000)
            
            # Send to SHM param server
            requests.put(server_url, json={"p_profitable_gate_bps": prob_bps})
            
        except Exception as e:
            print(f"Inference error: {e}")
            
        time.sleep(0.1) # 100ms cadence

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer"], default="train")
    args = parser.parse_args()
    
    if args.mode == "train":
        train_signal_model()
    else:
        async_inference_loop()
