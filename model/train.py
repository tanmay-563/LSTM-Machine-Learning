#!/usr/bin/env python3
"""
model/train.py

Single-file trainer:
- Generates lightweight synthetic data if --data is not provided
- Builds sliding windows, trains LSTM autoencoder
- Saves model -> model/model.pt and scaler -> model/scaler.pkl

Usage:
    python model/train.py                # generate synthetic, train quick
    python model/train.py --data path/to/your.csv --epochs 8
"""
import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import joblib
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def generate_synthetic(npatients=50, minutes=600, anomaly_prob=0.001, out_csv=None):
    rows = []
    start = datetime.now() - timedelta(minutes=minutes)
    for pid in range(npatients):
        hr_base = 60 + np.random.randn()*5
        rr_base = 14 + np.random.randn()*2
        spo2_base = 98 + np.random.randn()*0.5
        t = 0
        while t < minutes:
            ts = start + timedelta(minutes=t)
            # small fluctuations
            hr = hr_base + np.random.randn()*2
            rr = rr_base + np.random.randn()*0.5
            spo2 = spo2_base + np.random.randn()*0.2
            # occasionally inject an anomalous block
            if np.random.rand() < anomaly_prob:
                dur = np.random.randint(3, 12)
                for dt in range(dur):
                    ts2 = start + timedelta(minutes=t+dt)
                    hr2 = hr + np.random.uniform(25, 60)
                    rr2 = rr + np.random.uniform(3, 8)
                    spo22 = spo2 - np.random.uniform(5, 12)
                    rows.append((pid, ts2.isoformat(), float(hr2), float(rr2), float(spo22), True))
                t += dur
            else:
                rows.append((pid, ts.isoformat(), float(hr), float(rr), float(spo2), False))
                t += 1
    df = pd.DataFrame(rows, columns=['patient_id','timestamp','hr','rr','spo2','is_anomaly'])
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df

def build_windows(df, window=30):
    windows=[]
    labels=[]
    for pid, g in df.groupby('patient_id'):
        g = g.sort_values('timestamp')
        arr = g[['hr','rr','spo2']].values.astype(np.float32)
        an = g['is_anomaly'].values.astype(bool)
        if len(arr) >= window:
            for i in range(len(arr)-window+1):
                windows.append(arr[i:i+window])
                labels.append(int(an[i:i+window].any()))
    return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.int32)

class WindowDataset(Dataset):
    def __init__(self, arr): self.arr = torch.tensor(arr, dtype=torch.float32)
    def __len__(self): return len(self.arr)
    def __getitem__(self, idx): return self.arr[idx]

class LSTMAE(nn.Module):
    def __init__(self, n_features=3, hidden=32, latent=8):
        super().__init__()
        self.enc = nn.LSTM(n_features, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, latent)
        self.dec_fc = nn.Linear(latent, hidden)
        self.dec = nn.LSTM(hidden, n_features, batch_first=True)
    def forward(self, x):
        _, (h,_) = self.enc(x)
        z = self.fc(h[-1])
        h2 = self.dec_fc(z).unsqueeze(0)
        out, _ = self.dec(x, (h2, torch.zeros_like(h2)))
        return out

def main(args):
    os.makedirs('model', exist_ok=True)
    # load or generate data
    if args.data and os.path.exists(args.data):
        print("Loading data from", args.data)
        df = pd.read_csv(args.data)
        # If your CSV doesn't have is_anomaly column, create False (we don't need it for training)
        if 'is_anomaly' not in df.columns:
            df['is_anomaly'] = False
    else:
        print("No data CSV provided or not found — generating synthetic data for dev")
        df = generate_synthetic(npatients=args.npatients, minutes=args.minutes, anomaly_prob=args.anomaly_prob)
        if args.dump_synthetic:
            df.to_csv('data/sim.csv', index=False)
            print("Saved synthetic to data/sim.csv")

    # windows
    W, labels = build_windows(df, window=args.window)
    print("Built windows:", W.shape, "Anomalous windows:", int(labels.sum()))

    # fit scaler on normal windows (reshape to 2D)
    normal_idx = np.where(labels == 0)[0]
    if len(normal_idx) == 0:
        # fallback: use all windows
        flat = W.reshape(-1, W.shape[2])
    else:
        flat = W[normal_idx].reshape(-1, W.shape[2])
    scaler = StandardScaler().fit(flat)

    W_scaled = scaler.transform(W.reshape(-1, W.shape[2])).reshape(W.shape)

    # train/test split (simple)
    # here we keep it simple: train on all windows (unsupervised)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    ds = WindowDataset(W_scaled)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True)

    model = LSTMAE(n_features=W.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        tot = 0.0; steps = 0
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = loss_fn(out, batch)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()); steps += 1
        print(f"Epoch {epoch+1}/{args.epochs} loss={tot/max(1,steps):.6f}")

    # save artifacts
    torch.save({'model_state': model.state_dict(), 'n_features': W.shape[2]}, args.out_model)
    joblib.dump(scaler, args.out_scaler)
    print("Saved model:", args.out_model)
    print("Saved scaler:", args.out_scaler)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to CSV with columns patient_id,timestamp,hr,rr,spo2 (optional is_anomaly)")
    parser.add_argument("--out_model", default="model/model.pt")
    parser.add_argument("--out_scaler", default="model/scaler.pkl")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--npatients", type=int, default=50)
    parser.add_argument("--minutes", type=int, default=600)
    parser.add_argument("--anomaly_prob", type=float, default=0.001)
    parser.add_argument("--dump_synthetic", action="store_true", help="Write synthetic CSV to data/sim.csv")
    args = parser.parse_args()
    main(args)
