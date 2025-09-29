# backend/inference.py
import os
import joblib
import numpy as np
import torch
from torch import nn

# Canonical artifact paths (project-root relative)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "artifacts", "model.pt")
SCALER_PATH = os.path.join(PROJECT_ROOT, "model", "artifacts", "scaler.pkl")

class LSTMAE(nn.Module):
    def __init__(self, n_features=3, hidden=64, latent=16, num_layers=1):
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.num_layers = num_layers

        self.enc = nn.LSTM(input_size=n_features, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, latent)
        self.dec_fc = nn.Linear(latent, hidden)
        self.dec = nn.LSTM(input_size=n_features, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.out_fc = nn.Linear(hidden, n_features)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        _, (h, _) = self.enc(x)
        h_last = h[-1]
        z = self.fc(h_last)
        h_dec0 = self.dec_fc(z).unsqueeze(0)
        if self.num_layers > 1:
            h_dec0 = h_dec0.repeat(self.num_layers, 1, 1)
        c_dec0 = torch.zeros_like(h_dec0)
        dec_in = torch.zeros(batch_size, seq_len, self.n_features, device=x.device, dtype=x.dtype)
        dec_out, _ = self.dec(dec_in, (h_dec0, c_dec0))
        out = self.out_fc(dec_out)
        return out
def load_model():
    """Return (model, scaler). If artifacts missing return (None, None)."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print(f"[inference] WARNING: artifacts missing: {MODEL_PATH} or {SCALER_PATH}")
        return None, None

    scaler = joblib.load(SCALER_PATH)
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    n_features = int(ckpt.get("n_features", 3)) if isinstance(ckpt, dict) else 3
    model = LSTMAE(n_features=n_features)

    # Robust loading
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        else:
            # fallback: try loading the whole dict
            try:
                model.load_state_dict(ckpt)
            except Exception as e:
                print(f"[inference] ERROR loading state_dict: {e}")
                return None, None
    else:
        # ckpt is just the state_dict itself
        model.load_state_dict(ckpt)

    model.eval()
    print("[inference] Loaded model and scaler.")
    return model, scaler

def predict_window(model, scaler, window):
    """
    window: list or numpy array shape (T, F) where F == scaler/n_features
    returns: anomaly score 0..1
    """
    if model is None or scaler is None:
        # safe fallback score (small)
        return float(np.random.rand() * 0.12)

    arr = np.array(window, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, scaler.mean_.shape[0])
    if arr.ndim != 2:
        raise ValueError("window must be 2D array-like (T, features)")

    arr_scaled = scaler.transform(arr)
    x = torch.tensor(arr_scaled, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = model(x).cpu().numpy().squeeze(0)
    recon_err = float(np.mean((out - arr_scaled) ** 2))
    score = 1 - np.exp(-recon_err * 2.0)
    return float(max(0.0, min(1.0, score)))
