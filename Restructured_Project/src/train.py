import numpy as np, torch, torch.nn as nn
from .model import BeatCNN

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def class_weights(y, n=4):
    cnt = np.bincount(y, minlength=n).astype(float)
    w = 1 / np.sqrt(np.maximum(cnt, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)

def fit(X, RR, y, epochs=15, bs=256, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    model = BeatCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss(weight=class_weights(y).to(DEVICE))
    Xt = torch.tensor(X).unsqueeze(1)
    Rt = torch.tensor(RR)
    yt = torch.tensor(y)
    n = len(Xt)
    for ep in range(epochs):
        model.train()
        perm, tot = torch.randperm(n), 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, rb, yb = Xt[idx].to(DEVICE), Rt[idx].to(DEVICE), yt[idx].to(DEVICE)
            opt.zero_grad()
            loss = lossf(model(xb, rb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        print(f'  epoch {ep + 1}: loss {tot / n:.4f}')
    return model

@torch.no_grad()
def get_logits(model, X, RR, bs=1024):
    model.eval()
    out = [model(torch.tensor(X[i:i + bs]).unsqueeze(1).to(DEVICE),
                 torch.tensor(RR[i:i + bs]).to(DEVICE)).cpu()
           for i in range(0, len(X), bs)]
    return torch.cat(out).numpy()