import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn

from .model import BeatCNN

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def class_weights(y, n=4):
    cnt = np.bincount(y, minlength=n).astype(float)
    w = 1 / np.sqrt(np.maximum(cnt, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def augment(xb):
    """xb: [B,1,L] z-scored beats. Small random time-stretch, shift, and noise."""
    B, _, L = xb.shape
    t = torch.linspace(-1, 1, L, device=xb.device)[None, :]
    scale = torch.empty(B, 1, device=xb.device).uniform_(0.9, 1.1)
    shift = torch.empty(B, 1, device=xb.device).uniform_(-0.05, 0.05)
    gx = ((t / scale) + shift).clamp(-1, 1)
    grid = torch.stack([gx, torch.zeros_like(gx)], -1).unsqueeze(1)     # [B,1,L,2]
    xb = F.grid_sample(xb.unsqueeze(2), grid, align_corners=True).squeeze(2)
    sigma = torch.empty(B, 1, 1, device=xb.device).uniform_(0.0, 0.1)
    return xb + sigma * torch.randn_like(xb)


def fit(X, RR, y, epochs=15, bs=256, lr=1e-3, seed=42, aug=False, wd=0.0 , model_fn=BeatCNN):
    torch.manual_seed(seed)
    model = model_fn().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss(weight=class_weights(y).to(DEVICE))
    Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    Rt = torch.tensor(RR, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    n = len(Xt)
    for ep in range(epochs):
        model.train()
        perm, tot = torch.randperm(n), 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, rb, yb = Xt[idx].to(DEVICE), Rt[idx].to(DEVICE), yt[idx].to(DEVICE)
            if aug:
                xb = augment(xb)
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

    out = [
        model(
            torch.tensor(
                X[i:i + bs],
                dtype=torch.float32
            ).unsqueeze(1).to(DEVICE),

            torch.tensor(
                RR[i:i + bs],
                dtype=torch.float32
            ).to(DEVICE)
        ).cpu()

        for i in range(0, len(X), bs)
    ]

    return torch.cat(out).numpy()
def fit_curve(Xtr, Rtr, ytr, Xva, Rva, yva, epochs=40, bs=256, lr=1e-3, seed=42, wd=0.0,
              model_fn=BeatCNN):
    """Train and record held-out metrics after every epoch."""
    torch.manual_seed(seed)
    model = model_fn().to(DEVICE)  

    opt = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=wd
    )

    lossf = nn.CrossEntropyLoss(
        weight=class_weights(ytr).to(DEVICE)
    )

    Xt = torch.tensor(
        Xtr, dtype=torch.float32
    ).unsqueeze(1)

    Rt = torch.tensor(
        Rtr, dtype=torch.float32
    )

    yt = torch.tensor(
        ytr, dtype=torch.long
    )

    n = len(Xt)

    hist = []

    for ep in range(epochs):

        model.train()

        perm = torch.randperm(n)

        for i in range(0, n, bs):

            idx = perm[i:i + bs]

            xb = Xt[idx].to(DEVICE)
            rb = Rt[idx].to(DEVICE)
            yb = yt[idx].to(DEVICE)

            opt.zero_grad()

            loss = lossf(
                model(xb, rb),
                yb
            )

            loss.backward()
            opt.step()

        # -----------------------------
        # Validation after this epoch
        # -----------------------------

        lg = get_logits(
            model,
            Xva,
            Rva
        )

        pr = lg.argmax(1)

        # Unweighted validation NLL
        nll = F.cross_entropy(
            torch.tensor(lg, dtype=torch.float32),
            torch.tensor(yva, dtype=torch.long)
        ).item()

        # N / S / V recall
        rec_ = [
            (pr[yva == c] == c).mean()
            if (yva == c).any()
            else np.nan
            for c in range(3)
        ]

        hist.append([
            nll,
            (pr == yva).mean(),
            *rec_
        ])

    return np.array(hist)