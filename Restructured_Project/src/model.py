import torch, torch.nn as nn

class BeatCNN(nn.Module):
    def __init__(self, n_classes=4, n_extra=4):
        super().__init__()
        def block(i, o, k):
            return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2),
                                 nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2))
        self.features = nn.Sequential(block(1, 16, 7), block(16, 32, 5),
                                      block(32, 64, 5), block(64, 64, 3))
        self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64 + n_extra, n_classes))

    def forward(self, x, rr):
        return self.head(torch.cat([self.pool(self.features(x)), rr], 1))

# ---- APPEND to src/model.py (below BeatCNN) ----
class ResBlock(nn.Module):
    def __init__(self, cin, cout, k, stride):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, stride, k // 2, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, 1, k // 2, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.skip = (nn.Identity() if (cin == cout and stride == 1) else
                     nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False),
                                   nn.BatchNorm1d(cout)))

    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return torch.relu(y + self.skip(x))


class ResBeatCNN(nn.Module):
    """Same widths (16,32,64,64), global pooling, RR concat and head as BeatCNN.
    Only the conv blocks are residual, so the architecture is the one thing that changes."""
    def __init__(self, n_classes=4, n_extra=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(),
            ResBlock(16, 32, 5, 2), ResBlock(32, 64, 5, 2), ResBlock(64, 64, 3, 2))
        self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64 + n_extra, n_classes))

    def forward(self, x, rr):
        return self.head(torch.cat([self.pool(self.features(x)), rr], 1))