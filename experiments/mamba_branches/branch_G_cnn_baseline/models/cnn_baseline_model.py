"""G_basic: CNN+MLP 基线, G_lstm: CNN+LSTM — 魔鬼代言人核心质疑对照"""

import torch, torch.nn as nn, torch.nn.functional as F

class CNNEncoder(nn.Module):
    """Branch E 的轻量级 CNN 编码器 (0.46M)"""
    def __init__(self, in_channels=1, output_dim=256):
        super().__init__()
        layers, c = [], 32
        for i, (cin, cout) in enumerate([(in_channels, 32), (32, 64), (64, 128), (128, 256)]):
            layers.append(nn.Sequential(
                nn.Conv2d(cin, cout, 3, 2, 1), nn.BatchNorm2d(cout), nn.GELU()))
        self.convs = nn.ModuleList(layers)
        self.pool, self.fc = nn.AdaptiveAvgPool2d(1), nn.Linear(256, output_dim)
    def forward(self, x):
        for c in self.convs:
            x = c(x)
        return self.fc(self.pool(x).flatten(1))


class CNNMLPNet(nn.Module):
    """G_basic: CNN + 2层MLP (无SSM), 0.62M, 0.74ms"""
    def __init__(self, embed_dim=256, dropout=0.1):
        super().__init__()
        self.encoder = CNNEncoder(output_dim=embed_dim)
        self.head = nn.Sequential(
            nn.Linear(263, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 3))
    def forward(self, X):
        if X[2] is None:
            X[2] = torch.zeros((X[0].shape[0], 4), device=X[0].device); X[2][:, 0] = 1
        if X[0].shape[-2:] != (60, 90):
            X[0] = F.interpolate(X[0], (60, 90), mode='bilinear')
        feat = self.encoder(X[0])
        meta = torch.cat((X[1] * 0.1, X[2]), dim=1)
        return self.head(torch.cat((feat, meta), dim=1)), None


class CNNLSTMNet(nn.Module):
    """G_lstm: CNN + 单层LSTM, 0.79M, 1.00ms — 用于延迟对比"""
    def __init__(self, embed_dim=256, lstm_hidden=128):
        super().__init__()
        self.encoder = CNNEncoder(output_dim=embed_dim)
        self.lstm = nn.LSTM(263, lstm_hidden, batch_first=True)
        self.fc_out = nn.Linear(lstm_hidden, 3)
    def forward(self, X):
        if X[2] is None:
            X[2] = torch.zeros((X[0].shape[0], 4), device=X[0].device); X[2][:, 0] = 1
        if X[0].shape[-2:] != (60, 90):
            X[0] = F.interpolate(X[0], (60, 90), mode='bilinear')
        B = X[0].shape[0]
        feat = self.encoder(X[0])
        meta = torch.cat((X[1] * 0.1, X[2]), dim=1)
        lstm_in = torch.cat((feat, meta), dim=1).unsqueeze(1)
        lstm_out, _ = self.lstm(lstm_in)
        return self.fc_out(lstm_out[:, -1, :]), None


def create_cnn_baseline_model(config):
    return CNNMLPNet(embed_dim=config.get('embed_dim', 256), dropout=config.get('dropout', 0.1))

def create_cnn_lstm_model(config):
    return CNNLSTMNet(embed_dim=config.get('embed_dim', 256))


if __name__ == '__main__':
    for name, cls in [('G_basic', CNNMLPNet), ('G_lstm', CNNLSTMNet)]:
        m = cls(); p = sum(x.numel() for x in m.parameters())
        print(f"{name}: {p:,} ({p/1e6:.2f}M)")
