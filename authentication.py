import torch
import torch.nn as nn

class CNNRNNEmbedding(nn.Module):
    def __init__(self):
        super(CNNRNNEmbedding, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(13, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.rnn = nn.GRU(64, 128, batch_first=True)
        self.fc = nn.Linear(128, 64)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x).squeeze(-1)
        x = x.unsqueeze(1)
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x
