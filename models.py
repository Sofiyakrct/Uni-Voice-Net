import torch
import torch.nn as nn

class CNNRNNEmbedding(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=128, num_layers=1, embedding_dim=64):
        super(CNNRNNEmbedding, self).__init__()

        # CNN expects (batch, channels=input_dim, seq)
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # RNN expects (batch, seq, features)
        self.rnn = nn.GRU(64, hidden_dim, num_layers, batch_first=True)

        # Final embedding layer
        self.fc = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, mfcc_features)
        x = x.transpose(1, 2)               # (batch, features, seq_len)
        x = self.cnn(x).squeeze(-1)         # (batch, 64)

        x = x.unsqueeze(1)                  # (batch, 1, 64)
        x, _ = self.rnn(x)                  # RNN output
        x = self.fc(x[:, -1, :])            # Final embedding
        return x
