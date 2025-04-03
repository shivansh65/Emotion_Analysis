# model.py
import torch
import torch.nn as nn

class EmotionDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super(EmotionDetectionModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, 5, padding=2)  # Input channels = 1
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(128, 64, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.lstm1 = nn.LSTM(64, 128, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = x.permute(0, 2, 1)  # Reshape for LSTM: (batch_size, seq_len, channels)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x[:, -1, :])  # Use the last output of the LSTM
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
