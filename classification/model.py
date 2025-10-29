import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(
        self, cnn_out_dim=32, lstm_hidden=64, num_layers=1, num_classes=10, frame_length=5, device="cpu"
    ):
        super().__init__()
        self.device = device
        self.frame_length = frame_length

        # CNN for multi-feature frame input (4 x F)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(2, 2)),  # input (B,1,4,F)
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(128, cnn_out_dim, kernel_size=(2, self.frame_length-2)), # output (B,cnn_out_dim,1,1)
            nn.ReLU(),
            nn.Flatten(),  # (B, cnn_out_dim)
        ).to(device)

        # LSTM for temporal sequence of frames
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,  # will set dynamically after CNN flatten
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        ).to(device)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Linear(lstm_hidden, num_classes),
        ).to(device)

    def forward(self, x):
        B, T_total, _ = x.size()

        # Number of frames per sample
        T_frames = T_total // self.frame_length
        x = x[:, :T_frames * self.frame_length, :]  # trim extra
        x = x.view(B, T_frames, 1, 4, self.frame_length) # (B, T_frames, 1, 4, F)

        # CNN
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        cnn_feats = self.cnn(x)  # (B*T, cnn_out_dim)
        cnn_feats = cnn_feats.view(B, T, -1)

        # LSTM over frames
        lstm_out, _ = self.lstm(cnn_feats)
        out = lstm_out[:, -1, :]  # last frame
        logits = self.classifier(out)
        return logits
