import torch
import torch.nn as nn
import numpy as np

SEQ_LEN = 100

class SO3Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * (SEQ_LEN - 1), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)

# Preload model
model = SO3Encoder()
model.load_state_dict(torch.load("models/so3_encoder.pth", map_location="cpu"))
model.eval()

# Dummy mean/std from training – must match training script!
mean_value = [[[-0.0016748812049627304, 0.0022705025039613247, -0.0030523024033755064]]]  # <== your actual values
std_value = [[[0.35158345103263855, 0.3539157509803772, 0.35216039419174194]]]

mean = np.array(mean_value).reshape(1, 1, 3)
std = np.array(std_value).reshape(1, 1, 3)
mean = np.tile(mean, (1, SEQ_LEN - 1, 1))
std = np.tile(std, (1, SEQ_LEN - 1, 1))

def predict_omega(x_seq: list[list[float]]) -> list[float]:
    x_np = np.array(x_seq, dtype=np.float32).reshape(1, SEQ_LEN - 1, 3)
    x_norm = (x_np - mean) / std
    with torch.no_grad():
        x_tensor = torch.tensor(x_norm, dtype=torch.float32)
        pred = model(x_tensor).cpu().numpy().squeeze()
        return pred.tolist()