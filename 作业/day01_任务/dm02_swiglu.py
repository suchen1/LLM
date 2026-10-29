import torch
import torch.nn as nn

# Swish激活函数
def swish(x):
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model = 1024, hidd_dim = 3076):
        super().__init__()
        self.up = nn.Linear(d_model, hidd_dim)
        self.gate = nn.Linear(d_model, hidd_dim)
        self.down = nn.Linear(hidd_dim, d_model)
    def forward(self, x):
        w1 = self.up(x)
        w2 = self.gate(x)
        silu = swish(w2)
        return self.down(silu * w1)