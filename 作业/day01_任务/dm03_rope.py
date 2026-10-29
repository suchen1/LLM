import torch
import torch.nn as nn

def compute_rope_params(head_dim=128, theta_base=1_000_000, context_length=40_960):
    """返回 cos, sin，形状都是 (context_length, head_dim)"""
    assert head_dim % 2 == 0, "head_dim 必须是偶数"
    even_idx = torch.arange(0, head_dim, 2,dtype=torch.float32)  # (head_dim/2,)
    # even_idx = even_idx[:head_dim//2]  # (head_dim/2,)
    theta = 1.0 / theta_base ** (even_idx / head_dim)  # (head_dim/2,)  1000000 ^(-(2i)/d_model)
    seq_idx = torch.arange(context_length)  # (context_length,)
    angle = seq_idx.unsqueeze(1) * theta.unsqueeze(0)  # (context_length, head_dim/2)
    angle = torch.cat([angle,angle],dim = 1)
    return torch.cos(angle), torch.sin(angle)

def apply_rope(x, cos, sin, offset=0):
    """x: (B, H, T, head_dim)，返回同形状"""
    batch_size,head_size,seq_len,head_dim = x.shape
    assert head_dim % 2 == 0, "head_dim 必须是偶数"
    x1 = x[..., :head_dim//2]  # (B, H, T, head_dim/2)
    x2 = x[..., head_dim//2:]  # (B, H, T, head_dim/2)
    cos = cos[offset:offset+seq_len, :].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    sin = sin[offset:offset+seq_len, :].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    rotated = torch.cat([-x2, x1], dim=-1)  # (B, H, T, head_dim)
    x_rotated = x * cos + rotated * sin  # (B, H, T, head_dim)   [a,b] -> [a*cos - b*sin, a*sin + b*cos] = [a,b] * cos + [-b,a] * sin
    return x_rotated




if __name__== "__main__":
    cos, sin = compute_rope_params()
    print(cos.shape, sin.shape)  # (40960, 128) (40960, 128)