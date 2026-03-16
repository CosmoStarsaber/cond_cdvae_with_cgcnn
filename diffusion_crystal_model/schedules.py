"""
schedules.py

材料扩散模型的噪声调度器 (Noise Schedulers)
负责生成连续坐标和晶格扩散所需的 beta, alpha 和 sigma 序列。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_beta_schedule(timesteps, s=0.008):
    """
    余弦调度 (Cosine Schedule)
    相比线性调度，余弦调度在 t 接近 0 时能保留更多结构信息，
    非常适合极其敏感的晶体原子坐标扩散。
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """
    线性调度 (Linear Schedule)
    经典的 DDPM 调度器，通常用于非坐标特征（如隐藏状态或晶格参数）的扩散。
    """
    return torch.linspace(beta_start, beta_end, timesteps)

class ContinuousScheduler(nn.Module):
    """
    连续变量扩散调度器
    预先计算所有扩散过程所需的数学常数，避免在训练循环中重复计算。
    """
    def __init__(self, timesteps=1000, schedule_type="cosine"):
        super().__init__()
        self.timesteps = timesteps
        
        if schedule_type == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif schedule_type == "linear":
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f"不支持的调度类型: {schedule_type}")
            
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 注册为 buffer，这样它们会自动跟随模型移动到 GPU/CPU，且不需要计算梯度
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # 前向加噪过程的计算常数：q(x_t | x_0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # 反向去噪过程的计算常数：p(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        
        # 对数方差被截断以防止在 t=0 时出现 -inf
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    def q_sample(self, x_start, t, noise=None):
        """
        前向加噪过程：直接从 x_0 跳到 x_t
        公式：x_t = sqrt(bar_alpha_t) * x_0 + sqrt(1 - bar_alpha_t) * epsilon
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # 提取当前批次中每个样本对应时间步 t 的常数
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def _extract(self, a, t, x_shape):
        """从预计算的张量 'a' 中提取特定时间步 't' 的值，并将其形状扩展以匹配 'x_shape'"""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)