"""
diffusion_cdvae.py

晶体扩散变分自编码器主架构 (Diffusion-CDVAE)
将 GNN 编码器、多任务预测器 (Predictors) 与扩散解码引擎 (Diffusion Decoder) 完美整合。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入我们前两步写好的硬核模块
from schedules import ContinuousScheduler
from dynamics import CrystalDynamics

# ==========================================
# 1. 辅助预测器 (Predictors)
# ==========================================
class PropertyPredictor(nn.Module):
    """从 z 预测物理属性，迫使隐空间有序聚类"""
    def __init__(self, latent_dim, cond_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, cond_dim)
        )
    def forward(self, z): return self.net(z)

class LatticePredictor(nn.Module):
    """从 z 预测晶格矩阵 (3x3)"""
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 9)
        )
    def forward(self, z):
        # 输出形状 (B, 3, 3)
        return self.net(z).view(-1, 3, 3)

class LengthPredictor(nn.Module):
    """从 z 预测晶胞内的原子数量 (分类任务)"""
    def __init__(self, latent_dim, max_atoms=20):
        super().__init__()
        self.max_atoms = max_atoms
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.SiLU(),
            nn.Linear(128, max_atoms + 1) # 类别为 0 到 max_atoms
        )
    def forward(self, z):
        return self.net(z) # 返回 logits，形状 (B, max_atoms + 1)

# ==========================================
# 2. 扩散解码器 (Diffusion Decoder)
# ==========================================
class DiffusionDecoder(nn.Module):
    """
    统筹扩散的前向加噪与反向去噪过程
    """
    def __init__(self, node_dim=64, time_dim=64, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        
        # 1. 实例化调度器和动力学网络
        self.scheduler = ContinuousScheduler(timesteps=timesteps, schedule_type="cosine")
        self.dynamics = CrystalDynamics(node_dim=node_dim, time_dim=time_dim)
        
        # 2. 元素种类预测器 (由于元素是离散的，通常在扩散的最后一步或每一步并行预测)
        self.species_predictor = nn.Sequential(
            nn.Linear(node_dim, 128), nn.SiLU(),
            nn.Linear(128, 100) # 假设最多 100 种元素
        )

    def forward_training(self, z_nodes, frac_coords, lattice, num_atoms_list, batch_indices):
        """
        训练时的前向传播：随机挑选时间步 t，加噪，然后让网络预测加入的噪声
        """
        device = frac_coords.device
        batch_size = lattice.size(0)
        
        # 1. 随机采样时间步 t (对于 Batch 中的每个晶体)
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        
        # 2. 模拟扩散：在真实坐标上加上高斯噪声 epsilon
        noise = torch.randn_like(frac_coords)
        
        # 注意：这里需要把 batch 级别的 t 扩展到 node 级别，才能传给 scheduler
        t_nodes = t[batch_indices] 
        
        # 获取加噪后的混沌坐标 x_t (调用 schedules.py 中的公式)
        x_t = self.scheduler.q_sample(frac_coords, t_nodes, noise=noise)
        
        # ⚠️ 关键物理约束：将加噪后的坐标强行 wrap 回 [0, 1) 的晶胞内
        x_t = x_t - torch.floor(x_t)
        
        # 3. 召唤等变图神经网络：预测去噪梯度 (在这个语境下，预测当初加进去的 noise)
        pred_noise = self.dynamics(z_nodes, t, x_t, lattice, num_atoms_list, batch_indices)
        
        # 4. 计算扩散 Loss (MSE)
        loss_diffusion = F.mse_loss(pred_noise, noise)
        
        return loss_diffusion

    @torch.no_grad()
    def sample(self, z_nodes, lattice, num_atoms_list, batch_indices):
        """
        推理时的反向生成：从纯噪声开始，迭代 T 步，慢慢降温得到完美晶体
        """
        device = lattice.device
        batch_size = lattice.size(0)
        num_total_atoms = sum(num_atoms_list)
        
        # 1. 初始化纯噪声坐标 x_T ~ N(0, I)
        x_t = torch.randn(num_total_atoms, 3, device=device)
        x_t = x_t - torch.floor(x_t) # 保证在晶胞内
        
        # 2. 朗之万动力学去噪循环 (从 t=T-1 到 0)
        for time_step in reversed(range(self.timesteps)):
            # 构造当前的时间步张量
            t = torch.full((batch_size,), time_step, device=device, dtype=torch.long)
            
            # 预测噪声
            pred_noise = self.dynamics(z_nodes, t, x_t, lattice, num_atoms_list, batch_indices)
            
            # 提取调度器中的常数 (根据 DDPM 的反向采样公式)
            t_nodes = t[batch_indices]
            alphas_t = self.scheduler._extract(1.0 - self.scheduler.betas, t_nodes, x_t.shape)
            sqrt_one_minus_alphas_cumprod_t = self.scheduler._extract(self.scheduler.sqrt_one_minus_alphas_cumprod, t_nodes, x_t.shape)
            
            # 核心去噪公式：x_{t-1} = 1/sqrt(alpha) * (x_t - (1-alpha)/sqrt(1-bar_alpha) * pred_noise)
            x_t_prev_mean = (1.0 / torch.sqrt(alphas_t)) * (x_t - (1.0 - alphas_t) / sqrt_one_minus_alphas_cumprod_t * pred_noise)
            
            if time_step > 0:
                # 如果还没到最后一步，需要加一点微小的扰动防止陷入局部死胡同
                posterior_variance_t = self.scheduler._extract(self.scheduler.posterior_variance, t_nodes, x_t.shape)
                noise = torch.randn_like(x_t)
                x_t = x_t_prev_mean + torch.sqrt(posterior_variance_t) * noise
            else:
                x_t = x_t_prev_mean
                
            # 随时保持分数坐标在单胞内部
            x_t = x_t - torch.floor(x_t)
            
        # 3. 坐标收敛完毕后，预测每个位点应该放什么元素
        # 这里用一个小技巧：用 t=0 走最后一遍 dynamics 的隐藏状态来预测元素种类
        # 为了极简，我们直接假设 dynamics 有个内部方法提取隐藏状态，这里暂用 MLP 替代演示
        species_logits = self.species_predictor(z_nodes) 
        
        return x_t, species_logits

# ==========================================
# 3. 总指挥部 (Diffusion-CDVAE)
# ==========================================
class DiffusionCDVAE(nn.Module):
    """
    主模型：将 VAE 与 Diffusion 结合
    包含：编码器 (需外部传入), 预测器群, 扩散解码器
    """
    def __init__(self, encoder, latent_dim=128, cond_dim=1, timesteps=1000, max_atoms=20):
        super().__init__()
        self.latent_dim = latent_dim
        
        # 1. 编码器 (为了灵活性，从外部传入我们之前写的 GNNEncoder)
        self.encoder = encoder
        
        # 2. 三大预测器
        self.property_predictor = PropertyPredictor(latent_dim, cond_dim)
        self.lattice_predictor = LatticePredictor(latent_dim)
        self.length_predictor = LengthPredictor(latent_dim, max_atoms)
        
        # 3. 扩散引擎
        self.decoder = DiffusionDecoder(timesteps=timesteps)

    def encode(self, lattice, fracs, species, batch_indices, num_atoms_list):
        return self.encoder(lattice, fracs, species, batch_indices, num_atoms_list)