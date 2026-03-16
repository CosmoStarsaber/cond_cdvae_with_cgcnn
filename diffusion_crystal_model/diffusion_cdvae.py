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
        return self.net(z).view(-1, 3, 3)

class LengthPredictor(nn.Module):
    """从 z 预测晶胞内的原子数量 (分类任务)"""
    def __init__(self, latent_dim, max_atoms=20):
        super().__init__()
        self.max_atoms = max_atoms
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.SiLU(),
            nn.Linear(128, max_atoms + 1) 
        )
    def forward(self, z):
        return self.net(z)

# ==========================================
# 2. 扩散解码器 (Diffusion Decoder)
# ==========================================
class DiffusionDecoder(nn.Module):
    """
    统筹扩散的前向加噪与反向去噪过程
    """
    def __init__(self, latent_dim=128, node_dim=64, time_dim=64, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        
        self.scheduler = ContinuousScheduler(timesteps=timesteps, schedule_type="cosine")
        self.dynamics = CrystalDynamics(node_dim=node_dim, time_dim=time_dim)
        
        # 🌟 修复：输入维度接收 latent_dim (128)
        self.species_predictor = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.SiLU(),
            nn.Linear(128, 100) 
        )

    def forward_training(self, z_nodes, frac_coords, lattice, num_atoms_list, batch_indices, species):
        """
        训练时的前向传播：加噪，预测噪声，并学习元素种类
        """
        device = frac_coords.device
        batch_size = lattice.size(0)
        
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        noise = torch.randn_like(frac_coords)
        t_nodes = t[batch_indices] 
        
        x_t = self.scheduler.q_sample(frac_coords, t_nodes, noise=noise)
        x_t = x_t - torch.floor(x_t)
        
        pred_noise = self.dynamics(z_nodes, t, x_t, lattice, num_atoms_list, batch_indices)
        loss_diffusion = F.mse_loss(pred_noise, noise)
        
        # 🌟 修复：让网络学习真正的化学元素
        species_logits = self.species_predictor(z_nodes)
        loss_species = F.cross_entropy(species_logits, species)
        
        return loss_diffusion, loss_species

    @torch.no_grad()
    def sample(self, z_nodes, lattice, num_atoms_list, batch_indices):
        """
        推理时的反向生成：朗之万动力学去噪
        """
        device = lattice.device
        batch_size = lattice.size(0)
        num_total_atoms = sum(num_atoms_list)
        
        x_t = torch.randn(num_total_atoms, 3, device=device)
        x_t = x_t - torch.floor(x_t) 
        
        for time_step in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), time_step, device=device, dtype=torch.long)
            
            pred_noise = self.dynamics(z_nodes, t, x_t, lattice, num_atoms_list, batch_indices)
            
            t_nodes = t[batch_indices]
            alphas_t = self.scheduler._extract(1.0 - self.scheduler.betas, t_nodes, x_t.shape)
            sqrt_one_minus_alphas_cumprod_t = self.scheduler._extract(self.scheduler.sqrt_one_minus_alphas_cumprod, t_nodes, x_t.shape)
            
            x_t_prev_mean = (1.0 / torch.sqrt(alphas_t)) * (x_t - (1.0 - alphas_t) / sqrt_one_minus_alphas_cumprod_t * pred_noise)
            
            if time_step > 0:
                posterior_variance_t = self.scheduler._extract(self.scheduler.posterior_variance, t_nodes, x_t.shape)
                noise = torch.randn_like(x_t)
                x_t = x_t_prev_mean + torch.sqrt(posterior_variance_t) * noise
            else:
                x_t = x_t_prev_mean
                
            x_t = x_t - torch.floor(x_t)
            
        # 根据最终降温的节点特征预测元素
        species_logits = self.species_predictor(z_nodes) 
        
        return x_t, species_logits

# ==========================================
# 3. 总指挥部 (Diffusion-CDVAE)
# ==========================================
class DiffusionCDVAE(nn.Module):
    """
    主模型：将 VAE 与 Diffusion 结合
    """
    def __init__(self, encoder, latent_dim=128, cond_dim=1, timesteps=1000, max_atoms=20):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.encoder = encoder
        
        self.property_predictor = PropertyPredictor(latent_dim, cond_dim)
        self.lattice_predictor = LatticePredictor(latent_dim)
        self.length_predictor = LengthPredictor(latent_dim, max_atoms)
        
        # 🌟 修复：传入 latent_dim
        self.decoder = DiffusionDecoder(latent_dim=latent_dim, timesteps=timesteps)

    def encode(self, lattice, fracs, species, batch_indices, num_atoms_list):
        return self.encoder(lattice, fracs, species, batch_indices, num_atoms_list)