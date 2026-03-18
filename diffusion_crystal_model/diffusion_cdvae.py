"""
diffusion_cdvae.py

晶体扩散变分自编码器主架构 (Diffusion-CDVAE) - 生产级完全体
集成 CGCNN Encoder，支持 Classifier-Free Guidance (CFG) 与 Temperature 控制采样。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入底层组件
from schedules import ContinuousScheduler
from dynamics import CrystalDynamics
from cgcnn_encoder import CGCNNEncoder  # 🌟 直接引入真正的 CGCNN

# ==========================================
# 1. 辅助预测器 (Predictors)
# ==========================================
class PropertyPredictor(nn.Module):
    def __init__(self, latent_dim, cond_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, cond_dim)
        )
    def forward(self, z): return self.net(z)

class LatticePredictor(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 9)
        )
    def forward(self, z): return self.net(z).view(-1, 3, 3)

class LengthPredictor(nn.Module):
    def __init__(self, latent_dim, max_atoms=20):
        super().__init__()
        self.max_atoms = max_atoms
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.SiLU(),
            nn.Linear(128, max_atoms + 1) 
        )
    def forward(self, z): return self.net(z)

# ==========================================
# 2. 扩散解码器 (Diffusion Decoder)
# ==========================================
class DiffusionDecoder(nn.Module):
    def __init__(self, latent_dim=128, node_dim=64, time_dim=64, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        self.scheduler = ContinuousScheduler(timesteps=timesteps, schedule_type="cosine")
        self.dynamics = CrystalDynamics(node_dim=node_dim, time_dim=time_dim)
        
        self.species_predictor = nn.Sequential(
            nn.Linear(node_dim, 128), nn.SiLU(),
            nn.Linear(128, 100) 
        )

    def forward_training(self, z_nodes, frac_coords, lattice, num_atoms_list, batch_indices, species, cond_drop_prob=0.1):
        """
        训练前向：加入 CFG (Classifier-Free Guidance) 逻辑
        """
        device = frac_coords.device
        batch_size = lattice.size(0)
        
        # 🌟 CFG 训练：以一定概率 (如 10%) 将条件 z_nodes 设为全 0，让模型学习"无条件生成"
        if cond_drop_prob > 0 and torch.rand(1).item() < cond_drop_prob:
            z_nodes = torch.zeros_like(z_nodes)

        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        noise = torch.randn_like(frac_coords)
        t_nodes = t[batch_indices] 
        
        x_t = self.scheduler.q_sample(frac_coords, t_nodes, noise=noise)
        x_t = x_t - torch.floor(x_t)
        
        pred_noise, h_final = self.dynamics(z_nodes, t, x_t, lattice, num_atoms_list, batch_indices)
        loss_diffusion = F.mse_loss(pred_noise, noise)
        
        species_logits = self.species_predictor(h_final)
        loss_species = F.cross_entropy(species_logits, species)
        
        return loss_diffusion, loss_species

    @torch.no_grad()
    def sample(self, z_nodes, lattice, num_atoms_list, batch_indices, guidance_scale=2.0, temperature=1.0):
        """
        采样生成：集成 CFG 引导与温度控制
        """
        device = lattice.device
        batch_size = lattice.size(0)
        num_total_atoms = sum(num_atoms_list)
        
        x_t = torch.randn(num_total_atoms, 3, device=device)
        x_t = x_t - torch.floor(x_t) 
        
        # 为了 CFG 准备纯无条件的特征
        z_nodes_uncond = torch.zeros_like(z_nodes)
        h_final = None 
        
        for time_step in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), time_step, device=device, dtype=torch.long)
            
            # 🌟 CFG 采样核心：同时计算有条件和无条件的 Score
            if guidance_scale > 1.0:
                pred_noise_cond, h_cond = self.dynamics(z_nodes, t, x_t, lattice, num_atoms_list, batch_indices)
                pred_noise_uncond, _ = self.dynamics(z_nodes_uncond, t, x_t, lattice, num_atoms_list, batch_indices)
                # 引导公式：无条件梯度 + scale * (条件梯度 - 无条件梯度)
                pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)
                h_current = h_cond
            else:
                pred_noise, h_current = self.dynamics(z_nodes, t, x_t, lattice, num_atoms_list, batch_indices)

            if time_step == 0:
                h_final = h_current
            
            t_nodes = t[batch_indices]
            alphas_t = self.scheduler._extract(1.0 - self.scheduler.betas, t_nodes, x_t.shape)
            sqrt_one_minus_alphas_cumprod_t = self.scheduler._extract(self.scheduler.sqrt_one_minus_alphas_cumprod, t_nodes, x_t.shape)
            
            x_t_prev_mean = (1.0 / torch.sqrt(alphas_t)) * (x_t - (1.0 - alphas_t) / sqrt_one_minus_alphas_cumprod_t * pred_noise)
            
            if time_step > 0:
                posterior_variance_t = self.scheduler._extract(self.scheduler.posterior_variance, t_nodes, x_t.shape)
                # 🌟 温度控制：乘以 temperature 调节退火的随机性
                noise = torch.randn_like(x_t) * temperature
                x_t = x_t_prev_mean + torch.sqrt(posterior_variance_t) * noise
            else:
                x_t = x_t_prev_mean
                
            x_t = x_t - torch.floor(x_t)
            
        species_logits = self.species_predictor(h_final) 
        return x_t, species_logits

# ==========================================
# 3. 总指挥部 (Diffusion-CDVAE)
# ==========================================
class DiffusionCDVAE(nn.Module):
    """
    高度内聚的主模型
    包含了编码、预测、扩散的全工作流，外部只需调用 compute_loss。
    """
    def __init__(self, latent_dim=128, cond_dim=1, timesteps=1000, max_atoms=20):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = CGCNNEncoder(latent_dim=latent_dim) # 直接内化 CGCNN
        
        self.property_predictor = PropertyPredictor(latent_dim, cond_dim)
        self.lattice_predictor = LatticePredictor(latent_dim)
        self.length_predictor = LengthPredictor(latent_dim, max_atoms)
        
        self.decoder = DiffusionDecoder(latent_dim=latent_dim, timesteps=timesteps)

    def compute_loss(self, batch, device):
        """
        🌟 完整封装 Loss：Train.py 再也不需要写长篇大论的 loss 计算了！
        """
        lattice, fracs, species, props = batch['lattice'].to(device), batch['fracs'].to(device), batch['species'].to(device), batch['props'].to(device)
        batch_indices, num_atoms_list = batch['batch_indices'].to(device), batch['num_atoms']

        # 1. CGCNN 编码
        mu, logvar = self.encoder(lattice, fracs, species, batch_indices, num_atoms_list)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        loss_kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))

        # 2. 预测器损失
        loss_prop = F.mse_loss(self.property_predictor(z), props)
        loss_lattice = F.mse_loss(self.lattice_predictor(z), lattice)
        target_lengths = torch.tensor(num_atoms_list, device=device).clamp(max=self.length_predictor.max_atoms)
        loss_length = F.cross_entropy(self.length_predictor(z), target_lengths)

        # 3. 扩散与元素损失 (内置 10% 概率 Drop Condition)
        z_nodes = z[batch_indices]
        loss_diff, loss_species = self.decoder.forward_training(
            z_nodes, fracs, lattice, num_atoms_list, batch_indices, species, cond_drop_prob=0.1
        )

        total_loss = (loss_diff + 0.5 * loss_prop + 0.1 * loss_lattice + 
                      0.1 * loss_length + 0.5 * loss_species + 0.01 * loss_kl)

        # 返回详尽的 log dict 方便后续接入 wandb
        loss_dict = {
            "loss_total": total_loss, "loss_diff": loss_diff.item(), 
            "loss_prop": loss_prop.item(), "loss_species": loss_species.item(), 
            "loss_kl": loss_kl.item()
        }
        return total_loss, loss_dict