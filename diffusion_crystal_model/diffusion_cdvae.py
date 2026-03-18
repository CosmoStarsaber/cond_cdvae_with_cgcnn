"""
diffusion_cdvae.py

晶体扩散变分自编码器主架构 (Diffusion-CDVAE) - 生产级完全体
集成：
1. 真正的 CGCNN 门控图卷积编码器
2. 隐含 x0 排斥力惩罚 (防原子坍缩/黑洞效应)
3. Classifier-Free Guidance (CFG) 与 Temperature 控制采样
4. 局部几何感知 (基于 h_final 的元素预测)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入底层组件
from schedules import ContinuousScheduler
from dynamics import CrystalDynamics
from cgcnn_encoder import CGCNNEncoder  # 🌟 引入真正的 CGCNN

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
        
        # 接收 dynamics 输出的局部特征 h_final (维度为 node_dim) 进行元素预测
        self.species_predictor = nn.Sequential(
            nn.Linear(node_dim, 128), nn.SiLU(),
            nn.Linear(128, 100) 
        )

    def forward_training(self, z_nodes, frac_coords, lattice, num_atoms_list, batch_indices, species, cond_drop_prob=0.1):
        """
        训练前向：加入 CFG (Classifier-Free Guidance) 逻辑与隐含排斥力惩罚
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
        
        # 预测去噪梯度和降温后的局部节点特征 h_final
        pred_noise, h_final = self.dynamics(z_nodes, t, x_t, lattice, num_atoms_list, batch_indices)
        
        # 1. 基础扩散 MSE 损失
        loss_diffusion = F.mse_loss(pred_noise, noise)
        
        # 🌟 2. 核心修复：隐含 x_0 排斥惩罚 (Implied x_0 Repulsion) 防坍缩
        # 提取调度常数
        sqrt_alphas_t = self.scheduler._extract(self.scheduler.sqrt_alphas_cumprod, t_nodes, x_t.shape)
        sqrt_one_minus_alphas_t = self.scheduler._extract(self.scheduler.sqrt_one_minus_alphas_cumprod, t_nodes, x_t.shape)
        
        # 反推模型当前认为的干净坐标 (x_0)
        pred_x0 = (x_t - sqrt_one_minus_alphas_t * pred_noise) / sqrt_alphas_t
        pred_x0 = pred_x0 - torch.floor(pred_x0) # PBC wrap
        
        l_repulsion = 0.0
        start_idx = 0
        for i, n in enumerate(num_atoms_list):
            if n > 1: # 只有包含两个及以上原子的晶胞才计算排斥
                f = pred_x0[start_idx : start_idx + n]
                lat = lattice[i]
                
                # 计算 PBC 距离
                diff = f.unsqueeze(1) - f.unsqueeze(0)
                diff = diff - torch.round(diff)
                diff_c = torch.matmul(diff, lat)
                dist_sq = torch.sum(diff_c ** 2, dim=-1)
                dist_sq.fill_diagonal_(float('inf'))
                
                # 添加 1e-8 防止原子完全重合时导致梯度爆炸 (NaN)
                dist = torch.sqrt(dist_sq + 1e-8)
                
                # 硬截断：如果距离小于 0.8 埃，产生极其强烈的二次惩罚
                rep = torch.relu(0.8 - dist) ** 2
                l_repulsion += rep.sum() / n
            start_idx += n
            
        l_repulsion = l_repulsion / batch_size
        
        # 强行将斥力注入主损失流，权重设为 5.0 (可根据情况微调)
        loss_diffusion = loss_diffusion + 5.0 * l_repulsion
        
        # 3. 元素预测损失 (基于感知了邻居的 h_final)
        species_logits = self.species_predictor(h_final)
        loss_species = F.cross_entropy(species_logits, species)
        
        return loss_diffusion, loss_species, l_repulsion

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

            # 在最后一步彻底降温时，提取几何节点特征用于预测化学元素
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
        self.encoder = CGCNNEncoder(latent_dim=latent_dim) # 直接内化正宗 CGCNN
        
        self.property_predictor = PropertyPredictor(latent_dim, cond_dim)
        self.lattice_predictor = LatticePredictor(latent_dim)
        self.length_predictor = LengthPredictor(latent_dim, max_atoms)
        
        self.decoder = DiffusionDecoder(latent_dim=latent_dim, timesteps=timesteps)

    def compute_loss(self, batch, device):
        """
        完整封装 Loss 逻辑，包含所有辅助任务和斥力惩罚
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

        # 3. 扩散与元素损失 (内置 10% 概率 Drop Condition 和 Repulsion 斥力计算)
        z_nodes = z[batch_indices]
        loss_diff, loss_species, l_rep = self.decoder.forward_training(
            z_nodes, fracs, lattice, num_atoms_list, batch_indices, species, cond_drop_prob=0.1
        )

        # 总损失聚合 (loss_diff 内部已经加上了 5.0 * l_repulsion)
        total_loss = (loss_diff + 0.5 * loss_prop + 0.1 * loss_lattice + 
                      0.1 * loss_length + 0.5 * loss_species + 0.01 * loss_kl)

        # 返回详尽的 log dict 供 train.py 打印和监控
        loss_dict = {
            "loss_total": total_loss, 
            "loss_diff": loss_diff.item(), 
            "loss_prop": loss_prop.item(), 
            "loss_species": loss_species.item(), 
            "loss_kl": loss_kl.item(),
            "loss_rep": l_rep.item() if isinstance(l_rep, torch.Tensor) else l_rep
        }
        return total_loss, loss_dict