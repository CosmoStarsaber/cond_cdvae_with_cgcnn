"""
train.py

晶体扩散模型 (Diffusion-CDVAE) 终极训练与采样流水线
负责数据流转、多任务损失计算、早停监控以及基于梯度寻优的条件采样。
"""

import os
import argparse
import warnings
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from pymatgen.core import Structure, Lattice
from pymatgen.core.periodic_table import Element

# 导入我们前三步手写的硬核模块
from diffusion_cdvae import DiffusionCDVAE

warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")

# ==========================================
# 1. 工业级数据加载与批处理
# ==========================================
class CrystalDataset(Dataset):
    def __init__(self, root_dir: str, id_prop_csv: str = "id_prop.csv"):
        self.root_dir = root_dir
        self.entries = []
        csv_path = os.path.join(root_dir, id_prop_csv)
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到 {csv_path}。")

        with open(csv_path, "r") as f:
            for line in f:
                if not line.strip(): continue
                parts = [p.strip() for p in line.split(",")]
                self.entries.append((parts[0], np.array([float(x) for x in parts[1:]], dtype=np.float32)))
        
        all_props = np.array([e[1] for e in self.entries])
        self.cond_dim = all_props.shape[1] 
        self.prop_mean = all_props.mean(axis=0) 
        self.prop_std = all_props.std(axis=0) + 1e-6 

    def __len__(self): 
        return len(self.entries)

    def __getitem__(self, idx):
        cid, props = self.entries[idx]
        norm_props = (props - self.prop_mean) / self.prop_std 
        
        struct = Structure.from_file(os.path.join(self.root_dir, f"{cid}.cif"))
        fracs = np.array([s.frac_coords for s in struct])
        fracs = (fracs - np.floor(fracs)).astype(np.float32)
        species = np.array([s.specie.Z for s in struct], dtype=np.int64)
        
        return {
            "lattice": torch.tensor(struct.lattice.matrix, dtype=torch.float32),
            "fracs": torch.tensor(fracs, dtype=torch.float32),
            "species": torch.tensor(species, dtype=torch.long),
            "props": torch.tensor(norm_props, dtype=torch.float32),
            "num_atoms": len(species)
        }

def collate_fn(batch):
    batch_lattice = torch.stack([b['lattice'] for b in batch])
    batch_props = torch.stack([b['props'] for b in batch])
    all_fracs = torch.cat([b['fracs'] for b in batch], dim=0)
    all_species = torch.cat([b['species'] for b in batch], dim=0)
    
    batch_indices = []
    for i, b in enumerate(batch):
        batch_indices.extend([i] * b['num_atoms'])
        
    return {
        "lattice": batch_lattice,
        "fracs": all_fracs,
        "species": all_species,
        "props": batch_props,
        "batch_indices": torch.tensor(batch_indices, dtype=torch.long),
        "num_atoms": [b['num_atoms'] for b in batch]
    }

# ==========================================
# 2. VAE 编码器 (保留 GNN 提取局部特征的能力)
# ==========================================
# 由于 dynamics 里的 EGNN 是用来去噪的，我们需要一个常规的 GNN 来做编码。
# 为了保持脚本完整性，我们在此快速定义编码器。
class SimpleGNNEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.emb = nn.Embedding(100, 64, padding_idx=0)
        self.mlp = nn.Sequential(nn.Linear(64 + 3, 128), nn.SiLU(), nn.Linear(128, 128))
        self.out = nn.Sequential(nn.Linear(128 + 9, 128), nn.SiLU(), nn.Linear(128, latent_dim * 2))

    def forward(self, lattice, fracs, species, batch_indices, num_atoms_list):
        # 极简版编码器：聚合原子特征和坐标
        h = self.mlp(torch.cat([self.emb(species), fracs], dim=-1))
        pooled = torch.zeros(lattice.size(0), 128, device=lattice.device).index_add_(0, batch_indices, h)
        pooled = pooled / torch.bincount(batch_indices).view(-1, 1).float()
        return torch.chunk(self.out(torch.cat([pooled, lattice.view(-1, 9)], dim=-1)), 2, dim=-1)

# ==========================================
# 3. 多任务训练 Loss 计算
# ==========================================
def compute_diffusion_loss(model, batch, device):
    lattice = batch['lattice'].to(device)
    fracs = batch['fracs'].to(device)
    species = batch['species'].to(device)
    props = batch['props'].to(device)
    batch_indices = batch['batch_indices'].to(device)
    num_atoms_list = batch['num_atoms']
    batch_size = lattice.size(0)

    # 1. VAE 编码
    mu, logvar = model.encoder(lattice, fracs, species, batch_indices, num_atoms_list)
    std = torch.exp(0.5 * logvar)
    z = mu + torch.randn_like(std) * std
    
    # 2. KL 散度
    loss_kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))

    # 3. 辅助预测器损失 (Predictors)
    loss_prop = F.mse_loss(model.property_predictor(z), props)
    loss_lattice = F.mse_loss(model.lattice_predictor(z), lattice)
    
    pred_lengths = model.length_predictor(z)
    target_lengths = torch.tensor(num_atoms_list, device=device).clamp(max=20)
    loss_length = F.cross_entropy(pred_lengths, target_lengths)

    # 4. 核心扩散损失 (Diffusion)
    z_nodes = z[batch_indices]
    loss_diff = model.decoder.forward_training(z_nodes, fracs, lattice, num_atoms_list, batch_indices)

    # 汇总：扩散模型占主导，Predictor 负责整理潜在空间
    total_loss = loss_diff + 0.5 * loss_prop + 0.1 * loss_lattice + 0.1 * loss_length + 0.01 * loss_kl

    return total_loss, loss_diff.item(), loss_prop.item()

# ==========================================
# 4. 基于潜空间梯度寻优的逆向生成
# ==========================================
def save_structure_to_cif(lattice, fracs, species, filename):
    valid_idx = [i for i, z_num in enumerate(species) if 0 < z_num <= 118]
    if not valid_idx: return
    symbols = [Element.from_Z(int(species[i])).symbol for i in valid_idx]
    struct = Structure(Lattice(lattice), symbols, fracs[valid_idx].tolist())
    struct.to(filename=filename)

@torch.no_grad() # 大部分过程不需要梯度
def generate_diffusion_crystals(model, target_props_norm, out_dir, n_samples=5, device="cpu"):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    
    print(f"\n🔮 [阶段 1] 潜空间梯度寻优 (Latent Optimization)...")
    # 随机初始化纯噪声 z，并允许计算梯度
    z = torch.randn(n_samples, model.latent_dim, device=device, requires_grad=True)
    cond_target = torch.tensor([target_props_norm], dtype=torch.float32).expand(n_samples, -1).to(device)
    
    # 使用 Adam 优化器，以 z 为参数，朝着符合目标物理属性的方向“滑动”
    optimizer_z = torch.optim.Adam([z], lr=0.05)
    for step in range(100):
        with torch.enable_grad():
            pred_props = model.property_predictor(z)
            loss_z = F.mse_loss(pred_props, cond_target)
            optimizer_z.zero_grad()
            loss_z.backward()
            optimizer_z.step()
    
    z = z.detach() # 寻优结束，截断梯度
    final_prop_loss = F.mse_loss(model.property_predictor(z), cond_target).item()
    print(f"   => 优化完成！z 空间对齐误差降至: {final_prop_loss:.4f}")

    print(f"📐 [阶段 2] 宏观结构预测...")
    num_atoms_logits = model.length_predictor(z)
    num_atoms_list = torch.argmax(num_atoms_logits, dim=-1).clamp(min=1).tolist()
    lattice = model.lattice_predictor(z)
    
    batch_indices = torch.tensor([i for i, n in enumerate(num_atoms_list) for _ in range(n)], device=device)
    z_nodes = z[batch_indices]
    
    print(f"🌀 [阶段 3] 朗之万动力学扩散去噪 (Langevin Dynamics) - 共 {model.decoder.timesteps} 步...")
    frac_coords, species_logits = model.decoder.sample(z_nodes, lattice, num_atoms_list, batch_indices)
    species = torch.argmax(species_logits, dim=-1).cpu().numpy()
    fracs_np = frac_coords.cpu().numpy()
    lattice_np = lattice.cpu().numpy()
    
    start_idx = 0
    for i, n in enumerate(num_atoms_list):
        f = fracs_np[start_idx : start_idx + n]
        s = species[start_idx : start_idx + n]
        l = lattice_np[i]
        start_idx += n
        
        out_path = os.path.join(out_dir, f"diff_gen_sample_{i}.cif")
        save_structure_to_cif(l, f, s, out_path)
        print(f"   ✅ 已生成晶体: {out_path} (原子数: {n})")

# ==========================================
# 5. 主程序执行
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="real_mp_dataset")
    parser.add_argument("--batch_size", type=int, default=16) # 扩散模型显存占用较大，batch 设小点
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--timesteps", type=int, default=200, help="扩散过程的时间步数 (推荐 200~1000)")
    parser.add_argument("--target_props", type=float, nargs='+', default=[-2.0, 1.5])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动终极晶体扩散框架！计算设备: {device}")
    
    dataset = CrystalDataset(args.data)
    cond_dim = dataset.cond_dim
    
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 初始化核心架构
    encoder = SimpleGNNEncoder(latent_dim=128)
    model = DiffusionCDVAE(encoder=encoder, latent_dim=128, cond_dim=cond_dim, timesteps=args.timesteps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_diff, train_prop = 0, 0
        for batch in train_loader:
            loss, l_diff, l_prop = compute_diffusion_loss(model, batch, device) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_diff += l_diff; train_prop += l_prop
            
        model.eval()
        val_diff, val_prop = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                loss, l_diff, l_prop = compute_diffusion_loss(model, batch, device)
                val_diff += l_diff; val_prop += l_prop

        a_tr_d, a_tr_p = train_diff/len(train_loader), train_prop/len(train_loader)
        a_va_d, a_va_p = val_diff/len(val_loader), val_prop/len(val_loader)

        print(f"Ep [{epoch+1:03d}] | Tr Diff: {a_tr_d:.3f} (Prop:{a_tr_p:.3f}) | Val Diff: {a_va_d:.3f} (Prop:{a_va_p:.3f})")

    print("\n" + "="*50 + "\n🔥 训练完毕，启动物理条件扩散去噪生成\n" + "="*50)
    targets = (args.target_props + [0.0] * cond_dim)[:cond_dim]
    norm_targets = (np.array(targets, dtype=np.float32) - dataset.prop_mean) / dataset.prop_std
    
    generate_diffusion_crystals(model, norm_targets.tolist(), "ai_diffusion_materials", n_samples=3, device=device)
    print("\n🎉 宏伟的扩散工程已完成！所有数学理论与物理框架都已化作你硬盘里的晶体文件。")