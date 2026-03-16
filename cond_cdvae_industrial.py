"""
cdvae_industrial.py

完整的工业级条件变分自编码器 (CondCDVAE) 
特性：
1. 支持可变原子数晶体的 Batch 批处理 (Dynamic Batching)
2. 自动处理目标性质的 Z-score 标准化与反标准化
3. 包含完整的训练循环与条件生成 (Inverse Design) 流程
"""

import os
import argparse
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from pymatgen.core import Structure, Lattice
from pymatgen.core.periodic_table import Element
from scipy.optimize import linear_sum_assignment

# ==========================================
# 1. 工业级数据加载与批处理
# ==========================================
class CrystalDataset(Dataset):
    def __init__(self, root_dir: str, id_prop_csv: str = "id_prop.csv"):
        self.root_dir = root_dir
        self.entries = []
        csv_path = os.path.join(root_dir, id_prop_csv)
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到 {csv_path}。请先运行 download_mp_data.py 获取真实数据。")

        with open(csv_path, "r") as f:
            for line in f:
                if not line.strip(): continue
                parts = [p.strip() for p in line.split(",")]
                self.entries.append((parts[0], np.array([float(x) for x in parts[1:]], dtype=np.float32)))
        
        # 统计性质用于标准化 (Z-score normalization)
        all_props = np.array([e[1] for e in self.entries])
        self.prop_mean = all_props.mean()
        self.prop_std = all_props.std() + 1e-6

    def __len__(self): 
        return len(self.entries)

    def __getitem__(self, idx):
        cid, props = self.entries[idx]
        norm_props = (props - self.prop_mean) / self.prop_std # 标准化
        
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
    """将不同原子数的晶体打包成一个 Batch"""
    batch_lattice = torch.stack([b['lattice'] for b in batch])
    batch_props = torch.stack([b['props'] for b in batch])
    
    all_fracs = torch.cat([b['fracs'] for b in batch], dim=0)
    all_species = torch.cat([b['species'] for b in batch], dim=0)
    
    batch_indices = []
    for i, b in enumerate(batch):
        batch_indices.extend([i] * b['num_atoms'])
    batch_indices = torch.tensor(batch_indices, dtype=torch.long)
    
    return {
        "lattice": batch_lattice,
        "fracs": all_fracs,
        "species": all_species,
        "props": batch_props,
        "batch_indices": batch_indices,
        "num_atoms": [b['num_atoms'] for b in batch]
    }

# ==========================================
# 2. 增强型模型组件
# ==========================================
class SimpleEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.species_emb = nn.Embedding(100, 32, padding_idx=0)
        self.atom_mlp = nn.Sequential(nn.Linear(32 + 3, 128), nn.ReLU(), nn.Linear(128, 128))
        self.final_mlp = nn.Sequential(nn.Linear(128 + 9, 128), nn.ReLU(), nn.Linear(128, latent_dim * 2))

    def forward(self, lattice, fracs, species, batch_indices):
        atom_feat = self.atom_mlp(torch.cat([self.species_emb(species), fracs], dim=1))
        
        num_graphs = lattice.size(0)
        pooled = torch.zeros(num_graphs, 128, device=lattice.device)
        pooled.index_add_(0, batch_indices, atom_feat)
        atom_counts = torch.bincount(batch_indices).view(-1, 1).float()
        pooled = pooled / atom_counts
        
        combined = torch.cat([pooled, lattice.view(num_graphs, 9)], dim=1)
        params = self.final_mlp(combined)
        mu, logvar = torch.chunk(params, 2, dim=-1)
        return mu, logvar

class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim=128, K=20):
        super().__init__()
        self.K = K
        self.latent_mlp = nn.Sequential(nn.Linear(latent_dim + 1, 128), nn.ReLU(), nn.Linear(128, 128))
        self.lattice_out = nn.Linear(128, 9)
        self.site_embeddings = nn.Parameter(torch.randn(K, 128))
        self.site_mlp = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 3 + 100 + 1))

    def forward(self, z, cond):
        B = z.size(0)
        hid = self.latent_mlp(torch.cat([z, cond], dim=-1)) 
        lat = self.lattice_out(hid).view(B, 3, 3)
        
        hid_expanded = hid.unsqueeze(1).expand(-1, self.K, -1) 
        sites = hid_expanded + self.site_embeddings.unsqueeze(0)
        
        out = self.site_mlp(sites)
        fracs = torch.sigmoid(out[..., :3])
        species_logits = out[..., 3:103]
        occ_logits = out[..., 103]
        return lat, fracs, species_logits, occ_logits

class CondCDVAE(nn.Module):
    def __init__(self, latent_dim=128, K=20):
        super().__init__()
        self.encoder = SimpleEncoder(latent_dim=latent_dim)
        self.decoder = SimpleDecoder(latent_dim=latent_dim, K=K)
        self.latent_dim = latent_dim

# ==========================================
# 3. 批处理损失计算
# ==========================================
def compute_batch_loss(model, batch, device):
    lattice = batch['lattice'].to(device)
    fracs = batch['fracs'].to(device)
    species = batch['species'].to(device)
    props = batch['props'].to(device)
    batch_indices = batch['batch_indices'].to(device)
    
    # Encoder
    mu, logvar = model.encoder(lattice, fracs, species, batch_indices)
    std = (0.5 * logvar).exp()
    z = mu + torch.randn_like(std) * std
    
    # Decoder
    lat_p, fracs_p, spec_p, occ_p = model.decoder(z, props)
    
    total_l_coord, total_l_spec, total_l_occ = 0, 0, 0
    start_idx = 0
    
    # 对 Batch 内每个样本单独计算匈牙利匹配
    for i, n in enumerate(batch['num_atoms']):
        f_true = fracs[start_idx : start_idx + n]
        s_true = species[start_idx : start_idx + n]
        start_idx += n
        
        f_pred = fracs_p[i]
        s_logits_pred = spec_p[i]
        o_logits_pred = occ_p[i]
        
        with torch.no_grad():
            diff = f_pred.unsqueeze(1) - f_true.unsqueeze(0)
            diff = diff - torch.round(diff)
            dist = torch.norm(diff, dim=-1)
            r, c = linear_sum_assignment(dist.cpu().numpy())
            
        diff_matched = f_pred[r] - f_true[c]
        diff_matched = diff_matched - torch.round(diff_matched)
        
        total_l_coord += F.mse_loss(diff_matched, torch.zeros_like(diff_matched))
        total_l_spec += F.cross_entropy(s_logits_pred[r], s_true[c])
        
        target_occ = torch.zeros(model.decoder.K, device=device)
        target_occ[r] = 1.0
        total_l_occ += F.binary_cross_entropy_with_logits(o_logits_pred, target_occ)

    l_lat = F.mse_loss(lat_p, lattice)
    l_kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    
    batch_size = len(batch['num_atoms'])
    loss = (total_l_coord + total_l_spec + total_l_occ) / batch_size + 0.1 * l_lat + 0.01 * l_kl
    return loss

# ==========================================
# 4. 生成与保存工具
# ==========================================
def save_structure_to_cif(lattice: np.ndarray, fracs: np.ndarray, species: np.ndarray, filename: str):
    valid_idx = [i for i, z in enumerate(species) if 0 < z <= 118]
    if not valid_idx:
        print(f"警告: 生成的结构没有有效原子，跳过保存 {filename}")
        return
    symbols = [Element.from_Z(int(species[i])).symbol for i in valid_idx]
    valid_fracs = fracs[valid_idx].tolist()
    struct = Structure(Lattice(lattice), symbols, valid_fracs)
    struct.to(filename=filename)

def sample_and_save(model, target_prop_norm, out_dir, n_samples=5, device="cpu"):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    
    cond = torch.tensor([[target_prop_norm]], dtype=torch.float32).expand(n_samples, -1).to(device)
    z = torch.randn(n_samples, model.latent_dim).to(device) 
    
    print(f"正在 GPU 上并行生成 {n_samples} 个候选材料...")
    with torch.no_grad():
        lat_p, fracs_p, spec_p, occ_p = model.decoder(z, cond)
    
    for i in range(n_samples):
        occ_probs = torch.sigmoid(occ_p[i])
        chosen = occ_probs > 0.5
        
        if not chosen.any():
            top_k_idx = torch.topk(occ_probs, 4).indices
            chosen[top_k_idx] = True
            
        f_np = fracs_p[i][chosen].cpu().numpy()
        s_np = torch.argmax(spec_p[i][chosen], dim=-1).cpu().numpy()
        l_np = lat_p[i].cpu().numpy()
        
        out_path = os.path.join(out_dir, f"gen_target_energy_{i}.cif")
        save_structure_to_cif(l_np, f_np, s_np, out_path)
        print(f" - 已保存: {out_path} (包含 {len(s_np)} 个原子)")

# ==========================================
# 5. 主程序执行流程
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="real_mp_dataset", help="存放 id_prop.csv 和 CIF 文件的文件夹")
    parser.add_argument("--batch_size", type=int, default=32, help="批处理大小")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--target_energy", type=float, default=-3.5, help="生成材料的目标形成能 (eV/atom)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的计算设备: {device}")
    
    # 1. 加载数据
    dataset = CrystalDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # 2. 初始化模型
    model = CondCDVAE(latent_dim=128, K=20).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 3. 训练过程
    print(f"数据加载完成 (总计 {len(dataset)} 个样本)，开始训练...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for batch in loader:
            loss = compute_batch_loss(model, batch, device) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{args.epochs} | Avg Loss: {epoch_loss/len(loader):.4f}")

    # 4. 训练完成后：按指定能量值生成新材料
    print("\n" + "="*40)
    print("--- 训练完成，开始材料逆向设计 (Inverse Design) ---")
    print("="*40)
    
    # 关键步骤：将物理值转换为模型理解的标准化值
    norm_target = (args.target_energy - dataset.prop_mean) / dataset.prop_std
    print(f"设定的目标物理能量: {args.target_energy} eV/atom")
    print(f"映射为模型内部标准化条件值: {norm_target:.4f}\n")
    
    output_directory = "ai_designed_materials"
    
    # 并行生成 10 个候选材料
    sample_and_save(
        model=model, 
        target_prop_norm=norm_target, 
        out_dir=output_directory, 
        n_samples=10, 
        device=device
    )
    
    print(f"\n🎉 逆向设计完成！请用 VESTA 打开 '{output_directory}' 文件夹查看你生成的材料。")