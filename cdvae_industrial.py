import os
import argparse
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from pymatgen.core import Structure, Lattice
from scipy.optimize import linear_sum_assignment

# ==========================================
# 1. 工业级数据加载与批处理 (The Collate Function)
# ==========================================
class CrystalDataset(Dataset):
    def __init__(self, root_dir: str, id_prop_csv: str = "id_prop.csv"):
        self.root_dir = root_dir
        self.entries = []
        with open(os.path.join(root_dir, id_prop_csv), "r") as f:
            for line in f:
                if not line.strip(): continue
                parts = [p.strip() for p in line.split(",")]
                self.entries.append((parts[0], np.array([float(x) for x in parts[1:]], dtype=np.float32)))
        
        # 统计性质用于标准化
        all_props = np.array([e[1] for e in self.entries])
        self.prop_mean = all_props.mean()
        self.prop_std = all_props.std() + 1e-6

    def __len__(self): return len(self.entries)

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
    """
    核心：将不同原子数的晶体打包
    """
    batch_lattice = torch.stack([b['lattice'] for b in batch])
    batch_props = torch.stack([b['props'] for b in batch])
    
    # 将原子特征拼接在一起
    all_fracs = torch.cat([b['fracs'] for b in batch], dim=0)
    all_species = torch.cat([b['species'] for b in batch], dim=0)
    
    # 记录每个原子属于哪个 batch
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
# 2. 增强型模型组件 (支持 Batch 索引)
# ==========================================
class SimpleEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.species_emb = nn.Embedding(100, 32, padding_idx=0)
        self.atom_mlp = nn.Sequential(nn.Linear(32 + 3, 128), nn.ReLU(), nn.Linear(128, 128))
        self.final_mlp = nn.Sequential(nn.Linear(128 + 9, 128), nn.ReLU(), nn.Linear(128, latent_dim * 2))

    def forward(self, lattice, fracs, species, batch_indices):
        # 1. 原子特征
        atom_feat = self.atom_mlp(torch.cat([self.species_emb(species), fracs], dim=1))
        
        # 2. 基于 batch_indices 的 Scatter Mean (全局池化)
        num_graphs = lattice.size(0)
        # 简单的平均池化
        pooled = torch.zeros(num_graphs, 128, device=lattice.device)
        pooled.index_add_(0, batch_indices, atom_feat)
        atom_counts = torch.bincount(batch_indices).view(-1, 1).float()
        pooled = pooled / atom_counts
        
        # 3. 结合晶格信息
        combined = torch.cat([pooled, lattice.view(num_graphs, 9)], dim=1)
        params = self.final_mlp(combined)
        mu, logvar = torch.chunk(params, 2, dim=-1)
        return mu, logvar

# (SimpleDecoder 和 DifferentiableCGCNN 部分保持逻辑一致，但需适配 batch 维度)
# 为了节省篇幅，假设 Decoder 生成每个晶格固定 K=20 个位点进行筛选

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
        hid = self.latent_mlp(torch.cat([z, cond], dim=-1)) # (B, 128)
        lat = self.lattice_out(hid).view(B, 3, 3)
        
        # 为每个 batch 的每个 site 组合特征
        hid_expanded = hid.unsqueeze(1).expand(-1, self.K, -1) # (B, K, 128)
        sites = hid_expanded + self.site_embeddings.unsqueeze(0)
        
        out = self.site_mlp(sites)
        fracs = torch.sigmoid(out[..., :3])
        species_logits = out[..., 3:103]
        occ_logits = out[..., 103]
        return lat, fracs, species_logits, occ_logits

# ==========================================
# 3. 训练逻辑与损失函数 (支持多样本并行)
# ==========================================
def compute_batch_loss(model, surrogate, batch, device):
    lattice = batch['lattice'].to(device)
    fracs = batch['fracs'].to(device)
    species = batch['species'].to(device)
    props = batch['props'].to(device)
    batch_indices = batch['batch_indices'].to(device)
    
    # 1. VAE Forward
    mu, logvar = model.encoder(lattice, fracs, species, batch_indices)
    std = (0.5 * logvar).exp()
    z = mu + torch.randn_like(std) * std
    lat_p, fracs_p, spec_p, occ_p = model.decoder(z, props)
    
    # 2. 计算结构损失 (对 Batch 内每个材料分别计算匹配)
    total_l_coord, total_l_spec, total_l_occ = 0, 0, 0
    start_idx = 0
    for i, n in enumerate(batch['num_atoms']):
        # 提取第 i 个晶体的真实原子
        f_true = fracs[start_idx : start_idx + n]
        s_true = species[start_idx : start_idx + n]
        start_idx += n
        
        # 提取第 i 个预测的原子位点 (K=20)
        f_pred = fracs_p[i]
        s_logits_pred = spec_p[i]
        o_logits_pred = occ_p[i]
        
        # 匈牙利匹配
        with torch.no_grad():
            dist = torch.norm(f_pred.unsqueeze(1) - f_true.unsqueeze(0) - 
                              torch.round(f_pred.unsqueeze(1) - f_true.unsqueeze(0)), dim=-1)
            r, c = linear_sum_assignment(dist.cpu().numpy())
            
        total_l_coord += F.mse_loss(f_pred[r] - f_true[c], torch.zeros_like(f_pred[r]))
        total_l_spec += F.cross_entropy(s_logits_pred[r], s_true[c])
        
        target_occ = torch.zeros(model.decoder.K, device=device)
        target_occ[r] = 1.0
        total_l_occ += F.binary_cross_entropy_with_logits(o_logits_pred, target_occ)

    # 3. 性质指导损失 (Surrogate) - 此处简化为对 Batch 处理
    # 实际应用中需要对 DifferentiableCGCNN 进行对应的 Batch 维度适配
    # 暂以 L_prop = 0 占位，你可以按需接入上一章的 DifferentiableCGCNN
    
    l_kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    
    return (total_l_coord + total_l_spec + total_l_occ) / len(batch['num_atoms']) + 0.01 * l_kl

# ==========================================
# 4. 执行流程
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="real_mp_dataset")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = CrystalDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    model = nn.Module()
    model.encoder = SimpleEncoder().to(device)
    model.decoder = SimpleDecoder(K=20).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"数据加载完成，开始在 {device} 上训练...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            loss = compute_batch_loss(model, None, batch, device)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1} | Loss: {epoch_loss/len(loader):.4f}")
