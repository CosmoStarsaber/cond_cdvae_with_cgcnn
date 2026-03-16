"""
cd_indstrial_multi_cond.py

完整的工业级条件变分自编码器 (CondCDVAE) - 多条件约束版
特性：
1. [多条件] 支持任意数量的条件输入 (如 能量, 带隙, 体积等)，自动识别维度
2. [数据] 独立对每一个维度的属性进行精确的 Z-score 标准化
3. [物理] 包含 GNN 图编码器与原子防重叠惩罚机制 (Repulsion Loss)
4. [工程] 支持动态批处理 (Dynamic Batching)
"""

import os
import argparse
import warnings
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from pymatgen.core import Structure, Lattice
from pymatgen.core.periodic_table import Element
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")

# ==========================================
# 1. 工业级数据加载与多维特征标准化
# ==========================================
class CrystalDataset(Dataset):
    def __init__(self, root_dir: str, id_prop_csv: str = "id_prop.csv"):
        self.root_dir = root_dir
        self.entries = []
        csv_path = os.path.join(root_dir, id_prop_csv)
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到 {csv_path}。请确保 CSV 文件存在。")

        with open(csv_path, "r") as f:
            for line in f:
                if not line.strip(): continue
                parts = [p.strip() for p in line.split(",")]
                # 支持多个属性，解析为 Numpy 数组
                self.entries.append((parts[0], np.array([float(x) for x in parts[1:]], dtype=np.float32)))
        
        # 🌟 获取条件维度数量 (比如 2 表示有能量和带隙两个参数)
        all_props = np.array([e[1] for e in self.entries])
        self.cond_dim = all_props.shape[1] 
        
        # 🌟 分别计算每一个维度的均值和标准差
        self.prop_mean = all_props.mean(axis=0) # shape: (cond_dim,)
        self.prop_std = all_props.std(axis=0) + 1e-6 # shape: (cond_dim,)

    def __len__(self): 
        return len(self.entries)

    def __getitem__(self, idx):
        cid, props = self.entries[idx]
        # 对向量进行标准化 (Broadcasting)
        norm_props = (props - self.prop_mean) / self.prop_std 
        
        struct = Structure.from_file(os.path.join(self.root_dir, f"{cid}.cif"))
        fracs = np.array([s.frac_coords for s in struct])
        fracs = (fracs - np.floor(fracs)).astype(np.float32)
        species = np.array([s.specie.Z for s in struct], dtype=np.int64)
        
        return {
            "lattice": torch.tensor(struct.lattice.matrix, dtype=torch.float32),
            "fracs": torch.tensor(fracs, dtype=torch.float32),
            "species": torch.tensor(species, dtype=torch.long),
            "props": torch.tensor(norm_props, dtype=torch.float32), # Shape: (cond_dim,)
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
# 2. 增强型模型组件 (动态适配 cond_dim)
# ==========================================
class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=8.0, num_gaussians=64):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        return torch.exp(self.coeff * torch.pow(dist - self.offset.view(1, -1), 2))

class MessagePassingLayer(nn.Module):
    def __init__(self, node_dim=64, edge_dim=64):
        super().__init__()
        self.msg_mlp = nn.Sequential(nn.Linear(2 * node_dim + edge_dim, node_dim), nn.SiLU(), nn.Linear(node_dim, node_dim))
        self.update_mlp = nn.Sequential(nn.Linear(node_dim, node_dim), nn.SiLU(), nn.Linear(node_dim, node_dim))
        self.layer_norm = nn.LayerNorm(node_dim)

    def forward(self, node_feat, edge_src, edge_dst, edge_feat):
        src_feat = node_feat[edge_src]
        dst_feat = node_feat[edge_dst]
        msg_input = torch.cat([src_feat, dst_feat, edge_feat], dim=-1)
        messages = self.msg_mlp(msg_input)
        
        aggregated = torch.zeros_like(node_feat)
        aggregated.index_add_(0, edge_dst, messages)
        return self.layer_norm(node_feat + self.update_mlp(aggregated))

class GNNEncoder(nn.Module):
    def __init__(self, latent_dim=128, node_dim=64, edge_dim=64, k_neighbors=12):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.species_emb = nn.Embedding(100, node_dim, padding_idx=0)
        self.distance_expansion = GaussianSmearing(start=0.0, stop=8.0, num_gaussians=edge_dim)
        
        self.conv1 = MessagePassingLayer(node_dim, edge_dim)
        self.conv2 = MessagePassingLayer(node_dim, edge_dim)
        self.conv3 = MessagePassingLayer(node_dim, edge_dim)
        
        self.final_mlp = nn.Sequential(nn.Linear(node_dim + 9, 128), nn.SiLU(), nn.Linear(128, latent_dim * 2))

    def build_graph(self, lattice, fracs, num_atoms_list):
        edge_src, edge_dst, edge_dist = [], [], []
        start_idx = 0
        device = lattice.device
        
        for i, n in enumerate(num_atoms_list):
            f = fracs[start_idx : start_idx + n]
            lat = lattice[i]
            
            diff = f.unsqueeze(1) - f.unsqueeze(0)
            diff = diff - torch.round(diff)
            cart = torch.matmul(diff, lat)
            dist_matrix = torch.norm(cart, dim=-1)
            
            diag_mask = torch.eye(n, device=device).bool()
            dist_matrix = dist_matrix.masked_fill(diag_mask, float('inf'))
            
            k = min(self.k_neighbors, n - 1)
            if k > 0:
                topk_dist, topk_idx = torch.topk(dist_matrix, k, dim=-1, largest=False)
                src = topk_idx.flatten() + start_idx
                dst = torch.arange(n, device=device).unsqueeze(1).expand(-1, k).flatten() + start_idx
                edge_src.append(src)
                edge_dst.append(dst)
                edge_dist.append(topk_dist.flatten())
                
            start_idx += n
        return torch.cat(edge_src), torch.cat(edge_dst), torch.cat(edge_dist).unsqueeze(-1)

    def forward(self, lattice, fracs, species, batch_indices, num_atoms_list):
        node_feat = self.species_emb(species)
        edge_src, edge_dst, edge_dist = self.build_graph(lattice, fracs, num_atoms_list)
        edge_feat = self.distance_expansion(edge_dist)
        
        node_feat = self.conv1(node_feat, edge_src, edge_dst, edge_feat)
        node_feat = self.conv2(node_feat, edge_src, edge_dst, edge_feat)
        node_feat = self.conv3(node_feat, edge_src, edge_dst, edge_feat)
        
        num_graphs = lattice.size(0)
        pooled = torch.zeros(num_graphs, node_feat.size(-1), device=lattice.device)
        pooled.index_add_(0, batch_indices, node_feat)
        atom_counts = torch.bincount(batch_indices).view(-1, 1).float()
        pooled = pooled / atom_counts
        
        combined = torch.cat([pooled, lattice.view(num_graphs, 9)], dim=1)
        params = self.final_mlp(combined)
        return torch.chunk(params, 2, dim=-1)

class SimpleDecoder(nn.Module):
    # 🌟 修改了构造函数，接收 cond_dim 决定条件输入的宽度
    def __init__(self, latent_dim=128, cond_dim=1, K=20):
        super().__init__()
        self.K = K
        self.latent_mlp = nn.Sequential(nn.Linear(latent_dim + cond_dim, 128), nn.ReLU(), nn.Linear(128, 128))
        self.lattice_out = nn.Linear(128, 9)
        self.site_embeddings = nn.Parameter(torch.randn(K, 128))
        self.site_mlp = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 3 + 100 + 1))

    def forward(self, z, cond):
        B = z.size(0)
        hid = self.latent_mlp(torch.cat([z, cond], dim=-1)) # 拼接 z 和条件向量
        lat = self.lattice_out(hid).view(B, 3, 3)
        hid_expanded = hid.unsqueeze(1).expand(-1, self.K, -1) 
        sites = hid_expanded + self.site_embeddings.unsqueeze(0)
        
        out = self.site_mlp(sites)
        fracs = torch.sigmoid(out[..., :3])
        species_logits = out[..., 3:103]
        occ_logits = out[..., 103]
        return lat, fracs, species_logits, occ_logits

class CondCDVAE(nn.Module):
    # 🌟 将 cond_dim 传递给 Decoder
    def __init__(self, latent_dim=128, cond_dim=1, K=20):
        super().__init__()
        self.encoder = GNNEncoder(latent_dim=latent_dim, node_dim=64, edge_dim=64)
        self.decoder = SimpleDecoder(latent_dim=latent_dim, cond_dim=cond_dim, K=K)
        self.latent_dim = latent_dim

# ==========================================
# 3. 批处理损失计算 
# ==========================================
def compute_batch_loss(model, batch, device, r_cut=1.0, lambda_rep=5.0):
    lattice = batch['lattice'].to(device)
    fracs = batch['fracs'].to(device)
    species = batch['species'].to(device)
    props = batch['props'].to(device) # Shape is now (B, cond_dim)
    batch_indices = batch['batch_indices'].to(device)
    num_atoms_list = batch['num_atoms']
    
    mu, logvar = model.encoder(lattice, fracs, species, batch_indices, num_atoms_list)
    std = (0.5 * logvar).exp()
    z = mu + torch.randn_like(std) * std
    lat_p, fracs_p, spec_p, occ_p = model.decoder(z, props)
    
    B, K, _ = fracs_p.size()
    total_l_coord, total_l_spec, total_l_occ = 0, 0, 0
    start_idx = 0
    
    for i, n in enumerate(num_atoms_list):
        f_true = fracs[start_idx : start_idx + n]
        s_true = species[start_idx : start_idx + n]
        start_idx += n
        
        f_pred, s_logits_pred, o_logits_pred = fracs_p[i], spec_p[i], occ_p[i]
        
        with torch.no_grad():
            diff = f_pred.unsqueeze(1) - f_true.unsqueeze(0)
            diff = diff - torch.round(diff)
            dist = torch.norm(diff, dim=-1)
            r, c = linear_sum_assignment(dist.cpu().numpy())
            
        diff_matched = f_pred[r] - f_true[c]
        diff_matched = diff_matched - torch.round(diff_matched)
        
        total_l_coord += F.mse_loss(diff_matched, torch.zeros_like(diff_matched))
        total_l_spec += F.cross_entropy(s_logits_pred[r], s_true[c])
        
        target_occ = torch.zeros(K, device=device)
        target_occ[r] = 1.0
        total_l_occ += F.binary_cross_entropy_with_logits(o_logits_pred, target_occ)

    l_lat = F.mse_loss(lat_p, lattice)
    l_kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    
    diff_pred = fracs_p.unsqueeze(2) - fracs_p.unsqueeze(1) 
    diff_pred = diff_pred - torch.round(diff_pred) 
    cart_pred = torch.bmm(diff_pred.view(B, -1, 3), lat_p).view(B, K, K, 3)
    dist_pred = torch.norm(cart_pred, dim=-1) 
    
    mask = torch.eye(K, device=device).unsqueeze(0).bool()
    dist_pred = dist_pred.masked_fill(mask, float('inf'))
    
    repulsion = torch.relu(r_cut - dist_pred) ** 2 
    occ_probs = torch.sigmoid(occ_p) 
    weight_matrix = occ_probs.unsqueeze(2) * occ_probs.unsqueeze(1) 
    l_repulsion = (repulsion * weight_matrix).sum() / B

    batch_size = len(num_atoms_list)
    loss = ((total_l_coord + total_l_spec + total_l_occ) / batch_size 
            + 0.1 * l_lat 
            + 0.01 * l_kl 
            + lambda_rep * l_repulsion) 
            
    return loss, l_repulsion.item()

# ==========================================
# 4. 生成与保存工具 (支持多属性向量输入)
# ==========================================
def save_structure_to_cif(lattice: np.ndarray, fracs: np.ndarray, species: np.ndarray, filename: str):
    valid_idx = [i for i, z in enumerate(species) if 0 < z <= 118]
    if not valid_idx: return
    symbols = [Element.from_Z(int(species[i])).symbol for i in valid_idx]
    struct = Structure(Lattice(lattice), symbols, fracs[valid_idx].tolist())
    struct.to(filename=filename)

def sample_and_save(model, target_props_norm, out_dir, n_samples=5, device="cpu"):
    """
    target_props_norm: 形状为 (cond_dim,) 的列表或数组
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    
    # 🌟 将多维属性列表转换为张量，并扩充到 Batch 大小
    cond = torch.tensor([target_props_norm], dtype=torch.float32).expand(n_samples, -1).to(device)
    z = torch.randn(n_samples, model.latent_dim).to(device) 
    
    print(f"正在 {device.type.upper()} 上并行生成 {n_samples} 个候选材料...")
    with torch.no_grad():
        lat_p, fracs_p, spec_p, occ_p = model.decoder(z, cond)
    
    for i in range(n_samples):
        occ_probs = torch.sigmoid(occ_p[i])
        chosen = occ_probs > 0.5
        if not chosen.any():
            chosen[torch.topk(occ_probs, 4).indices] = True
            
        f_np = fracs_p[i][chosen].cpu().numpy()
        s_np = torch.argmax(spec_p[i][chosen], dim=-1).cpu().numpy()
        l_np = lat_p[i].cpu().numpy()
        
        out_path = os.path.join(out_dir, f"gen_sample_{i}.cif")
        save_structure_to_cif(l_np, f_np, s_np, out_path)
        print(f" - 已保存: {out_path} (包含 {len(s_np)} 个原子)")

# ==========================================
# 5. 主程序执行流程
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="real_mp_dataset")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    
    # 🌟 关键修改：允许传入用空格分隔的多个数值
    parser.add_argument("--target_props", type=float, nargs='+', default=[-3.5], 
                        help="输入的多个目标属性值，请用空格隔开。例如：-3.5 1.5 8.2")
    
    parser.add_argument("--r_cut", type=float, default=1.0)
    parser.add_argument("--lambda_rep", type=float, default=5.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的计算设备: {device}")
    
    # 1. 加载数据并自动获取属性维度
    dataset = CrystalDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    cond_dim = dataset.cond_dim
    print(f"检测到数据集包含 {cond_dim} 个维度的目标属性。")
    
    # 2. 初始化模型，传入动态的 cond_dim
    model = CondCDVAE(latent_dim=128, cond_dim=cond_dim, K=20).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"数据加载完成 (总计 {len(dataset)} 个样本)，开始训练...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_rep_loss = 0
        for batch in loader:
            loss, rep_loss_val = compute_batch_loss(model, batch, device, r_cut=args.r_cut, lambda_rep=args.lambda_rep) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_rep_loss += rep_loss_val
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{args.epochs} | 总 Loss: {epoch_loss/len(loader):.4f} | 物理斥力惩罚: {epoch_rep_loss/len(loader):.4f}")

    print("\n" + "="*40)
    print("--- 训练完成，开始多条件逆向设计 (Multi-conditional Inverse Design) ---")
    print("="*40)
    
    # 校验用户输入的参数数量是否与数据集对齐
    if len(args.target_props) != cond_dim:
        print(f"⚠️ 警告: 你输入了 {len(args.target_props)} 个参数，但数据集需要 {cond_dim} 个！")
        print(f"模型将强行使用默认值或截断输入来对齐...")
        # 补齐或截断到 cond_dim 长度
        targets = (args.target_props + [0.0] * cond_dim)[:cond_dim]
    else:
        targets = args.target_props
        
    print(f"设定的目标物理属性向量: {targets}")
    
    # 🌟 多维 Z-score 标准化 (向量运算)
    targets_np = np.array(targets, dtype=np.float32)
    norm_targets = (targets_np - dataset.prop_mean) / dataset.prop_std
    
    print(f"映射为内部标准化条件向量: {np.round(norm_targets, 4)}\n")
    
    output_directory = "ai_designed_materials"
    sample_and_save(model, norm_targets.tolist(), output_directory, n_samples=10, device=device)
    print(f"\n🎉 逆向设计完成！请用 VESTA 打开 '{output_directory}' 文件夹查看你生成的材料。")