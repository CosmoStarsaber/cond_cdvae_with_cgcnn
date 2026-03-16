"""
cd_indstrial_v3_physics.py

完整的工业级条件变分自编码器 (CondCDVAE) - 物理增强版
新增特性：
1. [潜在聚类] 引入 Latent Predictor，强制隐空间 z 编码物理属性，解决空间无序问题。
2. [物理松弛] 引入 M3GNet/ASE 结构弛豫，将生成出的粗糙结构拉回到真实的能量极小值点。
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
# 2. 增强型模型组件 (集成 GNN, Predictor)
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
        msg_input = torch.cat([node_feat[edge_src], node_feat[edge_dst], edge_feat], dim=-1)
        messages = self.msg_mlp(msg_input)
        aggregated = torch.zeros_like(node_feat).index_add_(0, edge_dst, messages)
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
            f, lat = fracs[start_idx : start_idx + n], lattice[i]
            diff = f.unsqueeze(1) - f.unsqueeze(0)
            dist_matrix = torch.norm(torch.matmul(diff - torch.round(diff), lat), dim=-1)
            dist_matrix = dist_matrix.masked_fill(torch.eye(n, device=device).bool(), float('inf'))
            
            k = min(self.k_neighbors, n - 1)
            if k > 0:
                topk_dist, topk_idx = torch.topk(dist_matrix, k, dim=-1, largest=False)
                edge_src.append(topk_idx.flatten() + start_idx)
                edge_dst.append(torch.arange(n, device=device).unsqueeze(1).expand(-1, k).flatten() + start_idx)
                edge_dist.append(topk_dist.flatten())
            start_idx += n
        return torch.cat(edge_src), torch.cat(edge_dst), torch.cat(edge_dist).unsqueeze(-1)

    def forward(self, lattice, fracs, species, batch_indices, num_atoms_list):
        node_feat = self.species_emb(species)
        edge_src, edge_dst, edge_dist = self.build_graph(lattice, fracs, num_atoms_list)
        node_feat = self.conv3(self.conv2(self.conv1(node_feat, edge_src, edge_dst, self.distance_expansion(edge_dist)), edge_src, edge_dst, self.distance_expansion(edge_dist)), edge_src, edge_dst, self.distance_expansion(edge_dist))
        
        pooled = torch.zeros(lattice.size(0), node_feat.size(-1), device=lattice.device).index_add_(0, batch_indices, node_feat) / torch.bincount(batch_indices).view(-1, 1).float()
        return torch.chunk(self.final_mlp(torch.cat([pooled, lattice.view(lattice.size(0), 9)], dim=1)), 2, dim=-1)

# 🌟 新增：潜在空间聚类预测器 (Latent Predictor)
class LatentPredictor(nn.Module):
    """迫使隐空间 z 编码物理属性，实现潜在空间的有序聚类"""
    def __init__(self, latent_dim, cond_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, cond_dim)
        )
    def forward(self, z):
        return self.net(z)

class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim=128, cond_dim=1, K=20):
        super().__init__()
        self.K = K
        self.latent_mlp = nn.Sequential(nn.Linear(latent_dim + cond_dim, 128), nn.ReLU(), nn.Linear(128, 128))
        self.lattice_out = nn.Linear(128, 9)
        self.site_embeddings = nn.Parameter(torch.randn(K, 128))
        self.site_mlp = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 3 + 100 + 1))

    def forward(self, z, cond):
        hid = self.latent_mlp(torch.cat([z, cond], dim=-1)) 
        out = self.site_mlp(hid.unsqueeze(1).expand(-1, self.K, -1) + self.site_embeddings.unsqueeze(0))
        return self.lattice_out(hid).view(z.size(0), 3, 3), torch.sigmoid(out[..., :3]), out[..., 3:103], out[..., 103]

class CondCDVAE(nn.Module):
    def __init__(self, latent_dim=128, cond_dim=1, K=20):
        super().__init__()
        self.encoder = GNNEncoder(latent_dim=latent_dim, node_dim=64, edge_dim=64)
        self.predictor = LatentPredictor(latent_dim=latent_dim, cond_dim=cond_dim) # 🌟 挂载预测器
        self.decoder = SimpleDecoder(latent_dim=latent_dim, cond_dim=cond_dim, K=K)
        self.latent_dim = latent_dim

# ==========================================
# 3. 批处理损失计算 (含 Predictor Loss)
# ==========================================
def compute_batch_loss(model, batch, device, r_cut=1.0, lambda_rep=5.0, lambda_pred=1.0):
    lattice, fracs, species, props = batch['lattice'].to(device), batch['fracs'].to(device), batch['species'].to(device), batch['props'].to(device)
    batch_indices, num_atoms_list = batch['batch_indices'].to(device), batch['num_atoms']
    
    mu, logvar = model.encoder(lattice, fracs, species, batch_indices, num_atoms_list)
    std = (0.5 * logvar).exp()
    z = mu + torch.randn_like(std) * std
    
    # 🌟 新增：预测损失计算
    pred_props = model.predictor(z)
    l_predict = F.mse_loss(pred_props, props)
    
    lat_p, fracs_p, spec_p, occ_p = model.decoder(z, props)
    B, K, _ = fracs_p.size()
    total_l_coord, total_l_spec, total_l_occ = 0, 0, 0
    start_idx = 0
    
    for i, n in enumerate(num_atoms_list):
        f_true, s_true = fracs[start_idx : start_idx + n], species[start_idx : start_idx + n]
        start_idx += n
        f_pred, s_logits_pred, o_logits_pred = fracs_p[i], spec_p[i], occ_p[i]
        
        with torch.no_grad():
            diff = f_pred.unsqueeze(1) - f_true.unsqueeze(0)
            r, c = linear_sum_assignment(torch.norm(diff - torch.round(diff), dim=-1).cpu().numpy())
            
        diff_matched = f_pred[r] - f_true[c]
        total_l_coord += F.mse_loss(diff_matched - torch.round(diff_matched), torch.zeros_like(diff_matched))
        total_l_spec += F.cross_entropy(s_logits_pred[r], s_true[c])
        
        target_occ = torch.zeros(K, device=device)
        target_occ[r] = 1.0
        total_l_occ += F.binary_cross_entropy_with_logits(o_logits_pred, target_occ)

    l_lat = F.mse_loss(lat_p, lattice)
    l_kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    
    diff_pred = fracs_p.unsqueeze(2) - fracs_p.unsqueeze(1) 
    dist_pred = torch.norm(torch.bmm((diff_pred - torch.round(diff_pred)).view(B, -1, 3), lat_p).view(B, K, K, 3), dim=-1) 
    dist_pred = dist_pred.masked_fill(torch.eye(K, device=device).unsqueeze(0).bool(), float('inf'))
    
    occ_probs = torch.sigmoid(occ_p) 
    l_repulsion = ((torch.relu(r_cut - dist_pred) ** 2) * (occ_probs.unsqueeze(2) * occ_probs.unsqueeze(1))).sum() / B

    # 🌟 将 l_predict 加入总体损失
    loss = ((total_l_coord + total_l_spec + total_l_occ) / len(num_atoms_list) 
            + 0.1 * l_lat + 0.01 * l_kl 
            + lambda_rep * l_repulsion 
            + lambda_pred * l_predict) 
            
    return loss, l_repulsion.item(), l_predict.item()

# ==========================================
# 4. 生成、保存与结构弛豫 (Relaxation)
# ==========================================
def relax_structure_m3gnet(cif_path):
    """🌟 新增：使用预训练的通用力场对生成的粗糙结构进行能量最小化"""
    try:
        import matgl
        from matgl.ext.ase import Relaxer
        import logging
        logging.getLogger("matgl").setLevel(logging.ERROR) # 屏蔽冗长的日志
    except ImportError:
        print(f" ⚠️ 未安装 matgl 或 ase，跳过物理松弛。")
        return None

    print(f" ⏳ 正在对 {os.path.basename(cif_path)} 进行 M3GNet 分子动力学松弛...")
    try:
        pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        relaxer = Relaxer(potential=pot)
        struct = Structure.from_file(cif_path)
        relax_results = relaxer.relax(struct, fmax=0.05)
        
        final_energy = relax_results['trajectory'].energies[-1]
        relaxed_path = cif_path.replace(".cif", "_relaxed.cif")
        relax_results['final_structure'].to(filename=relaxed_path)
        print(f" ✅ 松弛成功！最终体系能量: {final_energy:.4f} eV，已保存为 {os.path.basename(relaxed_path)}")
        return relaxed_path
    except Exception as e:
        print(f" ❌ 松弛失败 (可能是结构太离谱): {e}")
        return None

def sample_and_save(model, target_props_norm, out_dir, n_samples=5, device="cpu", run_relaxation=True):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    cond = torch.tensor([target_props_norm], dtype=torch.float32).expand(n_samples, -1).to(device)
    z = torch.randn(n_samples, model.latent_dim).to(device) 
    
    with torch.no_grad():
        lat_p, fracs_p, spec_p, occ_p = model.decoder(z, cond)
    
    generated_cifs = []
    for i in range(n_samples):
        occ_probs = torch.sigmoid(occ_p[i])
        chosen = occ_probs > 0.5
        if not chosen.any(): chosen[torch.topk(occ_probs, 4).indices] = True
            
        f_np, s_np, l_np = fracs_p[i][chosen].cpu().numpy(), torch.argmax(spec_p[i][chosen], dim=-1).cpu().numpy(), lat_p[i].cpu().numpy()
        valid_idx = [idx for idx, z_num in enumerate(s_np) if 0 < z_num <= 118]
        if not valid_idx: continue
        
        struct = Structure(Lattice(l_np), [Element.from_Z(int(s_np[idx])).symbol for idx in valid_idx], f_np[valid_idx].tolist())
        out_path = os.path.join(out_dir, f"gen_sample_{i}.cif")
        struct.to(filename=out_path)
        generated_cifs.append(out_path)
        print(f" - 粗糙结构已生成: {out_path} (包含 {len(valid_idx)} 个原子)")
        
    if run_relaxation:
        print("\n--- 启动后期物理验证 (Structure Relaxation) ---")
        for cif in generated_cifs:
            relax_structure_m3gnet(cif)

# ==========================================
# 5. 主程序执行流程
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="real_mp_dataset")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--target_props", type=float, nargs='+', default=[-2.0, 1.5])
    parser.add_argument("--r_cut", type=float, default=1.0)
    parser.add_argument("--lambda_rep", type=float, default=5.0)
    parser.add_argument("--lambda_pred", type=float, default=1.0, help="Latent Predictor 损失权重")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--skip_relax", action="store_true", help="如果加上此参数，将跳过生成后的 M3GNet 松弛过程")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用的计算设备: {device}")
    
    dataset = CrystalDataset(args.data)
    cond_dim = dataset.cond_dim
    
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    model = CondCDVAE(latent_dim=128, cond_dim=cond_dim, K=20).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    os.makedirs(args.save_dir, exist_ok=True)
    best_model_path = os.path.join(args.save_dir, "best_cond_cdvae.pt")

    print("\n" + "="*40 + "\n🔥 开始训练 (包含 Predictor 聚类与 Early Stopping)\n" + "="*40)
    best_val_loss, epochs_no_improve = float('inf'), 0

    for epoch in range(args.epochs):
        model.train()
        train_loss, train_rep, train_pred = 0, 0, 0
        for batch in train_loader:
            loss, rep_v, pred_v = compute_batch_loss(model, batch, device, args.r_cut, args.lambda_rep, args.lambda_pred) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item(); train_rep += rep_v; train_pred += pred_v
            
        model.eval()
        val_loss, val_rep, val_pred = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                loss, rep_v, pred_v = compute_batch_loss(model, batch, device, args.r_cut, args.lambda_rep, args.lambda_pred)
                val_loss += loss.item(); val_rep += rep_v; val_pred += pred_v

        a_tr_l, a_tr_p = train_loss/len(train_loader), train_pred/len(train_loader)
        a_va_l, a_va_p = val_loss/len(val_loader), val_pred/len(val_loader)

        print(f"Ep [{epoch+1:03d}] | Tr Loss: {a_tr_l:.3f} (预测聚类:{a_tr_p:.3f}) | Val Loss: {a_va_l:.3f} (预测聚类:{a_va_p:.3f})")

        if a_va_l < best_val_loss:
            best_val_loss, epochs_no_improve = a_va_l, 0
            torch.save(model.state_dict(), best_model_path)
            print(f"   🌟 最佳模型保存！")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\n🛑 早停触发！连续 {args.patience} 个 Epoch 验证集无提升。")
                break

    print("\n" + "="*40 + "\n--- 训练完成，开始逆向设计与物理验证 ---\n" + "="*40)
    model.load_state_dict(torch.load(best_model_path))
    targets = (args.target_props + [0.0] * cond_dim)[:cond_dim]
    norm_targets = (np.array(targets, dtype=np.float32) - dataset.prop_mean) / dataset.prop_std
    
    sample_and_save(model, norm_targets.tolist(), "ai_designed_materials", n_samples=3, device=device, run_relaxation=not args.skip_relax)