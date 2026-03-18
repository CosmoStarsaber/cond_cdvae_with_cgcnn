"""
train.py

晶体扩散模型 (Diffusion-CDVAE) 终极生产级流水线
特性：
1. 本地纯净运行 (无断网报错风险)
2. 内存级硬截断 (生成时直接过滤原子碰撞的废片，不写盘)
3. 断点续训与定期条件采样
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

# 导入终极版的主架构 (已内置 CGCNN Encoder 和 斥力惩罚)
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
            raise FileNotFoundError(f"找不到 {csv_path}。请检查数据集路径。")

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
# 2. 带有硬截断质检的 CFG 采样生成
# ==========================================
@torch.no_grad()
def generate_diffusion_crystals(model, target_props_norm, out_dir, n_samples=5, device="cpu", guidance_scale=2.0, temperature=1.0, epoch=None):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    
    prefix = f"ep{epoch}_" if epoch is not None else "final_"
    print(f"\n🔮 [{prefix.strip('_')}] 潜空间梯度寻优...")
    
    z = torch.randn(n_samples, model.latent_dim, device=device, requires_grad=True)
    cond_target = torch.tensor([target_props_norm], dtype=torch.float32).expand(n_samples, -1).to(device)
    
    optimizer_z = torch.optim.Adam([z], lr=0.05)
    for _ in range(100):
        with torch.enable_grad():
            loss_z = F.mse_loss(model.property_predictor(z), cond_target)
            optimizer_z.zero_grad()
            loss_z.backward()
            optimizer_z.step()
    
    z = z.detach()
    print(f"📐 宏观结构预测...")
    num_atoms_logits = model.length_predictor(z)
    num_atoms_list = torch.argmax(num_atoms_logits, dim=-1).clamp(min=1).tolist()
    lattice = model.lattice_predictor(z)
    
    batch_indices = torch.tensor([i for i, n in enumerate(num_atoms_list) for _ in range(n)], device=device)
    z_nodes = z[batch_indices]
    
    print(f"🌀 朗之万去噪 (CFG={guidance_scale}, Temp={temperature})...")
    frac_coords, species_logits = model.decoder.sample(
        z_nodes, lattice, num_atoms_list, batch_indices, 
        guidance_scale=guidance_scale, temperature=temperature
    )
    
    species = torch.argmax(species_logits, dim=-1).cpu().numpy()
    fracs_np = frac_coords.cpu().numpy()
    lattice_np = lattice.cpu().numpy()
    
    start_idx = 0
    valid_count = 0
    for i, n in enumerate(num_atoms_list):
        f, s, l = fracs_np[start_idx : start_idx + n], species[start_idx : start_idx + n], lattice_np[i]
        start_idx += n
        
        # --- 🌟 核心硬截断：内存级物理质检 ---
        valid_idx = [j for j, z_num in enumerate(s) if 0 < z_num <= 118]
        if len(valid_idx) < 1:
            print(f"   🚫 废片拦截: 样本 {i} 无有效化学元素。")
            continue
            
        try:
            symbols = [Element.from_Z(int(s[j])).symbol for j in valid_idx]
            struct = Structure(Lattice(l), symbols, f[valid_idx].tolist())
            
            # 使用 get_all_neighbors 计算包含周期性边界的真实物理距离 (阈值 0.8 Å)
            if len(struct) > 1:
                neighbors = struct.get_all_neighbors(r=0.8)
                has_collision = any(len(neigh) > 0 for neigh in neighbors)
                
                if has_collision:
                    print(f"   🚫 废片拦截: 样本 {i} 发生原子坍缩重叠，已在内存中销毁！")
                    continue
            
            # 通过质检，写入硬盘
            out_path = os.path.join(out_dir, f"{prefix}sample_{i}.cif")
            struct.to(filename=out_path)
            valid_count += 1
            print(f"   ✅ 生成成功: {out_path} (原子数: {len(valid_idx)})")
            
        except Exception as e:
            print(f"   🚫 废片拦截: 样本 {i} 晶格畸变严重无法解析 ({e})")
            
    print(f"🎯 本批次生成结束，存活率: {valid_count}/{n_samples}\n")

# ==========================================
# 3. 主程序执行 (生产级流水线)
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据与环境
    parser.add_argument("--data", type=str, default="real_mp_dataset")
    parser.add_argument("--batch_size", type=int, default=16) 
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--timesteps", type=int, default=200)
    
    # 监控与保存
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="模型保存路径")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的权重路径")
    parser.add_argument("--sample_every", type=int, default=50, help="每 N 轮采样一次示例，设为 0 则仅在结束时采样")
    
    # 物理条件采样超参
    parser.add_argument("--target_props", type=float, nargs='+', default=[-2.0, 1.5])
    parser.add_argument("--guidance_scale", type=float, default=2.0, help="CFG 引导强度")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动纯净生产级晶体扩散框架！计算设备: {device}")
    
    dataset = CrystalDataset(args.data)
    cond_dim = dataset.cond_dim
    
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 确保 diffusion_cdvae.py 已经包含了我们上一把加进去的斥力惩罚
    model = DiffusionCDVAE(latent_dim=128, cond_dim=cond_dim, timesteps=args.timesteps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    os.makedirs(args.save_dir, exist_ok=True)
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume and os.path.exists(args.resume):
        print(f"🔄 正在恢复训练状态: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"✅ 从 Epoch {start_epoch} 继续 (当前最佳 Val Loss: {best_val_loss:.4f})")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_metrics = {"loss_total": 0, "loss_diff": 0, "loss_prop": 0, "loss_species": 0, "loss_rep": 0}
        for batch in train_loader:
            loss, logs = model.compute_loss(batch, device) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for k in train_metrics: 
                if k in logs: train_metrics[k] += logs[k]
            
        for k in train_metrics: train_metrics[k] /= len(train_loader)

        model.eval()
        val_metrics = {"loss_total": 0, "loss_diff": 0, "loss_prop": 0, "loss_species": 0, "loss_rep": 0}
        with torch.no_grad():
            for batch in val_loader:
                loss, logs = model.compute_loss(batch, device)
                for k in val_metrics: 
                    if k in logs: val_metrics[k] += logs[k]
                
        for k in val_metrics: val_metrics[k] /= len(val_loader)

        # 🌟 终端现在会清楚地打印出排斥力 (Rep) 的下降趋势！
        print(f"Ep [{epoch+1:03d}/{args.epochs}] | Tr Diff: {train_metrics['loss_diff']:.3f} (Rep: {train_metrics['loss_rep']:.3f}) | Val Diff: {val_metrics['loss_diff']:.3f} (Rep: {val_metrics['loss_rep']:.3f}, Spec: {val_metrics['loss_species']:.3f}) | Total: {val_metrics['loss_total']:.3f}")

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint_data, os.path.join(args.save_dir, "latest_checkpoint.pt"))
        
        if val_metrics['loss_total'] < best_val_loss:
            best_val_loss = val_metrics['loss_total']
            torch.save(checkpoint_data, os.path.join(args.save_dir, "best_checkpoint.pt"))
            
        if args.sample_every > 0 and (epoch + 1) % args.sample_every == 0:
            norm_targets = (np.array(args.target_props[:cond_dim], dtype=np.float32) - dataset.prop_mean) / dataset.prop_std
            generate_diffusion_crystals(
                model, norm_targets.tolist(), "generated_cifs", n_samples=3, device=device, 
                guidance_scale=args.guidance_scale, temperature=args.temperature, epoch=epoch+1
            )

    print("\n" + "="*50 + "\n🔥 训练完毕，加载最佳权重进行最终物理条件采样\n" + "="*50)
    best_path = os.path.join(args.save_dir, "best_checkpoint.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device)['model_state_dict'])
        
    targets = (args.target_props + [0.0] * cond_dim)[:cond_dim]
    norm_targets = (np.array(targets, dtype=np.float32) - dataset.prop_mean) / dataset.prop_std
    
    generate_diffusion_crystals(
        model, norm_targets.tolist(), "generated_cifs", n_samples=10, device=device, 
        guidance_scale=args.guidance_scale, temperature=args.temperature, epoch=None
    )