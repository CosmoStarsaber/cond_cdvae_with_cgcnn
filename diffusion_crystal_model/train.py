"""
train.py

晶体扩散模型 (Diffusion-CDVAE) 生产级完全体
集成 WandB 日志监控、CFG/温度控制采样、断点续训以及定期生成策略。
"""

import os
import argparse
import warnings
import numpy as np
import torch
import wandb
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from pymatgen.core import Structure, Lattice
from pymatgen.core.periodic_table import Element

# 导入终极版的主架构 (已内置 CGCNN Encoder)
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
# 2. CFG 条件采样与潜空间梯度寻优
# ==========================================
def save_structure_to_cif(lattice, fracs, species, filename):
    valid_idx = [i for i, z_num in enumerate(species) if 0 < z_num <= 118]
    if not valid_idx: return
    symbols = [Element.from_Z(int(species[i])).symbol for i in valid_idx]
    struct = Structure(Lattice(lattice), symbols, fracs[valid_idx].tolist())
    struct.to(filename=filename)

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
    for i, n in enumerate(num_atoms_list):
        f, s, l = fracs_np[start_idx : start_idx + n], species[start_idx : start_idx + n], lattice_np[i]
        start_idx += n
        out_path = os.path.join(out_dir, f"{prefix}sample_{i}.cif")
        save_structure_to_cif(l, f, s, out_path)
    
    print(f"✅ 生成完成，已保存至 {out_dir}")

# ==========================================
# 3. 主程序执行 (生产级流水线)
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据与环境
    parser.add_argument("--data", type=str, default="real_mp_dataset")
    parser.add_argument("--batch_size", type=int, default=16) 
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--timesteps", type=int, default=200)
    
    # 监控与保存
    parser.add_argument("--wandb_project", type=str, default="crystal-diffusion-cgcnn", help="WandB 项目名称")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="模型保存路径")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的权重路径")
    parser.add_argument("--sample_every", type=int, default=50, help="每 N 轮采样一次示例，设为 0 则仅在结束时采样")
    
    # 物理条件采样超参
    parser.add_argument("--target_props", type=float, nargs='+', default=[-2.0, 1.5])
    parser.add_argument("--guidance_scale", type=float, default=2.0, help="CFG 引导强度 (越大越贴近条件，但多样性降低)")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度 (小于1.0可使结构更稳定，但容易陷入局部最优)")
    args = parser.parse_args()

    # 初始化 WandB
    wandb.init(project=args.wandb_project, config=vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动生产级晶体扩散框架！计算设备: {device}")
    
    dataset = CrystalDataset(args.data)
    cond_dim = dataset.cond_dim
    
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 初始化内聚好的 DiffusionCDVAE
    model = DiffusionCDVAE(latent_dim=128, cond_dim=cond_dim, timesteps=args.timesteps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    os.makedirs(args.save_dir, exist_ok=True)
    
    # 断点续训逻辑
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
        # ---------------- 训练阶段 ----------------
        model.train()
        train_metrics = {"loss_total": 0, "loss_diff": 0, "loss_prop": 0, "loss_species": 0}
        for batch in train_loader:
            loss, logs = model.compute_loss(batch, device) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for k in train_metrics: train_metrics[k] += logs[k]
            
        for k in train_metrics: train_metrics[k] /= len(train_loader)

        # ---------------- 验证阶段 ----------------
        model.eval()
        val_metrics = {"loss_total": 0, "loss_diff": 0, "loss_prop": 0, "loss_species": 0}
        with torch.no_grad():
            for batch in val_loader:
                loss, logs = model.compute_loss(batch, device)
                for k in val_metrics: val_metrics[k] += logs[k]
                
        for k in val_metrics: val_metrics[k] /= len(val_loader)

        # ---------------- 日志与保存 ----------------
        print(f"Ep [{epoch+1:03d}/{args.epochs}] | Tr Diff: {train_metrics['loss_diff']:.3f} | Val Diff: {val_metrics['loss_diff']:.3f} (Prop: {val_metrics['loss_prop']:.3f}, Spec: {val_metrics['loss_species']:.3f})")
        
        # 提交到 WandB 面板
        wandb.log({
            "train/total_loss": train_metrics["loss_total"],
            "train/diff_loss": train_metrics["loss_diff"],
            "val/total_loss": val_metrics["loss_total"],
            "val/diff_loss": val_metrics["loss_diff"],
            "val/prop_loss": val_metrics["loss_prop"],
            "val/species_loss": val_metrics["loss_species"],
            "epoch": epoch
        })

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
            
        # 定期中间采样检查点 (避免每轮都画图)
        if args.sample_every > 0 and (epoch + 1) % args.sample_every == 0:
            norm_targets = (np.array(args.target_props[:cond_dim], dtype=np.float32) - dataset.prop_mean) / dataset.prop_std
            generate_diffusion_crystals(
                model, norm_targets.tolist(), "generated_cifs", n_samples=2, device=device, 
                guidance_scale=args.guidance_scale, temperature=args.temperature, epoch=epoch+1
            )

    # ---------------- 最终生成 ----------------
    print("\n" + "="*50 + "\n🔥 训练完毕，加载最佳权重进行最终物理条件采样\n" + "="*50)
    best_path = os.path.join(args.save_dir, "best_checkpoint.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device)['model_state_dict'])
        
    targets = (args.target_props + [0.0] * cond_dim)[:cond_dim]
    norm_targets = (np.array(targets, dtype=np.float32) - dataset.prop_mean) / dataset.prop_std
    
    generate_diffusion_crystals(
        model, norm_targets.tolist(), "generated_cifs", n_samples=5, device=device, 
        guidance_scale=args.guidance_scale, temperature=args.temperature, epoch=None
    )
    wandb.finish()