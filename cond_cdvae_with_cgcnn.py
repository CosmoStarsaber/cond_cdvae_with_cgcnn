"""
cond_cdvae_with_cgcnn.py

A complete prototype of a Conditional Crystal VAE with a Fully Differentiable CGCNN Surrogate.
Key Features:
- End-to-end differentiable graph construction (maintains gradients for coords and lattice).
- Differentiable property guidance using a simplified CGCNN.
- Auto-generation of dummy datasets for instant testing.
"""

import os
import argparse
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from pymatgen.core import Structure, Lattice
from scipy.optimize import linear_sum_assignment

# ==========================================
# 1. Utility & Dummy Data Generation
# ==========================================
def wrap_frac(frac):
    return frac - torch.floor(frac) if isinstance(frac, torch.Tensor) else frac - np.floor(frac)

def save_structure_to_cif(lattice: np.ndarray, fracs: np.ndarray, species: np.ndarray, filename: str):
    from pymatgen.core.periodic_table import Element
    valid_idx = [i for i, z in enumerate(species) if 0 < z <= 118]
    if not valid_idx:
        return
    symbols = [Element.from_Z(int(species[i])).symbol for i in valid_idx]
    valid_fracs = fracs[valid_idx].tolist()
    struct = Structure(Lattice(lattice), symbols, valid_fracs)
    struct.to(filename=filename)

def create_dummy_dataset(root_dir="dummy_data"):
    """Creates a mock dataset so the script can be tested out-of-the-box."""
    os.makedirs(root_dir, exist_ok=True)
    csv_path = os.path.join(root_dir, "id_prop.csv")
    
    # Create BCC Iron
    fe = Structure(Lattice.cubic(2.866), ["Fe", "Fe"], [[0,0,0], [0.5,0.5,0.5]])
    fe.to(filename=os.path.join(root_dir, "struct_1.cif"))
    
    # Create Rock Salt NaCl
    nacl = Structure(Lattice.cubic(5.64), ["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl"], 
                     [[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5], 
                      [0.5,0.5,0.5], [0,0,0.5], [0,0.5,0], [0.5,0,0]])
    nacl.to(filename=os.path.join(root_dir, "struct_2.cif"))

    with open(csv_path, "w") as f:
        f.write("struct_1, 1.5\n") # id, mock_property (e.g., formation energy)
        f.write("struct_2, -3.2\n")
    print(f"Created dummy dataset in '{root_dir}/'")

# ==========================================
# 2. Dataset & VAE Encoder/Decoder
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

    def __len__(self): return len(self.entries)

    def __getitem__(self, idx):
        cid, props = self.entries[idx]
        struct = Structure.from_file(os.path.join(self.root_dir, f"{cid}.cif"))
        fracs = np.array([s.frac_coords for s in struct])
        return {
            "id": cid,
            "lattice": struct.lattice.matrix.astype(np.float32),
            "fracs": (fracs - np.floor(fracs)).astype(np.float32),
            "species": np.array([s.specie.Z for s in struct], dtype=np.int64),
            "props": props
        }

class SimpleEncoder(nn.Module):
    def __init__(self, species_emb_dim=32, atom_feat_dim=128, latent_dim=128):
        super().__init__()
        self.species_emb = nn.Embedding(100, species_emb_dim, padding_idx=0)
        self.atom_mlp = nn.Sequential(nn.Linear(species_emb_dim + 3, atom_feat_dim), nn.ReLU(), nn.Linear(atom_feat_dim, atom_feat_dim), nn.ReLU())
        self.final_mlp = nn.Sequential(nn.Linear(atom_feat_dim + 9, atom_feat_dim), nn.ReLU(), nn.Linear(atom_feat_dim, atom_feat_dim), nn.ReLU())
        self.fc_mu = nn.Linear(atom_feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(atom_feat_dim, latent_dim)

    def forward(self, lattice, fracs, species):
        atom_feat = self.atom_mlp(torch.cat([self.species_emb(species), fracs], dim=1))
        hid = self.final_mlp(torch.cat([atom_feat.mean(dim=0, keepdim=True), lattice.view(1, 9)], dim=1))
        return self.fc_mu(hid).squeeze(0), self.fc_logvar(hid).squeeze(0)

class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim=128, cond_dim=1, K=32, species_classes=100, atom_feat_dim=128):
        super().__init__()
        self.K = K
        self.latent_mlp = nn.Sequential(nn.Linear(latent_dim + cond_dim, atom_feat_dim), nn.ReLU(), nn.Linear(atom_feat_dim, atom_feat_dim), nn.ReLU())
        self.lattice_out = nn.Linear(atom_feat_dim, 9)
        self.site_embeddings = nn.Parameter(torch.randn(self.K, atom_feat_dim)) # Symmetry breaking!
        self.site_mlp = nn.Sequential(nn.Linear(atom_feat_dim, atom_feat_dim), nn.ReLU(), nn.Linear(atom_feat_dim, atom_feat_dim), nn.ReLU())
        self.frac_out = nn.Linear(atom_feat_dim, 3)
        self.species_out = nn.Linear(atom_feat_dim, species_classes)
        self.occ_out = nn.Linear(atom_feat_dim, 1)

    def forward(self, z, cond):
        hid = self.latent_mlp(torch.cat([z, cond], dim=0).unsqueeze(0))
        lat = self.lattice_out(hid).view(3, 3)
        hid_sites = self.site_mlp(hid.expand(self.K, -1) + self.site_embeddings)
        return lat, torch.sigmoid(self.frac_out(hid_sites)), self.species_out(hid_sites), self.occ_out(hid_sites).squeeze(-1)

# ==========================================
# 3. Differentiable CGCNN (The Surrogate)
# ==========================================
class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=8.0, num_gaussians=64):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        return torch.exp(self.coeff * torch.pow(dist.unsqueeze(-1) - self.offset.view(1, 1, -1), 2))

class CGCNNLayer(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len):
        super().__init__()
        self.fc_full = nn.Linear(2 * atom_fea_len + nbr_fea_len, 2 * atom_fea_len)
        self.ln = nn.LayerNorm(2 * atom_fea_len) # Using LayerNorm for batch_size=1 stability

    def forward(self, atom_in_fea, nbr_atom_fea, nbr_fea):
        N, M = nbr_atom_fea.shape[:2]
        atom_in_expanded = atom_in_fea.unsqueeze(1).expand(-1, M, -1)
        total_nbr_fea = torch.cat([atom_in_expanded, nbr_atom_fea, nbr_fea], dim=2)
        
        total_gated_fea = self.ln(self.fc_full(total_nbr_fea))
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_sumed = torch.sum(torch.sigmoid(nbr_filter) * F.softplus(nbr_core), dim=1)
        return F.softplus(atom_in_fea + nbr_sumed)

class DifferentiableCGCNN(nn.Module):
    """A minimal CGCNN that computes distances mathematically to preserve gradients."""
    def __init__(self, species_classes=100, atom_emb_dim=64, nbr_fea_len=64, max_nbrs=12):
        super().__init__()
        self.max_nbrs = max_nbrs
        self.atom_embedding = nn.Linear(species_classes, atom_emb_dim) # Linear map acts as soft-embedding
        self.expansion = GaussianSmearing(0.0, 8.0, nbr_fea_len)
        self.conv1 = CGCNNLayer(atom_emb_dim, nbr_fea_len)
        self.conv2 = CGCNNLayer(atom_emb_dim, nbr_fea_len)
        self.fc = nn.Linear(atom_emb_dim, 1)

    def forward(self, lat_pred, fracs_pred, species_logits, occ_logits):
        # 1. Soft Atom Features (Maintains gradients to species_logits)
        species_probs = torch.softmax(species_logits, dim=-1)
        atom_fea = self.atom_embedding(species_probs) 

        # 2. Differentiable Pairwise Distance & PBC
        diff = fracs_pred.unsqueeze(1) - fracs_pred.unsqueeze(0) # (K, K, 3)
        diff = diff - torch.round(diff) # Minimum image convention
        cart = torch.matmul(diff, lat_pred) # Fractional to Cartesian
        dist = torch.norm(cart, dim=-1) # (K, K)

        # 3. Mask self-interactions & find neighbors
        eye = torch.eye(dist.size(0), device=dist.device)
        dist = dist + eye * 100.0 # Make self-distance huge
        
        k_nbrs = min(self.max_nbrs, dist.size(1) - 1)
        nbr_dist, nbr_idx = torch.topk(dist, k_nbrs, dim=-1, largest=False)
        nbr_fea = self.expansion(nbr_dist)
        nbr_atom_fea = atom_fea[nbr_idx]

        # 4. Graph Convolutions
        atom_fea = self.conv1(atom_fea, nbr_atom_fea, nbr_fea)
        atom_fea = self.conv2(atom_fea, nbr_atom_fea, nbr_fea)

        # 5. Readout (Weighted by predicted occupancy to ignore dead nodes)
        occ_probs = torch.sigmoid(occ_logits).unsqueeze(-1)
        graph_fea = (atom_fea * occ_probs).sum(dim=0) / (occ_probs.sum() + 1e-6)
        
        return self.fc(graph_fea)

# ==========================================
# 4. Loss & Training Loop
# ==========================================
def compute_structure_loss(fracs_pred, fracs_true, species_logits, species_true, occ_logits):
    K, N = fracs_pred.size(0), fracs_true.size(0)
    with torch.no_grad():
        diff = fracs_pred.unsqueeze(1) - fracs_true.unsqueeze(0)
        dist = torch.norm(diff - torch.round(diff), dim=-1)
        row_ind, col_ind = linear_sum_assignment(dist.cpu().numpy())

    diff_matched = fracs_pred[row_ind] - fracs_true[col_ind]
    diff_matched = diff_matched - torch.round(diff_matched)
    L_coord = F.mse_loss(diff_matched, torch.zeros_like(diff_matched))
    L_species = F.cross_entropy(species_logits[row_ind], species_true[col_ind].long())
    
    occ_target = torch.zeros(K, device=occ_logits.device)
    occ_target[row_ind] = 1.0
    L_occ = F.binary_cross_entropy_with_logits(occ_logits, occ_target)
    
    return L_coord, L_species, L_occ

class CondCDVAE(nn.Module):
    def __init__(self, latent_dim=128, K=32):
        super().__init__()
        self.encoder = SimpleEncoder(latent_dim=latent_dim)
        self.decoder = SimpleDecoder(latent_dim=latent_dim, K=K)
        self.latent_dim = latent_dim
        self.K = K

    def forward(self, lattice, fracs, species, cond):
        mu, logvar = self.encoder(lattice, fracs, species)
        std = (0.5 * logvar).exp()
        z = mu + torch.randn_like(std) * std
        lat_pred, fracs_pred, species_logits, occ_logits = self.decoder(z, cond)
        return lat_pred, fracs_pred, species_logits, occ_logits, mu, logvar

def train_one_epoch(model, dataset, surrogate, optimizer, device):
    model.train()
    total_loss = 0.0
    for sample in dataset:
        lattice = torch.from_numpy(sample["lattice"]).to(device)
        fracs = torch.from_numpy(sample["fracs"]).to(device)
        species = torch.from_numpy(sample["species"]).to(device)
        cond = torch.from_numpy(sample["props"]).to(device)

        lat_p, fracs_p, spec_logits_p, occ_logits_p, mu, logvar = model(lattice, fracs, species, cond)
        
        L_coord, L_species, L_occ = compute_structure_loss(fracs_p, fracs, spec_logits_p, species, occ_logits_p)
        L_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        L_lat = F.mse_loss(lat_p, lattice)
        
        # 👑 THE MAGIC: Fully Differentiable Surrogate Loss
        surrogate_pred = surrogate(lat_p, fracs_p, spec_logits_p, occ_logits_p)
        L_prop = F.mse_loss(surrogate_pred.view(-1), cond.view(-1))
        
        loss = L_coord + L_species + L_occ + 0.1 * L_lat + 1e-3 * L_kl + 0.5 * L_prop
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataset)

# ==========================================
# 5. Main Execution Block
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="dummy_data", help="Dataset folder")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Create dummy data if it doesn't exist
    if not os.path.exists(args.data):
        create_dummy_dataset(args.data)

    # 2. Initialize Models
    ds = CrystalDataset(args.data)
    model = CondCDVAE(latent_dim=64, K=16).to(device)
    
    # Initialize our Differentiable CGCNN surrogate
    surrogate = DifferentiableCGCNN(species_classes=100).to(device)
    
    # CRITICAL: Freeze the surrogate! 
    # In a real scenario, you would load pre-trained weights here and NOT train it.
    for param in surrogate.parameters():
        param.requires_grad = False
    surrogate.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 3. Train
    print("\n--- Training Generator with Differentiable CGCNN Guidance ---")
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, ds, surrogate, optimizer, device)
        print(f"Epoch {epoch}/{args.epochs} | Avg Loss: {loss:.4f}")

    # 4. Generate
    print("\n--- Generating New Materials ---")
    model.eval()
    out_dir = "generated_cifs"
    os.makedirs(out_dir, exist_ok=True)
    target_prop = torch.tensor([-1.5], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        for i in range(3):
            z = torch.randn(model.latent_dim).to(device)
            lat_p, fracs_p, spec_logits_p, occ_logits_p = model.decoder(z, target_prop)
            
            chosen = torch.sigmoid(occ_logits_p) > 0.5
            if not chosen.any(): chosen[:4] = True # Fallback if none chosen
            
            save_structure_to_cif(
                lat_p.cpu().numpy(), 
                fracs_p[chosen].cpu().numpy(), 
                torch.argmax(spec_logits_p[chosen], dim=1).cpu().numpy(), 
                os.path.join(out_dir, f"gen_{i}.cif")
            )
    print(f"Done! Check the '{out_dir}/' folder for your AI-generated materials.")