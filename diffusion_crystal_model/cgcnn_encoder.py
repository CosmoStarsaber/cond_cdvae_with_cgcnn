"""
cgcnn_encoder.py

正宗的晶体图卷积神经网络 (CGCNN) 编码器
特性：
1. 使用门控卷积 (Gated Graph Convolution)
2. 动态自适应任意原子数 (打破 max_atoms 限制)
3. 严格处理周期性边界条件 (PBC) 的物理距离
"""

import torch
import torch.nn as nn

class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=8.0, num_gaussians=64):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        return torch.exp(self.coeff * torch.pow(dist - self.offset.view(1, -1), 2))

class CGCNNLayer(nn.Module):
    """标准的 CGCNN 门控卷积层"""
    def __init__(self, atom_fea_len=64, nbr_fea_len=64):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2 * atom_fea_len + nbr_fea_len, 2 * atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * atom_fea_len)
        self.bn2 = nn.BatchNorm1d(atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, edge_src, edge_dst):
        # 组装边特征：源原子 + 目标原子 + 键特征
        atom_src = atom_in_fea[edge_src]
        atom_dst = atom_in_fea[edge_dst]
        total_fea = torch.cat([atom_src, atom_dst, nbr_fea], dim=1)
        
        total_fea = self.fc_full(total_fea)
        total_fea = self.bn1(total_fea)
        
        # 门控机制：一半过 Sigmoid (决定传递多少信息)，一半过 Softplus (提取特征)
        filter_fea, core_fea = total_fea.chunk(2, dim=1)
        filter_fea = self.sigmoid(filter_fea)
        core_fea = self.softplus1(core_fea)
        
        # 消息传递
        nbr_msg = filter_fea * core_fea
        
        # 邻居聚合
        atom_update = torch.zeros_like(atom_in_fea)
        atom_update.index_add_(0, edge_dst, nbr_msg)
        
        # 残差连接
        atom_update = self.bn2(atom_update)
        atom_out_fea = self.softplus2(atom_in_fea + atom_update)
        return atom_out_fea

class CGCNNEncoder(nn.Module):
    def __init__(self, latent_dim=128, atom_fea_len=64, nbr_fea_len=64, n_conv=3):
        super().__init__()
        self.embedding = nn.Embedding(100, atom_fea_len, padding_idx=0)
        self.distance_expansion = GaussianSmearing(start=0.0, stop=8.0, num_gaussians=nbr_fea_len)
        
        self.convs = nn.ModuleList([CGCNNLayer(atom_fea_len, nbr_fea_len) for _ in range(n_conv)])
        
        # 输出层：映射到均值和对数方差 (VAE 核心)
        self.conv_to_fc = nn.Sequential(
            nn.Linear(atom_fea_len + 9, 128), 
            nn.Softplus(),
            nn.Linear(128, latent_dim * 2)
        )

    def build_graph(self, lattice, fracs, num_atoms_list, k_neighbors=12):
        """处理 PBC 并构建邻接图"""
        edge_src, edge_dst, edge_dist = [], [], []
        start_idx = 0
        device = lattice.device
        
        for i, n in enumerate(num_atoms_list):
            f, lat = fracs[start_idx : start_idx + n], lattice[i]
            diff = f.unsqueeze(1) - f.unsqueeze(0)
            diff = diff - torch.round(diff)
            dist_matrix = torch.norm(torch.matmul(diff, lat), dim=-1)
            dist_matrix.fill_diagonal_(float('inf'))
            
            k = min(k_neighbors, n - 1)
            if k > 0:
                topk_dist, topk_idx = torch.topk(dist_matrix, k, dim=-1, largest=False)
                edge_src.append(topk_idx.flatten() + start_idx)
                edge_dst.append(torch.arange(n, device=device).unsqueeze(1).expand(-1, k).flatten() + start_idx)
                edge_dist.append(topk_dist.flatten())
                
            start_idx += n
            
        return torch.cat(edge_src), torch.cat(edge_dst), torch.cat(edge_dist).unsqueeze(-1)

    def forward(self, lattice, fracs, species, batch_indices, num_atoms_list):
        atom_fea = self.embedding(species)
        edge_src, edge_dst, edge_dist = self.build_graph(lattice, fracs, num_atoms_list)
        nbr_fea = self.distance_expansion(edge_dist)
        
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, edge_src, edge_dst)
            
        # 平均池化 (Mean Pooling) 获取全局晶体特征
        num_graphs = lattice.size(0)
        crys_fea = torch.zeros(num_graphs, atom_fea.size(1), device=lattice.device)
        crys_fea.index_add_(0, batch_indices, atom_fea)
        crys_fea = crys_fea / torch.bincount(batch_indices).view(-1, 1).float()
        
        # 拼接晶格矩阵特征
        crys_fea = torch.cat([crys_fea, lattice.view(num_graphs, 9)], dim=1)
        
        # 输出 VAE 的 mu 和 logvar
        return torch.chunk(self.conv_to_fc(crys_fea), 2, dim=-1)