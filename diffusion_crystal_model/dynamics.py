"""
dynamics.py

晶体扩散模型的核心去噪引擎：基于 E(n)-等变图神经网络 (EGNN)
作用：在给定的时间步 t，读取加噪后的混沌坐标，预测出坐标的去噪梯度 (Score)。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalTimeEmbeddings(nn.Module):
    """
    时间步嵌入 (Time Embedding)
    告诉模型当前处于去噪的哪个阶段。时间 t 越大，表示坐标越混沌。
    （类似于 Transformer 中的位置编码）
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class EGNNLayer(nn.Module):
    """
    单层 E(n)-等变图神经网络层
    核心魔法：不仅更新节点特征 (h)，还基于物理等变性更新原子的三维坐标 (x)。
    """
    def __init__(self, node_dim, edge_dim, time_dim):
        super().__init__()
        self.node_dim = node_dim
        
        # 消息网络：根据两端节点特征、距离的平方、以及当前时间步，生成消息
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + 1 + time_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
            nn.SiLU()
        )
        
        # 坐标注意力网络：决定原子在相对向量上移动的步长（保证等变性的关键）
        self.coord_mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 1, bias=False) # 输出一个标量权重
        )
        
        # 节点更新网络：聚合消息并更新自身化学特征
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + time_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim)
        )

    def forward(self, h, diff_cart, dist_sq, edge_src, edge_dst, t_emb_edges, t_emb_nodes):
        """
        h: 节点特征 (num_atoms, node_dim)
        diff_cart: 考虑了周期边界条件 (PBC) 的笛卡尔坐标差矢量 (num_edges, 3)
        dist_sq: 距离的平方 (num_edges, 1)
        """
        # 1. 生成边消息 (Message Passing)
        h_src, h_dst = h[edge_src], h[edge_dst]
        edge_input = torch.cat([h_src, h_dst, dist_sq, t_emb_edges], dim=-1)
        m_ij = self.edge_mlp(edge_input) # (num_edges, node_dim)
        
        # 2. 坐标的等变更新 (Equivariant Coordinate Update) 🌟
        # 模型输出一个标量，乘以原本的位移矢量，这意味着坐标的更新方向严格受物理几何约束
        coord_weights = self.coord_mlp(m_ij) # (num_edges, 1)
        coord_shift = diff_cart * coord_weights # (num_edges, 3)
        
        # 将邻居对其产生的坐标位移累加到目标节点上
        coord_update = torch.zeros(h.size(0), 3, device=h.device)
        coord_update.index_add_(0, edge_src, coord_shift) # 注意：这里用 src 是为了将力作用回自身
        
        # 3. 节点特征的常规更新
        m_i = torch.zeros_like(h)
        m_i.index_add_(0, edge_dst, m_ij)
        
        node_input = torch.cat([h, m_i, t_emb_nodes], dim=-1)
        h_update = h + self.node_mlp(node_input) # 残差连接
        
        return h_update, coord_update

class CrystalDynamics(nn.Module):
    """
    晶体扩散动力学主干网络
    组装 EGNN 层，处理周期性边界条件 (PBC)，预测分数的去噪梯度。
    """
    def __init__(self, node_dim=64, time_dim=64, num_layers=4):
        super().__init__()
        self.node_dim = node_dim
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim)
        )
        
        # 我们假设输入的节点特征是隐变量 z，这里先做一个简单的映射
        self.node_embedding = nn.Linear(128, node_dim) # 假设上一层的隐变量 z 维度是 128
        
        # 堆叠多层 EGNN
        self.layers = nn.ModuleList([
            EGNNLayer(node_dim, edge_dim=1, time_dim=time_dim) 
            for _ in range(num_layers)
        ])

    def build_pbc_graph(self, frac_coords, lattice, num_atoms_list, k_neighbors=12):
        """动态构建感知周期性边界条件的最邻近图"""
        edge_src, edge_dst = [], []
        diff_cart_list, dist_sq_list = [], []
        start_idx = 0
        device = frac_coords.device
        
        for i, n in enumerate(num_atoms_list):
            f, lat = frac_coords[start_idx : start_idx + n], lattice[i]
            
            # 分数坐标差与最小镜像约定 (PBC 核心)
            diff_f = f.unsqueeze(1) - f.unsqueeze(0)
            diff_f = diff_f - torch.round(diff_f)
            
            # 转为笛卡尔坐标求真实物理向量
            diff_c = torch.matmul(diff_f, lat) # (n, n, 3)
            dist_sq = torch.sum(diff_c ** 2, dim=-1) # (n, n)
            dist_sq.fill_diagonal_(float('inf'))
            
            # 寻找 K 近邻
            k = min(k_neighbors, n - 1)
            if k > 0:
                topk_dist_sq, topk_idx = torch.topk(dist_sq, k, dim=-1, largest=False)
                
                src = topk_idx.flatten() + start_idx
                dst = torch.arange(n, device=device).unsqueeze(1).expand(-1, k).flatten() + start_idx
                
                # 提取对应的笛卡尔差矢量
                # diff_c 形状为 (n, n, 3)，我们要取出 (src, dst) 对应的矢量
                d_c = diff_c[torch.arange(n).unsqueeze(1), topk_idx].view(-1, 3)
                
                edge_src.append(src)
                edge_dst.append(dst)
                diff_cart_list.append(d_c)
                dist_sq_list.append(topk_dist_sq.flatten().unsqueeze(-1))
                
            start_idx += n
            
        return (torch.cat(edge_src), torch.cat(edge_dst), 
                torch.cat(diff_cart_list), torch.cat(dist_sq_list))

    def forward(self, z_nodes, t, frac_coords, lattice, num_atoms_list, batch_indices):
        """
        z_nodes: 节点的初始隐特征 (num_atoms, 128)
        t: 当前去噪时间步 (batch_size,)
        frac_coords: 当前被噪声污染的分数坐标 (num_atoms, 3)
        """
        # 1. 嵌入时间 t
        t_emb = self.time_mlp(t) # (batch_size, time_dim)
        
        # 将 batch 级别的时间扩展到 node 级别和 edge 级别
        t_emb_nodes = t_emb[batch_indices]
        
        # 2. 初始化节点特征
        h = self.node_embedding(z_nodes)
        
        # 3. 构建动态图 (每次迭代坐标改变了，图的邻居也会变)
        edge_src, edge_dst, diff_cart, dist_sq = self.build_pbc_graph(
            frac_coords, lattice, num_atoms_list
        )
        
        t_emb_edges = t_emb_nodes[edge_src] # 边的时间特征取自源节点
        
        # 4. 逐层通过等变网络
        total_coord_shift_cart = torch.zeros_like(frac_coords)
        
        for layer in self.layers:
            h, coord_update = layer(h, diff_cart, dist_sq, edge_src, edge_dst, t_emb_edges, t_emb_nodes)
            total_coord_shift_cart += coord_update
            
        # 5. 将笛卡尔坐标的平移量转化回分数坐标
        # 因为 x_cart = f_frac * lattice  =>  \Delta f = \Delta x * lattice^{-1}
        # 为了高效计算，我们需要按 batch 处理逆矩阵
        inv_lattice = torch.linalg.inv(lattice)
        inv_lattice_nodes = inv_lattice[batch_indices] # (num_atoms, 3, 3)
        
        # 矩阵乘法求解分数坐标漂移 (Score)
        # total_coord_shift_cart: (num_atoms, 3)
        # inv_lattice_nodes: (num_atoms, 3, 3)
        shift_frac = torch.bmm(total_coord_shift_cart.unsqueeze(1), inv_lattice_nodes).squeeze(1)
        
        return shift_frac