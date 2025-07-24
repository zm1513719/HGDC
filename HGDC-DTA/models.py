import torch
import torch.nn as nn
from sympy.abc import alpha
from torch import scatter_add
from torch.nn.functional import sigmoid

from torch_geometric.nn import DenseGCNConv, GCNConv, global_mean_pool as gep
from torch_geometric.utils import dropout_adj


class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias

class GCNBlock(nn.Module):
    def __init__(self, gcn_layers_dim, dropout_rate=0., gelu_layers_index=[], dropout_layers_index=[]):
        super(GCNBlock, self).__init__()

        self.conv_layers = nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            conv_layer = GCNConv(gcn_layers_dim[i], gcn_layers_dim[i + 1])
            self.conv_layers.append(conv_layer)
        # self.DyT = DyT()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.gelu_layers_index = gelu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, edge_index, edge_weight, batch):
        output = x
        embeddings = []
        for conv_layer_index in range(len(self.conv_layers)):
            output = self.conv_layers[conv_layer_index](output, edge_index, edge_weight)
            if conv_layer_index in self.gelu_layers_index:
                output = self.gelu(output)
            if conv_layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(gep(output, batch))

        return embeddings


class GCNModel(nn.Module):
    def __init__(self, layers_dim):
        super(GCNModel, self).__init__()

        self.num_layers = len(layers_dim) - 1
        self.graph_conv = GCNBlock(layers_dim, gelu_layers_index=list(range(self.num_layers)))

    def forward(self, graph_batchs):
        embedding_batchs = list(
                map(lambda graph: self.graph_conv(graph.x, graph.edge_index, None, graph.batch), graph_batchs))
        embeddings = []
        for i in range(self.num_layers):
            embeddings.append(torch.cat(list(map(lambda embedding_batch: embedding_batch[i], embedding_batchs)), 0))

        return embeddings


class DenseGCNBlock(nn.Module):
    def __init__(self, gcn_layers_dim, dropout_rate=0., gelu_layers_index=[], dropout_layers_index=[]):
        super(DenseGCNBlock, self).__init__()

        self.conv_layers = nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            conv_layer = DenseGCNConv(gcn_layers_dim[i], gcn_layers_dim[i + 1])
            self.conv_layers.append(conv_layer)

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.gelu_layers_index = gelu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, adj):
        output = x
        embeddings = []
        for conv_layer_index in range(len(self.conv_layers)):
            output = self.conv_layers[conv_layer_index](output, adj, add_loop=False)
            if conv_layer_index in self.gelu_layers_index:
                output = self.gelu(output)
            if conv_layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(torch.squeeze(output, dim=0))

        return embeddings


class DenseGCNModel(nn.Module):
    def __init__(self, layers_dim, edge_dropout_rate=0.):
        super(DenseGCNModel, self).__init__()

        self.edge_dropout_rate = edge_dropout_rate
        self.num_layers = len(layers_dim) - 1
        self.graph_conv = DenseGCNBlock(layers_dim, 0.1, gelu_layers_index=list(range(self.num_layers)),
                                        dropout_layers_index=list(range(self.num_layers)))

    def forward(self, graph):
        xs, adj, num_d, num_t = graph.x, graph.adj, graph.num_drug, graph.num_target
        indexs = torch.where(adj != 0)
        edge_indexs = torch.cat((torch.unsqueeze(indexs[0], 0), torch.unsqueeze(indexs[1], 0)), 0)
        edge_indexs_dropout, edge_weights_dropout = dropout_adj(edge_index=edge_indexs, edge_attr=adj[indexs],
                                                                p=self.edge_dropout_rate, force_undirected=True,
                                                                num_nodes=num_d + num_t, training=self.training)
        adj_dropout = torch.zeros_like(adj)
        adj_dropout[edge_indexs_dropout[0], edge_indexs_dropout[1]] = edge_weights_dropout

        embeddings = self.graph_conv(xs, adj_dropout)

        return embeddings


class LinearBlock(nn.Module):
    def __init__(self, linear_layers_dim, dropout_rate=0., gelu_layers_index=[], dropout_layers_index=[]):
        super(LinearBlock, self).__init__()

        self.layers = nn.ModuleList()
        # nn.Linear(linear_layers_dim[i], linear_layers_dim[i])
        for i in range(len(linear_layers_dim) - 1):
            layer = nn.Linear(linear_layers_dim[i], linear_layers_dim[i+1])


            self.layers.append(layer)

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.gelu_layers_index = gelu_layers_index
        self.dropout_layers_index = dropout_layers_index
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = x
        embeddings = []
        for layer_index in range(len(self.layers)):
            output = self.layers[layer_index](output)
            if layer_index in self.gelu_layers_index:
                output = self.gelu(output)
                # output = self.gelu(output)
            if layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(output)

        return embeddings


class EnhancedContrast(nn.Module):

    def __init__(self, hidden_dim, output_dim, tau, lam, alpha=1.0, beta=1.0):
        super(EnhancedContrast, self).__init__()

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # nn.GELU(),
            DyT(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.GELU(),
            DyT(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )


        self.node_transform = nn.Linear(output_dim * 2, output_dim)

        self.tau = tau
        self.lam = lam
        self.alpha = alpha
        self.beta = beta

        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.kaiming_normal_(model.weight, mode='fan_in', nonlinearity='relu')

    def project(self, z):

        z1_proj = self.proj(z)
        z2_proj = self.sigmoid(self.proj(z))
        z_proj = z1_proj * z2_proj
        return z_proj

    def global_contrast(self, za_proj, zb_proj, pos):

        matrix_a2b = self.sim(za_proj, zb_proj)
        matrix_b2a = matrix_a2b.t()


        matrix_a2b = matrix_a2b / (torch.sum(matrix_a2b, dim=1).view(-1, 1) + 1e-6)
        lori_a = -torch.log(matrix_a2b.mul(pos.to_dense()).sum(dim=-1)).mean()

        matrix_b2a = matrix_b2a / (torch.sum(matrix_b2a, dim=1).view(-1, 1) + 1e-6)
        lori_b = -torch.log(matrix_b2a.mul(pos.to_dense()).sum(dim=-1)).mean()

        return self.lam * lori_a + (1 - self.lam) * lori_b

    def local_contrast(self, z1, z2, edge_index_map, batch=None):

        if batch is None:

            batch = torch.zeros(z1.size(0), dtype=torch.long, device=z1.device)

        T = self.tau
        N = z1.size(0)
        G = batch.max().item() + 1 if batch.numel() > 0 else 1

        z1_abs = z1.norm(dim=1)
        z2_abs = z2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', z1, z2) / torch.einsum('i,j->ij', z1_abs, z2_abs)
        sim_matrix = torch.exp(sim_matrix / T)

        pos_sim = sim_matrix[range(N), range(N)]
        pos_sim_graph = torch.zeros(G, device=pos_sim.device)
        pos_sim_graph.scatter_add_(0, batch, pos_sim)

        batch_sim = torch.zeros((G, G), device=sim_matrix.device)
        for i in range(G):
            for j in range(G):
                batch_mask_i = batch == i
                batch_mask_j = batch == j
                batch_sim[i, j] = sim_matrix[batch_mask_i][:, batch_mask_j].sum()

        graph_sim = torch.diag(batch_sim)

        neg_sim0 = batch_sim.sum(dim=0) - graph_sim
        neg_sim1 = batch_sim.sum(dim=1) - graph_sim
        loss0 = pos_sim_graph / (neg_sim0 + 1e-5)
        loss1 = pos_sim_graph / (neg_sim1 + 1e-5)
        loss0 = torch.log(loss0).mean()
        loss1 = torch.log(loss1).mean()
        inter_loss = (loss0 + loss1) / 2.0

        neg_sim = graph_sim - pos_sim_graph
        loss = pos_sim_graph / (neg_sim + 1e-5)
        inner_loss = -torch.log(loss).mean()

        return inter_loss, inner_loss

    def sim(self, z1, z2):

        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, za, zb, pos, edge_index_map=None, batch=None):

        za_proj = self.project(za)
        zb_proj = self.project(zb)


        global_loss = self.global_contrast(za_proj, zb_proj, pos)

        local_loss = 0
        if edge_index_map is not None:

            inter_loss, inner_loss = self.local_contrast(za_proj, zb_proj, edge_index_map, batch)
            local_loss = self.alpha * inter_loss + self.beta * inner_loss

        total_loss = global_loss + local_loss
        return total_loss, torch.cat((za_proj, zb_proj), 1)


class CSCoDTA(nn.Module):
    def __init__(self, tau, lam, ns_dims, d_ms_dims, t_ms_dims, embedding_dim=128, dropout_rate=0.2,
                 alpha=1, beta=1):
        super(CSCoDTA, self).__init__()

        self.output_dim = embedding_dim * 2

        self.affinity_graph_conv1 = DenseGCNModel(ns_dims, dropout_rate)
        self.affinity_graph_conv2 = DenseGCNModel(ns_dims, dropout_rate)

        self.drug_graph_conv1 = GCNModel(d_ms_dims)
        self.drug_graph_conv2 = GCNModel(d_ms_dims)

        self.target_graph_conv1 = GCNModel(t_ms_dims)
        self.target_graph_conv2 = GCNModel(t_ms_dims)


        self.drug_contrast = EnhancedContrast(ns_dims[-1], embedding_dim, tau, lam, alpha, beta)
        self.target_contrast = EnhancedContrast(ns_dims[-1], embedding_dim, tau, lam, alpha, beta)

        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, affinity_graph, drug_graph_batchs, target_graph_batchs, drug_pos, target_pos):
        num_d = affinity_graph.num_drug


        affinity_graph_embedding1 = self.affinity_graph_conv1(affinity_graph)[-1]
        affinity_graph_embedding2 = self.sigmoid(self.affinity_graph_conv2(affinity_graph)[-1])
        affinity_graph_embedding = affinity_graph_embedding1 * affinity_graph_embedding2

        drug_graph_embedding1 = self.drug_graph_conv1(drug_graph_batchs)[-1]
        drug_graph_embedding2 = self.sigmoid(self.drug_graph_conv2(drug_graph_batchs)[-1])
        drug_graph_embedding = drug_graph_embedding1 * drug_graph_embedding2

        target_graph_embedding1 = self.target_graph_conv1(target_graph_batchs)[-1]
        target_graph_embedding2 = self.sigmoid(self.target_graph_conv2(target_graph_batchs)[-1])
        target_graph_embedding = target_graph_embedding1 * target_graph_embedding2

        drug_edge_index_map = self.create_edge_index_map(affinity_graph, is_drug=True)
        target_edge_index_map = self.create_edge_index_map(affinity_graph, is_drug=False)

        drug_batch = torch.zeros(num_d, dtype=torch.long, device=affinity_graph_embedding.device)
        target_batch = torch.zeros(affinity_graph.num_target, dtype=torch.long, device=affinity_graph_embedding.device)

        dru_loss, drug_embedding = self.drug_contrast(
            affinity_graph_embedding[:num_d],
            drug_graph_embedding,
            drug_pos,
            drug_edge_index_map,
            drug_batch
        )

        tar_loss, target_embedding = self.target_contrast(
            affinity_graph_embedding[num_d:],
            target_graph_embedding,
            target_pos,
            target_edge_index_map,
            target_batch
        )

        return dru_loss + tar_loss, drug_embedding, target_embedding

    def create_edge_index_map(self, affinity_graph, is_drug=True):

        num_d = affinity_graph.num_drug
        adj = affinity_graph.adj

        if is_drug:
            sub_adj = adj[:num_d, :num_d]
        else:
            sub_adj = adj[num_d:, num_d:]

        rows, cols = torch.where(sub_adj != 0)
        edge_index_map = torch.stack([rows, cols], dim=0)

        return edge_index_map

class PredictModule(nn.Module):
    def __init__(self, embedding_dim=256, output_dim=1):
        super(PredictModule, self).__init__()

        self.prediction_func, prediction_dim_func = (lambda x, y: torch.cat((x, y), -1), lambda dim: 2 * dim)

        mlp_layers_dim = [prediction_dim_func(embedding_dim), 1024, 512, output_dim]

        self.mlp = LinearBlock(mlp_layers_dim, 0.1, gelu_layers_index=[0, 1], dropout_layers_index=[0, 1])

    def forward(self, data, drug_embedding, target_embedding):
        drug_id, target_id, y = data.drug_id, data.target_id, data.y

        drug_feature = drug_embedding[drug_id.int().cpu().numpy()]
        target_feature = target_embedding[target_id.int().cpu().numpy()]

        concat_feature = self.prediction_func(drug_feature, target_feature)
        mlp_embeddings = self.mlp(concat_feature)
        link_embeddings = mlp_embeddings[-2]
        out = mlp_embeddings[-1]

        return out, link_embeddings

