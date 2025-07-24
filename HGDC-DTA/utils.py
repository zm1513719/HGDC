import torch
import numpy as np

from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric import data as DATA
from sklearn.metrics import roc_auc_score




class DTADataset(InMemoryDataset):
    def __init__(self, root='/tmp', transform=None, pre_transform=None, drug_ids=None, target_ids=None, y=None):
        super(DTADataset, self).__init__(root, transform, pre_transform)
        self.process(drug_ids, target_ids, y)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, drug_ids, target_ids, y):
        data_list = []
        for i in range(len(drug_ids)):
            DTA = DATA.Data(drug_id=torch.IntTensor([drug_ids[i]]), target_id=torch.IntTensor([target_ids[i]]), y=torch.FloatTensor([y[i]]))
            data_list.append(DTA)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class GraphDataset(InMemoryDataset):
    def __init__(self, root='/tmp', transform=None, pre_transform=None, graphs_dict=None, dttype=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.dttype = dttype
        self.process(graphs_dict)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, graphs_dict):
        data_list = []
        for key in graphs_dict:
            size, features, edge_index = graphs_dict[key]
            GCNData = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index).transpose(1, 0))
            GCNData.__setitem__(f'{self.dttype}_size', torch.LongTensor([size]))
            data_list.append(GCNData)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def minMaxNormalize(Y, Y_min=None, Y_max=None):
    if Y_min is None:
        Y_min = np.min(Y)
    if Y_max is None:
        Y_max = np.max(Y)
    normalize_Y = (Y - Y_min) / (Y_max - Y_min)
    return normalize_Y


def denseAffinityRefine(adj, k):
    refine_adj = np.zeros_like(adj)
    indexs1 = np.tile(np.expand_dims(np.arange(adj.shape[0]), 0), (k, 1)).transpose()
    indexs2 = np.argpartition(adj, -k, 1)[:, -k:]
    refine_adj[indexs1, indexs2] = adj[indexs1, indexs2]
    return refine_adj


def collate(data_list):
    batch = Batch.from_data_list(data_list)
    return batch


def get_mse(Y, P):
    Y = np.array(Y)
    P = np.array(P)
    return np.average((Y - P) ** 2)


def get_rm2(Y, P):
    r2 = r_squared_error(Y, P)
    r02 = squared_error_zero(Y, P)
    return r2 * (1 - np.sqrt(np.absolute(r2 ** 2 - r02 ** 2)))


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.mean(y_obs)
    y_pred_mean = np.mean(y_pred)
    mult = sum((y_obs - y_obs_mean) * (y_pred - y_pred_mean)) ** 2
    y_obs_sq = sum((y_obs - y_obs_mean) ** 2)
    y_pred_sq = sum((y_pred - y_pred_mean) ** 2)
    return mult / (y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    return sum(y_obs * y_pred) / sum(y_pred ** 2)


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.mean(y_obs)
    upp = sum((y_obs - k * y_pred) ** 2)
    down = sum((y_obs - y_obs_mean) ** 2)
    return 1 - (upp / down)


# def get_ci(true_values, predictions):
#     """
#     计算回归任务的 Concordance Index (CI)。
#     :param true_values: 真实值 (numpy array 或 list)
#     :param predictions: 预测值 (numpy array 或 list)
#     :return: CI 值
#     """
#     true_values = np.array(true_values).flatten()
#     predictions = np.array(predictions).flatten()
#
#     # 确保输入数据长度一致
#     if len(true_values) != len(predictions):
#         raise ValueError("true_values 和 predictions 的长度必须一致")
#
#     # 计算 CI
#     concordant = 0
#     total = 0
#
#     for i in range(len(true_values)):
#         for j in range(i + 1, len(true_values)):
#             if true_values[i] != true_values[j]:
#                 total += 1
#                 if (true_values[i] > true_values[j] and predictions[i] > predictions[j]) or \
#                    (true_values[i] < true_values[j] and predictions[i] < predictions[j]):
#                     concordant += 1
#
#     if total == 0:
#         return 0.0  # 如果没有可比较的对，返回 0
#
#     return concordant / total

def get_ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci

def model_evaluate(Y, P):

    return (get_mse(Y, P),
            get_rm2(Y, P))
