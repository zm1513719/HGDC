import utils
import math




def train(model, predictor, device, train_loader, drug_graphs_DataLoader, target_graphs_DataLoader, lr, epoch,
          batch_size, affinity_graph, drug_pos, target_pos, loss_list):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    predictor.train()
    LOG_INTERVAL = 10
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, chain(model.parameters(), predictor.parameters())), lr=lr, weight_decay=0)
    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))
    target_graph_batchs = list(map(lambda graph: graph.to(device), target_graphs_DataLoader))
    epoch_loss = 0
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        ssl_loss, drug_embedding, target_embedding = model(affinity_graph.to(device), drug_graph_batchs,
                                                                  target_graph_batchs, drug_pos, target_pos)
        output, _ = predictor(data.to(device), drug_embedding, target_embedding)

        loss = loss_fn(output, data.y.view(-1, 1).float().to(device)) + ssl_loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        progress = math.floor(100. * batch_idx / len(train_loader))

        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))
            # print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * batch_size, len(train_loader.dataset), progress,
            #     loss.item()))
    average_loss = epoch_loss / len(train_loader)
    loss_list.append(average_loss)

def test(model, predictor, device, loader, drug_graphs_DataLoader, target_graphs_DataLoader, affinity_graph, drug_pos,
         target_pos):
    model.eval()
    predictor.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))  # drug graphs
    target_graph_batchs = list(map(lambda graph: graph.to(device), target_graphs_DataLoader))  # target graphs
    with torch.no_grad():
        for data in loader:
            _, drug_embedding, target_embedding = model(affinity_graph.to(device), drug_graph_batchs, target_graph_batchs, drug_pos, target_pos)
            output, _ = predictor(data.to(device), drug_embedding, target_embedding)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def train_predict():
    print("Data preparation in progress for the {} dataset...".format(args.dataset))
    affinity_mat = load_data(args.dataset)
    train_data, test_data, affinity_graph, drug_pos, target_pos = process_data(affinity_mat, args.dataset, args.num_pos, args.pos_threshold)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    drug_graphs_dict = get_drug_molecule_graph(
        json.load(open(f'data/{args.dataset}/drugs.txt'), object_pairs_hook=OrderedDict))
    drug_graphs_Data = GraphDataset(graphs_dict=drug_graphs_dict, dttype="drug")
    drug_graphs_DataLoader = torch.utils.data.DataLoader(drug_graphs_Data, shuffle=False, collate_fn=collate,
                                                         batch_size=affinity_graph.num_drug)
    target_graphs_dict = get_target_molecule_graph(
        json.load(open(f'data/{args.dataset}/targets.txt'), object_pairs_hook=OrderedDict), args.dataset)
    target_graphs_Data = GraphDataset(graphs_dict=target_graphs_dict, dttype="target")
    target_graphs_DataLoader = torch.utils.data.DataLoader(target_graphs_Data, shuffle=False, collate_fn=collate,
                                                           batch_size=affinity_graph.num_target)

    print("Model preparation... ")
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
    model = CSCoDTA(tau=args.tau,
                    lam=args.lam,
                    ns_dims=[affinity_graph.num_drug + affinity_graph.num_target + 2, 512, 256],
                    d_ms_dims=[78, 78, 78 * 2, 256],
                    t_ms_dims=[54, 54, 54 * 2, 256],
                    embedding_dim=128,  # 128
                    dropout_rate=args.edge_dropout_rate,
                    alpha=args.alpha,   # New parameter for inter-local loss weight
                    beta=args.beta  # New parameter for inner-local loss weight
                    )
    predictor = PredictModule()
    drug_pos = drug_pos.to(device)
    target_pos = target_pos.to(device)
    model.to(device)
    predictor.to(device)

    print("Start training...")
    loss_list = []
    eval_results = []

    result_dir = 'result'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(result_dir, f'training_log_{timestamp}.csv')

    with open(log_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Training Loss', 'mse', 'r2', 'time'])

    params_text = f"lr={args.lr}\noptimizer=AdamW\nnn.init=kaiming_normal_"

    for epoch in range(args.epochs):

        start_time = time.time()

        train(model, predictor, device, train_loader, drug_graphs_DataLoader, target_graphs_DataLoader, args.lr, epoch+1,
              args.batch_size, affinity_graph, drug_pos, target_pos, loss_list)
        G, P = test(model, predictor, device, test_loader, drug_graphs_DataLoader, target_graphs_DataLoader,
                    affinity_graph, drug_pos, target_pos)

        # r = model_evaluate(G, P)
        # print(r)
        mse = utils.get_mse(G, P)
        r2 = utils.get_rm2(G, P)
        # ci = utils.get_ci(G, P)

        print(f"Epoch {epoch + 1}, Test MSE: {mse:.4f}, R²: {r2:.4f}")
        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f"Epoch {epoch + 1} duration: {epoch_duration:.2f} seconds")

        eval_results.append((epoch + 1, loss_list[-1], mse, r2,epoch_duration))

        with open(log_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, loss_list[-1], mse, r2, epoch_duration])

    print('\npredicting for test data')
    G, P = test(model, predictor, device, test_loader, drug_graphs_DataLoader, target_graphs_DataLoader, affinity_graph,
                drug_pos, target_pos)

    mse = utils.get_mse(G, P)
    r2 = utils.get_rm2(G, P)
    print(f"Final Test MSE: {mse:.4f}, R²: {r2:.4f}")



if __name__ == '__main__':
    import os
    import argparse
    import torch
    import json
    import warnings
    from collections import OrderedDict
    from torch import nn
    from itertools import chain
    from data_process import load_data, process_data, get_drug_molecule_graph, get_target_molecule_graph
    from utils import *
    from models import CSCoDTA, PredictModule
    import matplotlib.pyplot as plt
    import time
    import csv


    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='davis')
    parser.add_argument('--epochs', type=int, default=4000 )   # --kiba 6000，davis 4000
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--edge_dropout_rate', type=float, default=0.2)   # --kiba 0.
    parser.add_argument('--tau', type=float, default=0.4) # davis 0.4；kiba 0.8
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--num_pos', type=int, default=5)    # --kiba 10 --davis 5
    parser.add_argument('--pos_threshold', type=float, default=8.0)
    # New arguments for local contrastive loss
    parser.add_argument('--alpha', type=float, default=1,
                        help='Weight for inter-local contrastive loss')
    parser.add_argument('--beta', type=float, default=1,
                        help='Weight for inner-local contrastive loss')
    args, _ = parser.parse_known_args()
    # args = parser.parse_args()

    train_predict()


