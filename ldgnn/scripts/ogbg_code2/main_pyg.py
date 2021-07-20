import json
import random
from multiprocessing import Pool

import torch
import wandb
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from ldgnn.scripts.ogbg_code2.gnn import GNN

from tqdm import tqdm
import argparse
import time
import numpy as np
import pandas as pd
import os

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

### importing utils

### for data transform
from ldgnn.scripts.ogbg_code2.utils import get_vocab_mapping, ASTNodeEncoder
from ldgnn.scripts.ogbg_code2.utils import augment_edge, encode_y_to_arr, decode_arr_to_seq

multicls_criterion = torch.nn.CrossEntropyLoss()


def train(model, device, loader, optimizer):
    model.train()

    loss_accum = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred_list = model(batch)
            optimizer.zero_grad()

            loss = 0
            for i in range(len(pred_list)):
                loss += multicls_criterion(pred_list[i].to(torch.float32), batch.y_arr[:, i])

            loss = loss / len(pred_list)

            loss.backward()
            optimizer.step()

            loss_accum += loss.item()

    print('Average training loss: {}'.format(loss_accum / (step + 1)))
    return loss_accum / (step + 1)


def eval(model, device, loader, evaluator, arr_to_seq):
    model.eval()
    seq_ref_list = []
    seq_pred_list = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred_list = model(batch)

            mat = []
            for i in range(len(pred_list)):
                mat.append(torch.argmax(pred_list[i], dim=1).view(-1, 1))
            mat = torch.cat(mat, dim=1)

            seq_pred = [arr_to_seq(arr) for arr in mat]

            # PyG = 1.4.3
            # seq_ref = [batch.y[i][0] for i in range(len(batch.y))]

            # PyG >= 1.5.0
            seq_ref = [batch.y[i] for i in range(len(batch.y))]

            seq_ref_list.extend(seq_ref)
            seq_pred_list.extend(seq_pred)

    input_dict = {"seq_ref": seq_ref_list, "seq_pred": seq_pred_list}

    return evaluator.eval(input_dict)


def poolmapf(x):  return x.num_nodes


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbg-code2 data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gcn-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gcn-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--max_seq_len', type=int, default=5,
                        help='maximum sequence length to predict (default: 5)')
    parser.add_argument('--num_vocab', type=int, default=5000,
                        help='the number of vocabulary used for sequence prediction (default: 5000)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=25,
                        help='number of epochs to train (default: 25)')
    parser.add_argument('--random_split', action='store_true', help="Use a fully random split (not project split like original)")
    parser.add_argument('--use15', action='store_true', help="Use only 15% of training data")
    parser.add_argument('--removelarge', action='store_true', help="Remove extra large graphs from data.")
    parser.add_argument('--residual', action='store_true', help="Use residual connections.")
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-code2",
                        help='dataset name (default: ogbg-code2)')
    parser.add_argument('--trainevalinter', type=int, default=1, help="evaluate every 'trainevalinter' epochs")
    parser.add_argument('--seed', type=int, default=42, help="seed")
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    settings = args.__dict__.copy()
    print(json.dumps(settings, indent=3))
    wandb.init(project=f"ogbg-code2", config=settings, reinit=True)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name=args.dataset)

    seq_len_list = np.array([len(seq) for seq in dataset.data.y])
    print('Target sequence less or equal to {} is {}%.'.format(args.max_seq_len,
                                                              np.sum(seq_len_list <= args.max_seq_len) / len(
                                                                  seq_len_list)))

    split_idx = dataset.get_idx_split()

    if args.random_split:
        print('Using random split')
        perm = torch.randperm(len(dataset))
        num_train, num_valid, num_test = len(split_idx['train']), len(split_idx['valid']), len(split_idx['test'])
        split_idx['train'] = perm[:num_train]
        split_idx['valid'] = perm[num_train:num_train + num_valid]
        split_idx['test'] = perm[num_train + num_valid:]

        assert (len(split_idx['train']) == num_train)
        assert (len(split_idx['valid']) == num_valid)
        assert (len(split_idx['test']) == num_test)

    if args.use15:
        print("using 15% of training data")
        perm = torch.randperm(int(round(len(split_idx["train"]) * 0.15)))
        split_idx["train"] = split_idx["train"][perm]

    # print(split_idx['train'])
    # print(split_idx['valid'])
    # print(split_idx['test'])

    # train_method_name = [' '.join(dataset.data.y[i]) for i in split_idx['train']]
    # valid_method_name = [' '.join(dataset.data.y[i]) for i in split_idx['valid']]
    # test_method_name = [' '.join(dataset.data.y[i]) for i in split_idx['test']]
    # print('#train')
    # print(len(train_method_name))
    # print('#valid')
    # print(len(valid_method_name))
    # print('#test')
    # print(len(test_method_name))

    # train_method_name_set = set(train_method_name)
    # valid_method_name_set = set(valid_method_name)
    # test_method_name_set = set(test_method_name)

    # # unique method name
    # print('#unique train')
    # print(len(train_method_name_set))
    # print('#unique valid')
    # print(len(valid_method_name_set))
    # print('#unique test')
    # print(len(test_method_name_set))

    # # unique valid/test method name
    # print('#valid unseen during training')
    # print(len(valid_method_name_set - train_method_name_set))
    # print('#test unseen during training')
    # print(len(test_method_name_set - train_method_name_set))

    ### building vocabulary for sequence predition. Only use training data.

    vocab2idx, idx2vocab = get_vocab_mapping([dataset.data.y[i] for i in split_idx['train']], args.num_vocab)

    # test encoder and decoder
    # for data in dataset:
    #     # PyG >= 1.5.0
    #     print(data.y)
    #
    #     # PyG 1.4.3
    #     # print(data.y[0])
    #     data = encode_y_to_arr(data, vocab2idx, args.max_seq_len)
    #     print(data.y_arr[0])
    #     decoded_seq = decode_arr_to_seq(data.y_arr[0], idx2vocab)
    #     print(decoded_seq)
    #     print('')

    ## test augment_edge
    # data = dataset[2]
    # print(data)
    # data_augmented = augment_edge(data)
    # print(data_augmented)

    ### set the transform function
    # augment_edge: add next-token edge as well as inverse edges. add edge attributes.
    # encode_y_to_arr: add y_arr to PyG data object, indicating the array representation of a sequence.
    dataset.transform = transforms.Compose(
        [augment_edge, lambda data: encode_y_to_arr(data, vocab2idx, args.max_seq_len)])

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    # remove large graphs from training
    if args.removelarge:
        print("removing large graphs from training")
        # maxdepth = 15
        maxnodes = 370
        _trainidx = list(split_idx["train"].cpu().numpy())
        # pool = Pool(processes=4)
        # _numnodes = pool.map(poolmapf, dataset)
        _numnodes = [dataset[int(idx)].num_nodes for idx in tqdm(_trainidx)]
        trainidx = [idx for i, idx in tqdm(enumerate(_trainidx)) if _numnodes[i] < maxnodes]
        trainidx = torch.tensor(trainidx, device=split_idx["train"].device, dtype=split_idx["train"].dtype)

        _valididx = list(split_idx["valid"].cpu().numpy())
        _numnodes = [dataset[int(idx)].num_nodes for idx in tqdm(_valididx)]
        valididx = [idx for i, idx in tqdm(enumerate(_valididx)) if _numnodes[i] < maxnodes]
        valididx = torch.tensor(valididx, device=split_idx["valid"].device, dtype=split_idx["valid"].dtype)
        
        _testidx = list(split_idx["test"].cpu().numpy())
        _numnodes = [dataset[int(idx)].num_nodes for idx in tqdm(_testidx)]
        testidx = [idx for i, idx in tqdm(enumerate(_testidx)) if _numnodes[i] < maxnodes]
        testidx = torch.tensor(testidx, device=split_idx["test"].device, dtype=split_idx["test"].dtype)
        
        print(f"Retained {len(trainidx)/len(split_idx['train']) * 100:.1f}% of training examples ({len(trainidx)}/{len(split_idx['train'])})")
        print(f"Retained {len(valididx)/len(split_idx['valid']) * 100:.1f}% of validation examples ({len(valididx)}/{len(split_idx['valid'])})")
        print(f"Retained {len(testidx)/len(split_idx['test']) * 100:.1f}% of testing examples ({len(testidx)}/{len(split_idx['test'])})")
    else:
        trainidx = split_idx["train"]
        valididx = split_idx["valid"]
        testidx = split_idx["test"]

    train_loader = DataLoader(dataset[trainidx], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[valididx], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset[testidx], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    nodetypes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'typeidx2type.csv.gz'))
    nodeattributes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'attridx2attr.csv.gz'))

    print(nodeattributes_mapping)

    ### Encoding node features into emb_dim vectors.
    ### The following three node features are used.
    # 1. node type
    # 2. node attribute
    # 3. node depth
    node_encoder = ASTNodeEncoder(args.emb_dim, num_nodetypes=len(nodetypes_mapping['type']),
                                  num_nodeattributes=len(nodeattributes_mapping['attr']), max_depth=20)

    if args.gnn == 'gin':
        model = GNN(num_vocab=len(vocab2idx), max_seq_len=args.max_seq_len, node_encoder=node_encoder,
                    num_layer=args.num_layer, gnn_type='gin', emb_dim=args.emb_dim, drop_ratio=args.drop_ratio,
                    virtual_node=False, residual=args.residual).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(num_vocab=len(vocab2idx), max_seq_len=args.max_seq_len, node_encoder=node_encoder,
                    num_layer=args.num_layer, gnn_type='gin', emb_dim=args.emb_dim, drop_ratio=args.drop_ratio,
                    virtual_node=True, residual=args.residual).to(device)
    elif args.gnn == 'gcn':
        model = GNN(num_vocab=len(vocab2idx), max_seq_len=args.max_seq_len, node_encoder=node_encoder,
                    num_layer=args.num_layer, gnn_type='gcn', emb_dim=args.emb_dim, drop_ratio=args.drop_ratio,
                    virtual_node=False, residual=args.residual).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(num_vocab=len(vocab2idx), max_seq_len=args.max_seq_len, node_encoder=node_encoder,
                    num_layer=args.num_layer, gnn_type='gcn', emb_dim=args.emb_dim, drop_ratio=args.drop_ratio,
                    virtual_node=True, residual=args.residual).to(device)
    else:
        raise ValueError('Invalid GNN type')

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f'#Params: {sum(p.numel() for p in model.parameters())}')

    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        loss = train(model, device, train_loader, optimizer)

        print('Evaluating...')
        if (epoch-1) % args.trainevalinter == 0:
            train_perf = eval(model, device, train_loader, evaluator,
                              arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab))
        else:
            train_perf = {}
        valid_perf = eval(model, device, valid_loader, evaluator,
                          arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab))
        test_perf = eval(model, device, test_loader, evaluator,
                         arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab))

        results = {'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf}
        # print(json.dumps(results, indent=3))

        d = {}
        d["loss"] = loss
        for result_k, result_v in results.items():
            for metric_k, metric_v in result_v.items():
                d[f"{result_k}_{metric_k}"] = metric_v

        print(json.dumps(d, indent=3))
        wandb.log(d)

        if len(train_perf) == 0:
            train_curve.append(-1)
        else:
            train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

    print('F1')
    best_val_epoch = np.argmax(np.array(valid_curve))
    best_train = max(train_curve)
    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    settings.update({"best_valid_score": valid_curve[best_val_epoch]})
    settings.update({"final_test__score": test_curve[best_val_epoch]})

    wandb.config.update(settings)

    if not args.filename == '':
        result_dict = {'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch],
                       'Train': train_curve[best_val_epoch], 'BestTrain': best_train}
        torch.save(result_dict, args.filename)


if __name__ == "__main__":
    main()