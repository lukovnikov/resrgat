import json
import random
from multiprocessing import Pool

import dgl
import torch
import wandb
# from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import RandomSampler, SequentialSampler
from torch_geometric.data.dataloader import Collater
from torchvision import transforms

from tqdm import tqdm
import argparse
import time
import numpy as np
import pandas as pd
import os

from torch._six import int_classes as _int_classes

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
### importing utils

### for data transform
from ldgnn.cells import ResRGAT, LRGAT, LRTM
from ldgnn.scripts.ogbg_code2.utils import get_vocab_mapping, ASTNodeEncoder
from ldgnn.scripts.ogbg_code2.utils import encode_y_to_arr, decode_arr_to_seq
from ldgnn.scripts.ogbg_code2.conv import GNN_node

multicls_criterion = torch.nn.CrossEntropyLoss()


class AdaptiveBatchSampler(torch.utils.data.sampler.BatchSampler):
    def __init__(self, dataset, batch_size: int, max_nodes: int = np.infty, max_edges: int = np.infty, shuffle=False,
                 acceptable_size=1., drop_last: bool = False, _maxbacklogsize=None) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.batch_size = batch_size
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.drop_last = drop_last
        self.dataset = dataset
        self._maxbacklogsize = _maxbacklogsize if _maxbacklogsize is not None else 5 * batch_size
        self.acceptable_size = min(1., acceptable_size)
        if shuffle:
            # Cannot statically verify that dataset is Sized
            # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
            sampler = RandomSampler(dataset, generator=None)  # type: ignore
        else:
            sampler = SequentialSampler(dataset)
        self.sampler = sampler

        self.toobig = set()

    def __iter__(self):
        batch = []
        backlog = []
        newbacklog = []
        totnodes = 0
        totedges = 0
        # breakit = False
        batchdone = False

        sampleriter = iter(self.sampler)
        while True:
            if len(backlog) != len(set(backlog)):
                raise Exception("duplicates in backlog")
            if len(backlog) > 0:
                idx = backlog.pop(-1)
            else:
                try:
                    idx = next(sampleriter)
                except StopIteration as e:
                    batchdone = True            # backlog fix
                    if len(newbacklog) == 0:
                        break
                    # else:
                    #     print("non-empty backlog")


                    # yield batch
                    # batch = []
                    # totnodes = 0
                    # totedges = 0
                    # backlog = backlog + newbacklog[::-1]
                    # newbacklog = []
                    # if len(backlog) == 0:
                    #     break
                    # breakit = True
                    # if idx == 999:  # TODO remove
                    #     print("idx is 999")
                    #
                    # OLD:
                    # if len(newbacklog) == 0:
                    #     break

            if not batchdone:
                if self.dataset[idx].num_nodes + totnodes < self.max_nodes \
                        and self.dataset[idx].num_edges + totedges < self.max_edges \
                        and len(batch) < self.batch_size:
                    batch.append(idx)
                    totnodes += self.dataset[idx].num_nodes
                    totedges += self.dataset[idx].num_edges
                else:
                    if self.dataset[idx].num_nodes >= self.max_nodes \
                            or self.dataset[idx].num_edges >= self.max_edges:
                        # if example too big to fit alone in a batch, leave it
                        self.toobig.add(idx)
                    else:
                        newbacklog.append(idx)

            if batchdone or len(batch) == self.batch_size \
                    or (
                    totnodes >= self.acceptable_size * self.max_nodes or totedges >= self.acceptable_size * self.max_edges) \
                    or len(newbacklog) + len(backlog) >= self._maxbacklogsize:
                if len(batch) != len(set(batch)):
                    raise Exception("batch contains duplicates")
                yield batch
                batch = []
                totnodes = 0
                totedges = 0
                backlog = backlog + newbacklog[::-1]
                newbacklog = []
                batchdone = False

        if len(batch) > 0 and not self.drop_last:
            if len(batch) != len(set(batch)):
                raise Exception("batch contains duplicates")
            yield batch

    def __len__(self):
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore


class DummyGraph(object):
    def __init__(self, k, numnodes=None):
        super(DummyGraph, self).__init__()
        self.k = k
        self.num_nodes = random.randint(2, 10) if numnodes is None else numnodes
        self.num_edges = self.num_nodes


class DummyDataset(object):
    def __init__(self, n=1000):
        super(DummyDataset, self).__init__()
        self.examples = [DummyGraph(k) for k in range(n)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    def __iter__(self):
        for example in self.examples:
            yield example


def test_adaptive_batch_sampler():
    tries = 100
    for _ in range(tries):
        ds = DummyDataset(n=1000)
        ds.examples[-10:] = [DummyGraph(g.k, 45) for g in ds.examples[-10:]]
        bs = AdaptiveBatchSampler(ds, batch_size=10, max_nodes=50, shuffle=False, acceptable_size=0.9)
        examples = []
        batches = []
        for b in bs:
            examples = examples + b
            batches.append(b)
        ks = [example for example in examples]
        print(len(ks), len(set(ks)))
        # print(set(ks))
        if len(ks) != len(set(ks)):
            print("not the same size!!!")
        if set(ks) != set(list(range(1000))):
            assert set(ks) == set(list(range(1000)))

    print("trying dataloader")
    tries = 100
    for _ in range(tries):
        ds = DummyDataset(n=1000)
        ds.examples[-10:] = [DummyGraph(g.k, 45) for g in ds.examples[-10:]]
        # bs = AdaptiveBatchSampler(ds, batch_size=10, max_nodes=50, shuffle=False, acceptable_size=0.9)
        dl = DataLoader(ds, batch_size=10, max_nodes=50, shuffle=False, collate_fn=lambda x: [xe for xe in x])
        examples = []
        batches = []
        for b in dl:
            examples = examples + b
            batches.append(b)
        ks = [example.k for example in examples]
        print(len(ks), len(set(ks)))
        # print(set(ks))
        if len(ks) != len(set(ks)):
            print("not the same size!!!")
        if set(ks) != set(list(range(1000))):
            assert set(ks) == set(list(range(1000)))


class DataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`[]`)
    """

    def __init__(self, dataset, batch_size=1, max_nodes=np.infty, shuffle=False,
                 follow_batch=[], collate_fn=None,
                 **kwargs):
        """
        batch_size: maximum batch size
        max_nodes:  maximum number of nodes in this batch
        shuffle:
        """
        batchsampler = AdaptiveBatchSampler(dataset, batch_size=batch_size, max_nodes=max_nodes, shuffle=shuffle,
                                            acceptable_size=0.9, drop_last=False)
        if collate_fn is None:
            collate_fn = Collater(follow_batch)
        super(DataLoader,
              self).__init__(dataset, batch_sampler=batchsampler,
                             collate_fn=collate_fn,
                             **kwargs)


class GNN(torch.nn.Module):

    def __init__(self, num_vocab, max_seq_len, node_encoder, numlayers=1, numrepsperlayer=5, num_heads=4,
                 use_sgru=False, emb_dim=300,
                 gnn_type='gin', drop_ratio=0.5, graph_pooling="mean"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.numlayers = numlayers
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.num_vocab = num_vocab
        self.max_seq_len = max_seq_len
        self.graph_pooling = graph_pooling

        self.num_heads = num_heads
        self.use_sgru = use_sgru

        self.node_encoder = node_encoder
        self.emb_drop = torch.nn.Dropout(drop_ratio)

        ### GNN to generate node embeddings
        self.core = ResRGAT(emb_dim, numlayers=numlayers, numrepsperlayer=numrepsperlayer, numrels=10,
                            numheads=num_heads, dropout=drop_ratio, use_sgru=use_sgru)
        # self.core = LRTM(emb_dim, numlayers=numlayers, numrepsperlayer=numrepsperlayer, numrels=10,
        #                     numheads=num_heads, dropout=drop_ratio)
        # self.core = LRGAT(emb_dim, numlayers=numlayers, numrepsperlayer=numrepsperlayer, numrels=10,
        #                     numheads=num_heads, dropout=drop_ratio, use_sgru=use_sgru)

        # self.gnn_node = GNN_node(num_layer, emb_dim, node_encoder, JK = "last", drop_ratio = drop_ratio,
        #                          residual = False, gnn_type = gnn_type)

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                            torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        self.graph_pred_linear_list = torch.nn.ModuleList()

        if graph_pooling == "set2set":
            for i in range(max_seq_len):
                self.graph_pred_linear_list.append(torch.nn.Linear(2 * emb_dim, self.num_vocab))

        else:
            for i in range(max_seq_len):
                self.graph_pred_linear_list.append(torch.nn.Linear(emb_dim, self.num_vocab))

    def forward(self, batched_data):
        '''
            Return:
                A list of predictions.
                i-th element represents prediction at i-th position of the sequence.
        '''
        node_ids, node_depths = batched_data.x, batched_data.node_depth
        node_emb = self.node_encoder(node_ids, node_depths.view(-1))
        node_emb = self.emb_drop(node_emb)

        # convert to dgl graph
        g = dgl.DGLGraph()
        g.add_nodes(batched_data.num_nodes, data={"h": node_emb})
        g.add_edges(batched_data.edge_index[0], batched_data.edge_index[1], data={"id": batched_data.edge_attr[:, 0]})

        # run stack of gnn layers (implemented in dgl)
        self.core(g)
        h_node = g.ndata["h"]
        # h_node = self.gnn_node(batched_data)

        # do pyg pooling and prediction
        h_graph = self.pool(h_node, batched_data.batch)

        pred_list = []
        # for i in range(self.max_seq_len):
        #     pred_list.append(self.graph_pred_mlp_list[i](h_graph))

        for i in range(self.max_seq_len):
            pred_list.append(self.graph_pred_linear_list[i](h_graph))

        return pred_list


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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.)
            optimizer.step()

            loss_accum += loss.item()

    print('Average training loss: {}'.format(loss_accum / (step + 1)))
    return loss_accum / (step + 1)


def eval(model, device, loader, evaluator, arr_to_seq):
    model.eval()
    seq_ref_list = []
    seq_pred_list = []
    numgraphs = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        numgraphs += batch.num_graphs

        if batch.x.shape[0] == 1:
            raise Exception("batch is empty?!")
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

    assert len(seq_ref_list) == numgraphs

    print(f"Evaluated on {numgraphs} graphs.")

    input_dict = {"seq_ref": seq_ref_list, "seq_pred": seq_pred_list}

    return evaluator.eval(input_dict)


def poolmapf(x):  return x.num_nodes


def augment_edge(data):
    '''
        Input:
            data: PyG data object
        Output:
            data (edges are augmented in the following ways):
                data.edge_index: Added next-token edge. The inverse edges were also added.
                data.edge_attr (torch.Long):
                    data.edge_attr[:,0]: whether it is AST edge (0) for next-token edge (1)
                    data.edge_attr[:,1]: whether it is original direction (0) or inverse direction (1)
    '''

    ##### AST edge
    edge_index_ast = data.edge_index
    edge_attr_ast = 1 * torch.ones(edge_index_ast.size(1), 1, dtype=torch.long)

    ##### Inverse AST edge
    edge_index_ast_inverse = torch.stack([edge_index_ast[1], edge_index_ast[0]], dim=0)
    edge_attr_ast_inverse = 2 * torch.ones(edge_index_ast_inverse.size(1), 1, dtype=torch.long)

    ##### Next-token edge

    ## Obtain attributed nodes and get their indices in dfs order
    # attributed_node_idx = torch.where(data.node_is_attributed.view(-1,) == 1)[0]
    # attributed_node_idx_in_dfs_order = attributed_node_idx[torch.argsort(data.node_dfs_order[attributed_node_idx].view(-1,))]

    ## Since the nodes are already sorted in dfs ordering in our case, we can just do the following.
    attributed_node_idx_in_dfs_order = torch.where(data.node_is_attributed.view(-1, ) == 1)[0]

    ## build next token edge
    # Given: attributed_node_idx_in_dfs_order
    #        [1, 3, 4, 5, 8, 9, 12]
    # Output:
    #    [[1, 3, 4, 5, 8, 9]
    #     [3, 4, 5, 8, 9, 12]
    edge_index_nextoken = torch.stack([attributed_node_idx_in_dfs_order[:-1], attributed_node_idx_in_dfs_order[1:]],
                                      dim=0)
    edge_attr_nextoken = 3 * torch.ones(edge_index_nextoken.size(1), 1, dtype=torch.long)

    ##### Inverse next-token edge
    edge_index_nextoken_inverse = torch.stack([edge_index_nextoken[1], edge_index_nextoken[0]], dim=0)
    edge_attr_nextoken_inverse = 4 * torch.ones(edge_index_nextoken.size(1), 1, dtype=torch.long)

    ##### Self-edges
    self_edges = torch.arange(data.num_nodes, dtype=data.edge_index.dtype)
    self_edges = self_edges[None, :].repeat(2, 1)
    edge_attr_self = 5 * torch.ones(self_edges.shape[1], 1, dtype=torch.long)

    data.edge_index = torch.cat(
        [self_edges, edge_index_ast, edge_index_ast_inverse, edge_index_nextoken, edge_index_nextoken_inverse], dim=1)
    data.edge_attr = torch.cat(
        [edge_attr_self, edge_attr_ast, edge_attr_ast_inverse, edge_attr_nextoken, edge_attr_nextoken_inverse],
        dim=0)

    return data


def test_dataloader(dl, ds):
    def example_to_str(x):
        s = " | "
        s += str(x.edge_attr) + " | "
        s += str(x.edge_index) + " | "
        s += str(x.x) + " | "
        s += str(x.y) + " | "
        return s
    ds_strs = [example_to_str(dse) for dse in tqdm(ds)]

    print(len(ds_strs), len(set(ds_strs)))
    a = ds_strs[:]
    b = set()
    c = []
    for ae in a:
        if ae in b:
            c.append(ae)
        else:
            b.add(ae)

    # for ce in c:
    #     print(ce)
    # assert len(ds_strs) == len(set(ds_strs))

    dl_strs = []
    for batch in tqdm(dl):
        for ex in batch.to_data_list():
            dl_strs.append(example_to_str(ex))

    print(len(dl_strs), len(set(dl_strs)))
    # assert len(dl_strs) == len(set(dl_strs))

    assert sorted(dl_strs) == sorted(ds_strs)

    print("Examples iterated by dataloader are complete and unduplicated.")


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbg-code2 data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--gnn', type=str, default='resrgat',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gcn-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--max_seq_len', type=int, default=5,
                        help='maximum sequence length to predict (default: 5)')
    parser.add_argument('--num_vocab', type=int, default=5000,
                        help='the number of vocabulary used for sequence prediction (default: 5000)')
    parser.add_argument('--numlayers', type=int, default=1,
                        help='number of GNN message passing layers (default: 1)')
    parser.add_argument('--numreps', type=int, default=5,
                        help='number of repitions per GNN message passing layer (default: 5)')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='number of GNN message passing layers (default: 4)')
    parser.add_argument('--use_sgru', action="store_true", help="use sgru node update")
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--max_nodes', type=int, default=1000000,
                        help='maximum number of nodes per batch (default: 1000000)')
    parser.add_argument('--epochs', type=int, default=25,
                        help='number of epochs to train (default: 25)')
    parser.add_argument('--random_split', action='store_true',
                        help="Use a fully random split (not project split like original)")
    parser.add_argument('--use15', action='store_true', help="Use only 15% of training data")
    parser.add_argument('--use300', action='store_true', help="Use only 300 examples everywhere")
    parser.add_argument('--notrain', action='store_true', help="Skip training, just eval")
    parser.add_argument('--testdl', action='store_true', help="Test dataloaders")
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
    # add self-edges and features

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

    if args.use300:
        print("using 300 examples")
        perm = torch.randint(len(split_idx["train"]), (300,))
        split_idx["train"] = split_idx["train"][perm]
        perm = torch.randint(len(split_idx["test"]), (300,))
        split_idx["test"] = split_idx["test"][perm]
        perm = torch.randint(len(split_idx["valid"]), (300,))
        split_idx["valid"] = split_idx["valid"][perm]

    print("adding self-edges and edge features")
    # for x in tqdm(dataset):
    #     # x.num_edge_features = 1
    #     extra_edges = torch.arange(x.num_nodes, dtype=x.edge_index.dtype, device=x.edge_index.device)
    #     extra_edges = extra_edges[None, :].repeat(2, 1)
    #     edgefeats = 1 * torch.ones(x.edge_index.shape[1], 1, dtype=torch.long, device=x.edge_index.device)
    #     extra_edgefeats = 2 * torch.ones(extra_edges.shape[1], 1, dtype=torch.long, device=x.edge_index.device)
    #     x.edge_index = torch.cat([x.edge_index, extra_edges], 1)
    #     x.edge_attr = torch.cat([edgefeats, extra_edgefeats], 0)

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

        print(
            f"Retained {len(trainidx) / len(split_idx['train']) * 100:.1f}% of training examples ({len(trainidx)}/{len(split_idx['train'])})")
        print(
            f"Retained {len(valididx) / len(split_idx['valid']) * 100:.1f}% of validation examples ({len(valididx)}/{len(split_idx['valid'])})")
        print(
            f"Retained {len(testidx) / len(split_idx['test']) * 100:.1f}% of testing examples ({len(testidx)}/{len(split_idx['test'])})")
    else:
        trainidx = split_idx["train"]
        valididx = split_idx["valid"]
        testidx = split_idx["test"]

    print(f"Training graphs: {len(split_idx['train'])} \nValidation graphs: {len(split_idx['valid'])} \nTest graphs: {len(split_idx['test'])}")

    train_loader = DataLoader(dataset[trainidx], batch_size=args.batch_size, max_nodes=args.max_nodes, shuffle=True,
                              num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[valididx], batch_size=args.batch_size, max_nodes=args.max_nodes, shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset[testidx], batch_size=args.batch_size, max_nodes=args.max_nodes, shuffle=False,
                             num_workers=args.num_workers)

    if args.testdl:
        # test_dataloader(train_loader, dataset[trainidx])
        test_dataloader(valid_loader, dataset[valididx])
        test_dataloader(test_loader, dataset[testidx])

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

    model = GNN(num_vocab=len(vocab2idx), max_seq_len=args.max_seq_len, node_encoder=node_encoder,
                numlayers=args.numlayers, numrepsperlayer=args.numreps, num_heads=args.num_heads,
                use_sgru=args.use_sgru,
                gnn_type='resrgat', emb_dim=args.emb_dim, drop_ratio=args.drop_ratio).to(device)

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f'#Params: {sum(p.numel() for p in model.parameters())}')

    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')

        if not args.notrain:
            loss = train(model, device, train_loader, optimizer)
        else:
            loss = 0

        print('Evaluating...')
        if (epoch - 1) % args.trainevalinter == 0 and not args.notrain:
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
    # test_adaptive_batch_sampler()
    main()