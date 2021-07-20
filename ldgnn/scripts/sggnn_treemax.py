import collections
import math
import os
import random
import string
from copy import deepcopy
from functools import partial
from itertools import product
from typing import List

import qelos as q
import torch
import numpy as np
import re
import dgl
import ujson
import wandb
from dgl.nn.pytorch.utils import Identity
from nltk import Tree
from torch.nn import init

from torch.utils.data import DataLoader

from ldgnn.cells import SGGNNCell, GGNNCell, RGATCell, RMGATCell, RVGAT, str2act, GRGATCell, RMGAT, RelTransformer, \
    RelGraphLSTM, GRGAT, RelTransformerCell, ResRGAT, ResRGATCell, GatedGCNNet
from ldgnn.nn import GRUEncoder, RNNEncoder, AttentionReadout, local_var
from ldgnn.scripts.sggnn_recall_grad import RGCN, RGATCellStack
from ldgnn.vocab import SequenceEncoder
import networkx as nx
import matplotlib.pyplot as plt


class DatasetSplitProxy(object):
    def __init__(self, data, **kw):
        super(DatasetSplitProxy, self).__init__(**kw)
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def get_distributions(groups, D):
    r_i = torch.nn.Parameter(torch.zeros(len(groups)))
    r_ji = torch.nn.Parameter(torch.zeros(max(D.values()) + 1, len(groups)))
    r_ji_mask = torch.ones_like(r_ji)
    for i in range(len(groups)):
        for k, v in D.items():
            try:
                if abs(int(k)) > int(groups[i] / 2):
                    r_ji_mask[v, i] = 0
            except ValueError as e:
                r_ji_mask[v, i] = 0
    y_j = (r_ji_mask.sum(1) > 0).float()
    y_j = y_j / y_j.sum()
    params = [r_i, r_ji]

    lr = 1.
    iters = 100
    a, b, c = 1., .2, .001
    kl = torch.nn.KLDivLoss(reduction="sum")

    for iter in range(iters):
        for param in params:
            param.grad = None
        p_i = torch.softmax(r_i, 0)
        p_ji = torch.softmax(r_ji + torch.log(r_ji_mask), 0)
        p_j = torch.einsum("ji,i->j", p_ji, p_i)


        entropy_p_i = (torch.log(p_i) * p_i).sum()
        entropy_p_ji = (torch.log(p_ji + 1e-6) * p_ji).sum()
        ce = kl(torch.log(p_j + 1e-6), y_j) + kl(torch.log(y_j + 1e-6), p_j)

        l = a * ce + b * entropy_p_i + c * entropy_p_ji
        l.backward()
        for param in params:
            param.data = param.data  - lr * param.grad

        if iter == 0:
            print(f"Iter {iter}: {ce.item()}, {entropy_p_i.item()}, {entropy_p_ji.item()}")

    print(f"Iter {iter}: {ce.item()}, {entropy_p_i.item()}, {entropy_p_ji.item()}")

    # extract distributions:
    p_i = torch.softmax(r_i, 0).detach().cpu().numpy()
    print(p_i)
    p_ji = torch.softmax(r_ji + torch.log(r_ji_mask), 0).detach().cpu().numpy()
    print(p_j)
    p_jg = {groups[i]: p_ji[:, i] for i in range(len(groups))}
    return p_i, p_jg


def tree_size(t):
    sizes = [tree_size(te) for te in t]
    return sum(sizes) + 1


def tree_depth(t):
    return 1 + max([tree_depth(te) for te in t], default=0)


def gen_data(N=1000, min_depth=6, max_depth=10, p_leaf=60, p_1child=0, p_2children=30, p_3children=10,
             numbers_from=1, numbers_to=20, predictdiff=False, goldmode="all"):

    # build trees
    numchildprobs = np.array([p_leaf, p_1child, p_2children, p_3children])
    numchildprobs = numchildprobs / sum(numchildprobs)

    def sample_label(depth=0):
        return random.randint(numbers_from, numbers_to)

    maxsize = 0
    trees = []
    for i in range(N):
        depth = random.randint(min_depth, max_depth)
        queue = []
        root = Tree(sample_label(1), [])
        _depth = 1
        queue.append(root)
        nextqueue = []
        while _depth < depth - 1:
            if len(queue) == 0:
                queue = nextqueue
                nextqueue = []
                _depth += 1
            random.shuffle(queue)
            # sample how many children
            numchildren = np.random.choice([0, 1, 2, 3], 1, False, numchildprobs)
            if numchildren == 0 and len(queue) == 1 and _depth < depth:
                _numchildprobs = numchildprobs + 0
                _numchildprobs[0] = 0
                _numchildprobs = _numchildprobs / _numchildprobs.sum()
                numchildren = np.random.choice([0, 1, 2, 3], 1, False, _numchildprobs)
            if numchildren > 0 and _depth == depth:
                numchildren = 0
            for _ in range(int(numchildren)):
                node = Tree(sample_label(_depth), [])
                queue[0].append(node)
                nextqueue.append(node)
            queue.pop(0)
        trees.append(root)
        maxsize = max(maxsize, tree_size(root))

    print(f"max size: {maxsize}")

    # do min and max in tree
    def set_min_and_max(t, top_max=None, top_min=None):
        top_max = top_max if top_max is not None else -1000000
        top_min = top_min if top_min is not None else +1000000
        top_max = max(t.label(), top_max)
        top_min = min(t.label(), top_min)
        t._top_max = top_max
        t._top_min = top_min

        if len(t) == 0:
            t._min, t._max = t.label(), t.label()
            t._maxdist = 0
            return t.label(), t.label(), t._maxdist
        minmaxes = [set_min_and_max(te, top_max=top_max, top_min=top_min) for te in t]
        mins, maxes, maxdists = zip(*minmaxes)
        _min, _max = min(mins), max(maxes)
        _min = min(_min, t.label())
        _max = max(_max, t.label())
        t._min = _min
        t._max = _max
        # max dists
        maxdists = [d + 1 for d in maxdists]
        _maxdist = min([d for m, d in zip(maxes, maxdists) if m == _max], default=0)
        t._maxdist = _maxdist
        return _min, _max, _maxdist

    maxdiststats = dict(zip(range(max_depth+1), [0]*(max_depth+1)))

    def do_max_dist_stats(t, stats):
        stats[t._maxdist] += 1
        for te in t:
            stats = do_max_dist_stats(te, stats)
        return stats

    for tree in trees:
        set_min_and_max(tree)
        maxdiststats = do_max_dist_stats(tree, maxdiststats)

    total = 0
    over4 = 0
    over9 = 0
    for k, v in maxdiststats.items():
        if k >= 5:
            over4 += v
        if k >= 10:
            over9 += v
        total += v

    print(f"{100*over9/total:.3f}% needs 10 hops or more")
    print(f"{100*over4/total:.3f}% needs 5 hops or more")

    nodeD = {}
    nodeD["<PAD>"] = 0
    nodeD["<UNK>"] = 1
    for number in range(-numbers_to-1, numbers_to+1):
        nodeD[str(number)] = max(nodeD.values()) + 1
    relD = {
        "<PAD>": 0,
        "<SELF>": 1,
        "<CHILD:1>": 2,
        "<CHILD:2>": 3,
        "<CHILD:3>": 4,
        "<CHILD:1:OF>": 5,
        "<CHILD:2:OF>": 6,
        "<CHILD:3:OF>": 7,
    }

    # convert trees to graphs
    if predictdiff:
        compute_gold = lambda x: x._max - x._top_min
    else:
        compute_gold = lambda x: x._max
    def attach_to_graph(tree, g:dgl.DGLGraph):
        g.add_nodes(1, {"id": torch.tensor([nodeD[str(tree.label())]]),
                        "gold": torch.tensor([nodeD[str(compute_gold(tree))]]),
                        "dist": torch.tensor([tree._maxdist])})
        self_i = g.number_of_nodes() - 1
        g.add_edge(self_i, self_i, {"id": torch.tensor([relD["<SELF>"]])})

        for i, child in enumerate(tree):
            childnode, childnode_i = attach_to_graph(child, g)
            g.add_edge(self_i, childnode_i, {"id": torch.tensor([relD[f"<CHILD:{i+1}>"]])})
            g.add_edge(childnode_i, self_i, {"id": torch.tensor([relD[f"<CHILD:{i+1}:OF>"]])})

        return g, self_i

    gs = []
    for tree in trees:
        g = dgl.DGLGraph()  # create a graph
        g, g_i = attach_to_graph(tree, g)
        if goldmode == "reduced":
            # don't predict nodes for which gold is less than K hops away
            REDGOLD_K = 4
            K = min(REDGOLD_K, torch.max(g.ndata["dist"]).item() / 2)
            g.ndata["gold"] = g.ndata["gold"] * (g.ndata["dist"] >= K) #+ torch.zeros_like(g.ndata["gold"]) * (g.ndata["dist"] < REDGOLD_K)
            # select only fraction P of other nodes to predict
            nonzeros = g.ndata["gold"].nonzero()[:, 0]
            zeroout = torch.randperm(len(nonzeros))
            REDGOLD_P = 0.5     # retain this percentage of labels
            zeroout = zeroout[:max(1, int(len(nonzeros) * (1 - REDGOLD_P)))]
            zeroout = nonzeros[zeroout]
            g.ndata["gold"][zeroout] = 0
            # pmask = torch.rand(g.ndata["gold"].size()) < REDGOLD_P
            # g.ndata["gold"] = g.ndata["gold"] * pmask #+ torch.zeros_like(g.ndata["gold"]) * pmask
        else:
            assert(goldmode == "all")
        assert(g_i == 0)
        gs.append(g)

    return gs, trees, nodeD, relD


class NodeClassificationGraphDataset(object):
    def __init__(self, examples, nodeD, relD, **kw):
        super(NodeClassificationGraphDataset, self).__init__(**kw)
        self.data = {}
        self._gs, self.nodeD, self.relD = examples, nodeD, relD

        self.N = len(examples)

        splits = ["train"] * int(self.N * 0.5) + ["valid"] * int(self.N * 0.25) + ["test"] * int(self.N * 0.25)
        random.shuffle(splits)
        self.build_data(self._gs, splits)

    @property
    def numrels(self):
        return max(self.relD.values()) + 1

    @property
    def numy(self):
        return max(self.nodeD.values()) + 1

    def build_data(self, gs, splits):
        for g, split in zip(gs, splits):
            if split not in self.data:
                self.data[split] = []
            self.data[split].append(g)

    def get_split(self, split:str):
        return DatasetSplitProxy(self.data[split])

    @staticmethod
    def collate_fn(x:List):
        inps = dgl.batch(x)
        return (inps,)

    def dataloader(self, split:str=None, batsize:int=5, shuffle=None):
        if split is None:   # return all splits
            ret = {}
            for split in self.data.keys():
                ret[split] = self.dataloader(batsize=batsize, split=split, shuffle=shuffle)
            return ret
        else:
            assert(split in self.data.keys())
            shuffle = shuffle if shuffle is not None else split in ("train", "train+valid")
            dl = DataLoader(self.get_split(split), batch_size=batsize, shuffle=shuffle, collate_fn=type(self).collate_fn)
            return dl


class GNNModel(torch.nn.Module):
    def __init__(self, vocabsize, embdim, cell, numsteps=10, dropout=0., emb_dropout=0., numy=20, smoothing=0., **kw):
        super(GNNModel, self).__init__(**kw)
        self.vocabsize = vocabsize
        self.cell = cell
        self.hdim = self.cell.hdim
        self.numsteps = numsteps
        self.embdim = embdim

        if isinstance(cell, (ResRGAT, ResRGATCell, RelTransformerCell, RelTransformer)):
            emb_dropout = dropout / 2.

        self.emb = torch.nn.Embedding(vocabsize, self.embdim)
        self.emb_adapter = torch.nn.Linear(embdim, self.hdim, bias=False) if embdim != self.hdim else Identity()
        self.emb_dropout = torch.nn.Dropout(emb_dropout)
        self.dropout = torch.nn.Dropout(dropout)

        self.numy = numy
        self.outlayer = torch.nn.Linear(self.hdim, self.numy)
        self.outfc1 = torch.nn.Linear(self.hdim, self.hdim)
        # self.readout = AttentionReadout(self.cell.hdim, numy, numheads=2)

        self.ce = NodeCE(smoothing=smoothing)
        self.acc = NodeClassificationAccuracy()

    def forward(self, g):
        emb = self.emb(g.ndata["id"])
        if isinstance(self.cell, GatedGCNNet):
            h = self.cell(g, g.ndata["id"], g.edata["id"])
            g.ndata["h"] = h
        else:
            g.ndata["x"] = emb
            g.ndata["h"] = self.emb_dropout(self.emb_adapter(emb))
            if isinstance(self.cell, (RelGraphLSTM,)):
                g.ndata["c"] = g.ndata["h"] + 0
            # g.ndata["h"] = torch.cat([g.ndata["x"],
            #                           torch.zeros(g.ndata["x"].size(0), self.hdim - g.ndata["x"].size(-1), device=g.ndata["x"].device)],
            #                          -1)
            self.cell.init_node_states(g, g.ndata["x"].size(0), device=g.ndata["x"].device)

            # run updates
            self.cell.reset_dropout()

            for step in range(self.numsteps):
                self.cell(g, step=step)

        # region compute predictions and compute losses
        out = g.ndata["h"]
        # if isinstance(self.cell, (RelTransformer,)):
        #     out = self.outfc1(out)
        #     out = torch.tanh(out)
        if isinstance(self.cell, (RGCN, RelTransformerCell, RelTransformer, GatedGCNNet, ResRGAT, ResRGATCell)):  # ResRGATCell, ResRGAT,
            # if ResRGAT or RelTransformer, don't do final dropout because we already set embedding dropout to the same dropout
            pass
        else:
            out = self.dropout(out)
        probs = self.outlayer(out)
        ce = self.ce(g, probs, g.ndata["gold"])
        _, pred = probs.max(-1)
        elem_acc, acc, dists = self.acc(g, pred)
        # endregion
        ret = {}
        ret["outprobs"] = probs
        ret["ce"] = ce
        ret["elem_acc"] = elem_acc
        ret["dists"] = dists
        ret["acc"] = acc
        return ret


class NodeClassificationAccuracy(torch.nn.Module):
    def forward(self, g, pred):
        g = local_var(g)
        same = (pred == g.ndata["gold"]).float()
        mask = (g.ndata["gold"] != 0).float()
        g.ndata["same"] = torch.max(same, (1 - mask))
        g.ndata["sameref"] = torch.ones_like(same)
        allsame = dgl.sum_nodes(g, "same") == dgl.sum_nodes(g, "sameref")
        allsame = allsame.float()

        elemacc = torch.masked_select(same, mask.bool())
        dists = torch.masked_select(g.ndata["dist"], mask.bool())
        acc = allsame
        return elemacc, acc, dists


class NodeCE(torch.nn.Module):
    def __init__(self, mode="logits", inner_reduction="sum", smoothing=0, ignore_index=0, **kw):
        super(NodeCE, self).__init__(**kw)

        self.ignore_index = ignore_index
        self.inner_reduction = inner_reduction

        if smoothing > 0:
            self.ce = q.SmoothedCELoss(mode=mode, smoothing=smoothing, reduction="none", ignore_index=ignore_index)
        else:
            self.ce = q.CELoss(mode=mode, reduction="none", ignore_index=ignore_index)

    def forward(self, g, probs, gold):
        g = local_var(g)
        ce = self.ce(probs, gold)

        goldmask = (gold != self.ignore_index).float()
        if self.inner_reduction == "mean":
            g.ndata["goldmask"] = goldmask

        ce = ce * goldmask
        g.ndata["ce"] = ce
        ex_ce = dgl.sum_nodes(g, "ce")
        if self.inner_reduction == "mean":
            ex_ce = ex_ce / dgl.sum_nodes(g, "goldmask")
        return ex_ce


def run(lr=0.001,  # 0.001
        dropout=0.4,
        dropoutemb=0.2,
        zoneout=0.,
        wreg=0.,
        embdim=30,  # 20
        hdim=50,  # 50
        rdim=-1,
        maxrdim=-1,
        numheads=2,
        numlayers=-1,
        epochs=100,
        n=800,
        batsize=20,
        gradacc=1,
        patience=10,
        useshortcuts=False,
        gpu=-1,
        cell="sggnn",  # "rgcn", "ggnn", "sggnn", "rmgat", "graphlstm"
        innercell="dgru",
        seed=123456,
        noattention=False,
        usevallin=False,
        relmode="default",  # "original" or "prelu" seem to work (see other options in cell)
        gradnorm=3,
        smoothing=0.1,
        cosinelr=False,
        warmup=0,
        noearlystop=False,
        maxdepth=15,
        mindepth=5,
        maxnumber=100,
        predictdiff=False,
        windowhead=False,
        usegradskip=False,
        goldmode="all",         # "all" or "reduced"
        skipatt=False,
        nodamper=False,
        norelcat=False,
        shared=False,
        usesgru=False,
        aggregator="default",       # "default" or "max"
        ):
    # if noattention or nodgru or nonovallin:
    #     dropout = ablationdropout
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    settings = locals().copy()
    q.pp_dict(settings)

    if dropoutemb < 0:
        dropoutemb = dropout

    wandb.init(project="sggnn_treemax", config=settings, reinit=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if gpu < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda", gpu)

    tt = q.ticktock("script")
    tt.tick("making data")
    examples, trees, nodeD, relD, = gen_data(N=n, max_depth=maxdepth, min_depth=mindepth, numbers_to=maxnumber,
                                             predictdiff=predictdiff, goldmode=goldmode)
    maxdepth = max([tree_depth(t) for t in trees])
    maxsize = max([tree_size(t) for t in trees])
    ds = NodeClassificationGraphDataset(examples, nodeD, relD)
    tt.msg(f"{ds.N} total examples, {maxdepth} max depth, {maxsize} max size")
    tt.tock("made data")
    vocsize = ds.numy

    rdim = hdim if rdim == -1 else rdim
    if maxrdim >= 0 and rdim > maxrdim:
        rdim = maxrdim
    seqlen = maxdepth
    numsteps = seqlen + 1
    if numlayers >= 0:  # numlayers function kwarg overrides automatic computation of number of layers
        numsteps = numlayers
    totnumrels = ds.numrels
    print(f"Number of GNN layers/steps: {numsteps}")
    if cell == "rgcn":
        cell = RGCN(hdim, numsteps, dropout=dropout, numrels=totnumrels, rdim=rdim, shared=False, aggregator=aggregator)
    if cell == "sharedrgcn":
        cell = RGCN(hdim, numsteps, dropout=dropout, numrels=totnumrels, rdim=rdim, shared=True, aggregator=aggregator)
    if cell == "ggnn":
        cell = GGNNCell(hdim, dropout=dropout, numrels=totnumrels, rdim=rdim, use_dgru=innercell=="dgru")
    elif cell == "sggnn":
        cell = SGGNNCell(hdim, dropout=dropout, zoneout=zoneout, numheads=numheads, cell=innercell,
                         no_attention=noattention, usevallin=usevallin, relmode=relmode,
                         numrels=totnumrels, rdim=rdim, windowhead=windowhead)
    elif cell == "graphlstm":
        cell = RelGraphLSTM(hdim, dropout=dropout, numrels=totnumrels, rdim=rdim, relmode=relmode)
    elif cell == "grgat":
        cell = GRGATCell(hdim, numrels=totnumrels, numheads=numheads, dropout=dropout,
                            cell=innercell, rdim=rdim, usevallin=usevallin, relmode=relmode,
                         noattention=noattention, skipatt=skipatt, nodamper=nodamper,
                         usegradskip=usegradskip, cat_rel=not norelcat, aggregator=aggregator)
    elif cell == "resrgat":
        cell = ResRGAT(hdim, numlayers=1, numrepsperlayer=numsteps,
                       numrels=totnumrels, numheads=numheads, dropout=0., dropout_act=dropout,
                       rdim=rdim, usevallin=usevallin,
                       skipatt=skipatt, use_sgru=usesgru,
                       cat_rel=not norelcat)
    elif cell == "reltransformer":
        _numlayers = _numrepsperlayer = 1
        if shared:
            _numrepsperlayer = numsteps
        else:
            _numlayers = numsteps
        cell = RelTransformer(hdim, numlayers=_numlayers, numrepsperlayer=_numrepsperlayer, innercell=innercell,
                              numrels=totnumrels, numheads=numheads, dropout=dropout, rdim=hdim, relmode=relmode)
    elif cell == "rvgat":
        cell = RVGAT(hdim, dropout=dropout, numheads=numheads, cell=innercell, usevallin=usevallin, relmode=relmode, numrels=totnumrels)
    elif cell == "rmgat":
        cell = RMGAT(hdim, numsteps=numsteps, numrels=totnumrels, dropout=dropout, numheads=numheads, act=innercell)
    elif cell == "rgat":
        dropout = 0.1
        cell = RGATCell(hdim, hdim, num_heads=numheads, feat_drop=0, attn_drop=0., residual=True, numrels=totnumrels)
        cells = [cell] * numsteps
        cell = RGATCellStack(cells, feat_drop=dropout)
    elif cell == "gatedgcn":
        netparams = {
            "num_atom_type": vocsize,
            "num_bond_type": totnumrels,
            "emb_dim": embdim,
            "hidden_dim": hdim,
            "in_feat_dropout": dropout,
            "dropout": dropout,
            "L": numsteps,
            "readout": "none",
            "batch_norm": True,
            "residual": True,
            "edge_feat": True,
            "device": device,
            "pos_enc": False,
            "shared": shared,
        }
        cell = GatedGCNNet(netparams)

    m = GNNModel(vocsize, embdim, cell, numsteps=numsteps, dropout=dropout, emb_dropout=dropoutemb,
                 numy=ds.numy, smoothing=smoothing)
    print(m)
    numparams = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Total number of parameters: {numparams}")

    # dl = ds.dataloader("train", batsize=batsize, shuffle=True)
    # batch = iter(dl).next()
    # print(batch)
    # y = m(batch[0])
    # print(y.size())

    optim = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wreg)

    losses = [q.MetricWrapper(q.SelectedLinearLoss(i)) for i in ["ce", "elem_acc", "acc"]]
    vlosses = [q.MetricWrapper(q.SelectedLinearLoss(i)) for i in ["ce", "elem_acc", "acc"]]


    # 6. define training function
    # gradnorm clipping
    clipgradnorm = lambda: torch.nn.utils.clip_grad_norm_(m.parameters(), gradnorm)

    # lr schedule
    if cosinelr:
        # lrsched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)
        lrsched = q.sched.Linear(steps=warmup) >> q.sched.Cosine(steps=epochs-warmup) >> 0
    else:
        lrsched = q.sched.Linear(steps=warmup) >> 1
    lrsched = lrsched.get_lr_sched(optim)

    # clipgradnorm = lambda: None
    trainbatch = partial(q.train_batch, on_before_optim_step=[clipgradnorm],
                         gradient_accumulation_steps=gradacc)
    eyt = q.EarlyStopper(vlosses[1], patience=patience, more_is_better=True, min_epochs=100,        # TODO was 50
                         remember_f=lambda: deepcopy(m))
    def wandb_logger():
        d = {}
        for name, loss in zip(["ce", "elem_acc", "acc"], losses):
            d["train_"+name] = loss.get_epoch_error()
        for name, loss in zip(["ce", "elem_acc", "acc"], vlosses):
            d["valid_"+name] = loss.get_epoch_error()
        wandb.log(d)
    on_end = [lambda: lrsched.step()]
    on_end_v = [lambda: wandb_logger()]
    checkstop = []
    if not noearlystop:
        on_end_v.append(lambda: eyt.on_epoch_end())
        checkstop.append(lambda: eyt.check_stop())
    trainepoch = partial(q.train_epoch, model=m, dataloader=ds.dataloader("train", batsize=batsize, shuffle=True),
                         optim=optim, losses=losses, device=device, _train_batch=trainbatch,
                         on_end=on_end)
    validepoch = partial(q.test_epoch, model=m, dataloader=ds.dataloader("valid", batsize=batsize, shuffle=False),
                         losses=vlosses, device=device,
                         on_end=on_end_v)

    q.run_training(trainepoch, validepoch, max_epochs=epochs,
                   check_stop=checkstop)

    # vlosses = [q.MetricWrapper(q.SelectedLinearLoss(i)) for i in ["ce", "elem_acc", "acc"]]
    tlosses = [q.MetricWrapper(q.SelectedLinearLoss(i)) for i in ["ce", "elem_acc", "acc"]]

    if not noearlystop and epochs > patience:
        tt.msg("reloading best")
        m = eyt.remembered
    q.test_epoch(model=m, dataloader=ds.dataloader("train", batsize=batsize, shuffle=False),
                         losses=losses, device=device)
    q.test_epoch(model=m, dataloader=ds.dataloader("valid", batsize=batsize, shuffle=False),
                         losses=vlosses, device=device)
    q.test_epoch(model=m, dataloader=ds.dataloader("test", batsize=batsize, shuffle=False),
                         losses=tlosses, device=device)

    valid_preds = q.eval_loop(model=m, dataloader=ds.dataloader("valid", batsize=batsize, shuffle=False))
    dists = [x["dists"] for x in valid_preds[1][0]]
    accs = [x["elem_acc"] for x in valid_preds[1][0]]
    dists = torch.cat(dists, 0)
    accs = torch.cat(accs, 0)

    for dist in list(dists.unique().cpu().numpy()):
        dist_acc = ((dists == dist) * accs).sum() / (dists == dist).sum()
        print(f"Accuracy for distance {dist}: {dist_acc.item()} (total: {(dists == dist).sum()})")

    settings.update({"train_acc": losses[2].get_epoch_error()})
    settings.update({"valid_acc": vlosses[2].get_epoch_error()})
    settings.update({"test_acc": tlosses[2].get_epoch_error()})
    settings.update({"elem_train_acc": losses[1].get_epoch_error()})
    settings.update({"elem_valid_acc": vlosses[1].get_epoch_error()})
    settings.update({"elem_test_acc": tlosses[1].get_epoch_error()})
    wandb.config.update(settings)
    print(ujson.dumps(settings, indent=4))
    return settings


def run_experiment(n=800,
                   cell="grgat",
                   innercell="default",
                   usevallin=False,
                   noattention=False,
                   patience=10,
                   gpu=-1,
                   seed=-1,
                   lr=-1.,
                   batsize=20,
                   gradacc=1,
                   numlayers=-1,
                   useshortcuts=False,
                   hdim=-1,
                   embdim=30,
                   rdim=-1,
                   maxrdim=-1,
                   dropout=-1.,
                   dropoutemb=-1.,
                   wreg=0.,
                   epochs=-1,
                   noearlystop=False,
                   smoothing=0.1,
                   relmode="default",
                   goldmode="all",      # "all" or "reduced"
                   gradnorm=3.,
                   maxdepth=16,     # was 15
                   mindepth=5,
                   maxnumber=100,
                   predictdiff=False,
                   windowhead=False,
                   numheads=2,
                   usegradskip=False,
                   skipatt=False,
                   nodamper=False,
                   norelcat=False,
                   aggregator="default",
                   shared="default",
                   usesgru=False,
                   ):
    seqlen = maxdepth
    maxnumlayers = seqlen + 1
    _innercell = "default"
    if innercell == "default":
        if cell == "sggnn":
            innercell = "dgru"
        elif cell == "ggnn":
            innercell = "gru"
        elif cell == "rmgat":
            innercell = "none"
        elif cell == "grgat":
            innercell = "dgru"
    else:
        _innercell = innercell

    if relmode == "default":
        if cell == "reltransformer":
            relmode = "default"
        elif cell == "graphlstm":
            relmode = "default"
        elif cell == "resrgat":
            relmode = "default"
        elif cell != "grgat":
            relmode = "gatedcatpremap"

    if cell == "sggnn":
        ranges = {
            "dropout": [0, .1, .25, .5],
            "epochs": [200],
            "hdim": [150, 300],
            "numlayers": [maxnumlayers],
            "n": [n],
            "lr": [0.00075, 0.0005],
        }
    if cell == "grgat":         # -cell grgat -dropout 0.1 -dropoutemb 0.1 -hdim 150  -epochs 200 -patience 20 -lr 0.001 -numlayers 17 -gpu 0
        ranges = {
            "dropout": [0, .1, .25, .5],
            "epochs": [200],
            "hdim": [150, 300],
            "numlayers": [maxnumlayers],
            "n": [n],
            "lr": [0.001, 0.0003333, 0.0001],
            "innercell": ["dgru", "gate", "none", "gru"]
        }
    if cell == "resrgat":         # -cell grgat -dropout 0.1 -dropoutemb 0.1 -hdim 150  -epochs 200 -patience 20 -lr 0.001 -numlayers 17 -gpu 0
        ranges = {
            "dropout": [0, .1, .25, .5],
            "epochs": [200],
            "hdim": [150],
            "numlayers": [maxnumlayers],
            "n": [n],
            "lr": [0.001, 0.0003333, 0.0001],
            "dropoutemb":[-1.],
        }
    elif  cell == "gatedgcn":
        ranges = {
            "dropout": [0, .1, .25, .5],
            "epochs": [200],
            "hdim": [150, 300],
            "numlayers": [10, maxnumlayers],
            "n": [n],
            "lr": [0.001, 0.0003333, 0.0001],
            "dropoutemb":[-1.],
        }
        if shared != "default":
            ranges["shared"] = [bool(shared)]
    elif cell == "graphlstm":         # -cell grgat -dropout 0.1 -dropoutemb 0.1 -hdim 150  -epochs 200 -patience 20 -lr 0.001 -numlayers 17 -gpu 0
        ranges = {
            "dropout": [0, .1, .25, .5],
            "epochs": [200],
            "hdim": [150],
            "numlayers": [maxnumlayers],
            "n": [n],
            "lr": [0.001, 0.0003333, 0.0001],
        }
    elif cell=="reltransformer":
        # ranges = {
        #     "dropout": [0, .1, .25, .5],
        #     "epochs": [200],
        #     "hdim": [150, 300],
        #     "numlayers": [maxnumlayers],
        #     "n": [n],
        #     "lr": [0.001, 0.0003333, 0.0001],
        #     "innercell": ["none"],
        #     "shared": [True, False],
        # }
        ranges = {
            "dropoutemb": [-1.],
            "dropout": [0., 0.1, 0.25, 0.5],
            "epochs": [200],
            "hdim": [150],
            "numlayers": [maxnumlayers],
            "n": [n],
            "lr": [0.0001, 0.000333],
            "innercell": ["none"],
            "shared": [True, False],
        }
        if shared != "default":
            ranges["shared"] = [bool(shared)]
    elif cell == "rgcn" or cell == "sharedrgcn":    # -cell sharedrgcn -dropout 0.1 -dropoutemb 0.1 -hdim 300 -epochs 200 -patience 20 -lr 0.001 -seed 12345678 -numlayers 17 -gpu 0 -maxrdim 150 -batsize 10
        ranges = {
            "dropout": [0., .1, .25, .5],
            "epochs": [200],
            "hdim": [150, 300],
            "numlayers": [maxnumlayers, 10],
            "n": [n],
            "lr": [0.001, 0.0003333, 0.0001],
        }
    elif cell == "ggnn":        # -cell ggnn -dropout 0.1 -dropoutemb 0.1 -hdim 300 -maxrdim 150 -epochs 200 -patience 20 -lr 0.00075 -seed 12345678 -numlayers 17 -gpu 0 -batsize 10
        ranges = {
            "dropout": [0, .1, .25, .5],
            "epochs": [200],
            "hdim": [150, 300],
            "numlayers": [maxnumlayers, 10],
            "n": [n],
            "lr": [0.001, 0.0003333, 0.0001],
            }
    elif cell == "rmgat":
        ranges = {
            "dropout": [0, .1, .25, .5],
            "epochs": [200],
            "hdim": [100],
            "numlayers": [10, maxnumlayers],
            "n": [n],
            "lr": [0.001, 0.0003333, 0.0001],
            "innercell": ["none", "relu"],
            }
    ranges["dropoutemb"] = [0.0, 0.1] if "dropoutemb" not in ranges else ranges["dropoutemb"]   #, 0.2]
    # ranges["seed"] = [12345678, 34989987, 76850409, 45739488, 87646464]
    ranges["seed"] = [87646446, 34989987, 76850409]
    ranges["innercell"] = [innercell] if "innercell" not in ranges else ranges["innercell"]
    # ranges["numheads"] = [2, 4]
    ranges["numheads"] = [2]

    if seed >= 0:
        ranges["seed"] = [seed]
    if lr >= 0.:
        ranges["lr"] = [lr]
    if hdim >= 0:
        ranges["hdim"] = [hdim]
    if dropout >= 0:
        ranges["dropout"] = [dropout]
    if dropoutemb >= 0:
        ranges["dropoutemb"] = [dropoutemb]
    if numlayers >= 0:
        ranges["numlayers"] = [numlayers]
    if n >= 0:
        ranges["n"] = [n]
    if epochs >= 0:
        ranges["epochs"] = [epochs]
    if _innercell != "default":
        ranges["innercell"] = [_innercell]
    if numheads >= 0:
        ranges["numheads"] = [numheads]
    ranges["wreg"] = [wreg]
    print(__file__)
    celln = cell
    if cell == "sggnn":
        celln = cell + "-" + innercell
    suffix = ""
    if useshortcuts:
        suffix = ".withshortcuts"
    p = __file__ + f".{celln}.x{n}{suffix}"
    q.run_experiments(run, ranges, path_prefix=p,
                      cell=cell, embdim=embdim,
                      gpu=gpu, batsize=batsize, gradacc=gradacc, seed=seed,
                      noattention=noattention, usevallin=usevallin, patience=patience, relmode=relmode,
                      lr=lr, useshortcuts=useshortcuts, rdim=rdim, maxrdim=maxrdim, smoothing=smoothing,
                      noearlystop=noearlystop, gradnorm=gradnorm,
                      maxdepth=maxdepth, maxnumber=maxnumber, mindepth=mindepth, predictdiff=predictdiff,
                      windowhead=windowhead, usegradskip=usegradskip, goldmode=goldmode, usesgru=usesgru,
                      skipatt=skipatt, nodamper=nodamper, norelcat=norelcat, aggregator=aggregator)


if __name__ == '__main__':
    # q.argprun(run)
    q.argprun(run_experiment)