import collections
import json
import math
import pickle
import random
import string
from copy import deepcopy
from functools import partial

import qelos as q
import torch
import numpy as np
import re
import dgl
import ujson
import wandb
from dgl.nn.pytorch import edge_softmax
from dgl.nn.pytorch.utils import Identity
from torch.nn import init

from torch.utils.data import DataLoader

from ldgnn.cells import SGGNNCell, GGNNCell, RGATCell, RVGAT, RMGATCell, RMGAT, GRGATCell, DeepRGCN, GRGAT, GGNN, \
    RelTransformer, RelGraphLSTM, GatedGCNNet, ResRGAT, LRGAT
from ldgnn.nn import GRUEncoder, RNNEncoder, SimpleDGRUCell, DSDGRUCell, AttentionDropout
from ldgnn.relgraphconv import RelGraphConv
from ldgnn.vocab import SequenceEncoder


# region data
class DatasetSplitProxy(object):
    def __init__(self, data, **kw):
        super(DatasetSplitProxy, self).__init__(**kw)
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def gen_data(maxlen=10, NperY=2, within=-1):
    # random.seed(74850303)
    letters = [chr(x) for x in range(97, 123)]
    numbers = [chr(x) for x in range(48, 58)]
    uppercase = [chr(x) for x in range(65, 90)]
    # print(letters)
    # print(numbers)
    # print(uppercase)

    if within > 0:
        y = [l for i in range(NperY) for l in numbers] \
            + [l for i in range(NperY) for l in uppercase]
    else:
        y = [l for i in range(NperY) for l in letters] \
            + [l for i in range(NperY) for l in numbers] \
            + [l for i in range(NperY) for l in uppercase]

    ret = []
    for i in range(len(y)):
        rete = ""
        if y[i] in letters:
            for j in range(maxlen-1):
                rete += random.choice(letters)
            rete = y[i] + rete
        else:
            position = random.choice(list(range(0 if within <= 0 else maxlen - within, maxlen-1)))    # position at which y occurs
            if y[i] in numbers:
                allowed_before = letters + uppercase
                allowed_after = letters + uppercase + numbers
            else:   # y[i] is uppercase
                allowed_before = letters
                allowed_after = letters + uppercase
            for j in range(maxlen):
                if j < position:
                    rete += random.choice(allowed_before)
                elif j == position:
                    rete += y[i]
                else:
                    rete += random.choice(allowed_after)

        # check
        _y = None
        for j in rete:
            if _y in numbers:
                pass
            elif _y in uppercase:
                if j in numbers:
                    _y = j
            elif _y in letters:
                if j in numbers + uppercase:
                    _y = j
            else:
                assert(_y is None)
                _y = j
        if _y != y[i]:
            assert(_y == y[i])
        ret.append(rete)
    # for _y, _ret in zip(y, ret):
    #     print(_ret, _y)
    return ret, y


class ConditionalRecallDataset(object):
    def __init__(self, maxlen=10, NperY=10, within=-1, **kw):
        super(ConditionalRecallDataset, self).__init__(**kw)
        self.data = {}
        self.NperY, self.maxlen = NperY, maxlen
        self._seqs, self._ys = gen_data(self.maxlen, self.NperY, within=within)
        self.encoder = SequenceEncoder(tokenizer=lambda x: list(x))

        for seq, y in zip(self._seqs, self._ys):
            self.encoder.inc_build_vocab(seq)
            self.encoder.inc_build_vocab(y)

        self.N = len(self._seqs)
        N = self.N

        splits = ["train"] * int(N * 0.8) + ["valid"] * int(N * 0.1) + ["test"] * int(N * 0.1)
        random.shuffle(splits)

        self.encoder.finalize_vocab()
        self.build_data(self._seqs, self._ys, splits)

        # check test, valid overlap with train
        unique_test_examples = set([str(x[0].numpy()) for x in self.data["test"]])
        unique_valid_examples = set([str(x[0].numpy()) for x in self.data["valid"]])
        unique_train_examples = set([str(x[0].numpy()) for x in self.data["train"]])
        print("Training examples:")
        print(f"\tUnique: {len(unique_train_examples)} of total {len(self.data['train'])}")
        print("Valid examples:")
        print(f"\tUnique: {len(unique_valid_examples)} of total {len(self.data['valid'])}")
        print(f"\tOccur in train: {len(unique_train_examples & unique_valid_examples)}")
        print("Test examples:")
        print(f"\tUnique: {len(unique_test_examples)} of total {len(self.data['test'])}")
        print(f"\tOccur in train: {len(unique_train_examples & unique_test_examples)}")

    def build_data(self, seqs, ys, splits):
        for seq, y, split in zip(seqs, ys, splits):
            seq_tensor = self.encoder.convert(seq, return_what="tensor")
            y_tensor = self.encoder.convert(y, return_what="tensor")
            if split not in self.data:
                self.data[split] = []
            self.data[split].append((seq_tensor[0], y_tensor[0][0]))

    def get_split(self, split:str):
        return DatasetSplitProxy(self.data[split])

    def dataloader(self, split:str=None, batsize:int=5, shuffle=None):
        if split is None:   # return all splits
            ret = {}
            for split in self.data.keys():
                ret[split] = self.dataloader(batsize=batsize, split=split, shuffle=shuffle)
            return ret
        else:
            assert(split in self.data.keys())
            shuffle = shuffle if shuffle is not None else split in ("train", "train+valid")
            dl = DataLoader(self.get_split(split), batch_size=batsize, shuffle=shuffle)
            return dl
# endregion


class RGATCellStack(torch.nn.Module):
    def __init__(self, cells, feat_drop=0., **kw):
        super(RGATCellStack, self).__init__(**kw)
        self.cells = torch.nn.ModuleList(cells)
        self.feat_drop = torch.nn.Dropout(feat_drop)
        self.hdim = self.cells[0].hdim
        self.register_buffer("dropout_mask", torch.ones(self.hdim))

    def reset_dropout(self):
        device = self.dropout_mask.device
        ones = torch.ones(self.hdim, device=device)
        self.dropout_mask = self.feat_drop(ones).clamp_max(1)
        for cell in self.cells:
            cell.dropout_mask = self.dropout_mask

    def init_node_states(self, g, batsize, device):
        g.ndata["red"] = torch.zeros(batsize, self.hdim, device=device)

    def forward(self, g, step=0):
        cell = self.cells[step]
        g.update_all(cell.message_func, cell.reduce_func, cell.apply_node_func)


class RGCN(torch.nn.Module):
    def __init__(self, hdim, numlayers, dropout=0, numrels=24, rdim=None, num_bases=None,
                 use_self_loop=False, skip=False, shared=False, aggregator="default"):
        super(RGCN, self).__init__()
        if aggregator == "default":
            aggregator = "mean"
        self.hdim = hdim
        self.rdim = self.hdim if rdim is None else rdim
        self.num_rels = numrels
        self.num_bases = num_bases
        self.numlayers = numlayers
        self.use_self_loop = use_self_loop

        # create rgcn layers
        self.layers = torch.nn.ModuleList()
        effectivenumlayers = self.numlayers if not shared else 1
        for idx in range(effectivenumlayers):
            layer = RelGraphConv(self.hdim, self.hdim, self.num_rels, "simple",
                         self.num_bases, activation=torch.relu, self_loop=self.use_self_loop,
                         dropout=dropout, rdim=self.rdim, aggr=aggregator)
            self.layers.append(layer)

        self.dropout = torch.nn.Dropout(dropout) #dropout)
        self.register_buffer("dropout_mask", torch.ones(self.hdim))
        self.skip = skip

    def reset_dropout(self):
        device = self.dropout_mask.device
        # ones = torch.ones(self.hdim, device=device)
        # self.dropout_mask = self.dropout(ones).clamp_max(1)

    def init_node_states(self, g, batsize, device):
        g.ndata["red"] = torch.zeros(batsize, self.hdim, device=device)

    def forward(self, g, step=None):
        if step is None:
            for layernr in range(self.numlayers):
                g = self.forward(g, step=layernr)
        else:
            _h = g.ndata["h"]
            r = g.edata["id"]
            step = min(step, len(self.layers) - 1)
            layer = self.layers[step]
            h = layer(g, _h, r)
            if self.skip:
                h = (h + _h)/2
            g.ndata["h"] = h
        return g


class GRUGNN(torch.nn.Module):
    """
    GRU implemented as a GNN
    """
    def __init__(self, indim, hdim, dropout=0., numrels=16, **kw):
        super(GRUGNN, self).__init__(**kw)
        self.hdim = hdim
        self.indim = indim
        self.node_gru = torch.nn.GRUCell(self.indim, self.hdim)

        self.dropout = torch.nn.Dropout(dropout)

    def message_func(self, edges):
        msg = edges.src["h"]
        return {"msg": msg}

    def reduce_func(self, nodes):
        assert(nodes.mailbox["msg"].size(1) == 1)
        red = nodes.mailbox["msg"].sum(1)
        return {"red": red}

    def apply_node_func(self, nodes):
        h = self.node_gru(self.dropout(nodes.data["x"]), nodes.data["red"])
        return {"h": h}


class SeqGGNN(torch.nn.Module):
    useposemb = False
    def __init__(self, vocab, embdim, cell, numsteps=10, maxlen=10, emb_dropout=0., dropout=0., add_self_edge=True, **kw):
        super(SeqGGNN, self).__init__(**kw)
        self.vocab = vocab
        self.cell = cell
        self.hdim = self.cell.hdim
        self.numsteps = numsteps
        self.embdim = embdim
        self.maxlen = maxlen
        self.add_self_edge = add_self_edge

        if not isinstance(self.cell, (GatedGCNNet,)):
            self.emb = torch.nn.Embedding(vocab.number_of_ids(), self.embdim)
            self.emb_adapter = torch.nn.Linear(embdim, self.hdim, bias=False) if embdim != self.hdim else Identity()
            self.posemb = torch.nn.Embedding(maxlen, self.embdim)
            self.emb_dropout = torch.nn.Dropout(emb_dropout)
            self.dropout = torch.nn.Dropout(dropout)
        outlindim = self.hdim
        self.outlin = torch.nn.Linear(outlindim, vocab.number_of_ids())
        self.outfc1 = torch.nn.Linear(outlindim, outlindim)
        self.gradprobes = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(1, 2))])

    def forward(self, x:torch.LongTensor, gradprobe=True):
        # region create graph
        g = dgl.DGLGraph().to(x.device)
        g.add_nodes(x.size(0) * x.size(1))

        if isinstance(self.cell, (GatedGCNNet,)):
            g.ndata["id"] = x.view(-1)
        else:
            embs = self.emb(x)
            if gradprobe:
                gp = torch.nn.Parameter(torch.zeros_like(embs))
                self.gradprobes[0] = gp
                embs = embs + gp
            embs = self.emb_adapter(embs)
            if self.useposemb:
                positions = torch.arange(1, self.maxlen+1, device=x.device)
                positions = positions[None, :embs.size(1)].repeat(x.size(0), 1)
                posembs = self.posemb(positions)
                embs = torch.cat([embs, posembs], 2)
                # embs = embs + posembs
            embs = self.emb_dropout(embs)
            _embs = embs.view(-1, embs.size(-1))
            g.ndata["x"] = _embs
            # g.ndata["h"] = torch.zeros(_embs.size(0), self.hdim, device=x.device)
            g.ndata["h"] = _embs
            if isinstance(self.cell, (RelGraphLSTM)):
                g.ndata["c"] = g.ndata["h"]
            self.cell.init_node_states(g, _embs.size(0), device=_embs.device)

        xlen = x.size(1)
        for i in range(x.size(0)):
            if self.add_self_edge:  # self edges
                for j in range(xlen):
                    g.add_edge(i * xlen + j, i * xlen + j, {"id": torch.tensor([3], device=x.device)})
            for j in range(xlen-1):
                g.add_edge(i * xlen + j, i * xlen + j + 1, {"id": torch.tensor([1], device=x.device)})
                # g.add_edge(i * xlen + j + 1, i * xlen + j, {"id": torch.tensor([2], device=x.device)})

        # endregion

        if isinstance(self.cell, (GatedGCNNet,)):
            out = self.cell(g, g.ndata["id"], g.edata["id"])
        else:
            # run updates
            self.cell.reset_dropout()
            # for step in range(self.numsteps):
            self.cell(g)

            # region extract predictions
            if False: #isinstance(self.cell, DSSGGNNCell):
                out = torch.cat([g.ndata["h"], g.ndata["c"]], -1)
            else:
                out = g.ndata["h"]
        out = out.view(x.size(0), x.size(1), -1)

        lastout = out[:, -1, :]
        if isinstance(self.cell, RelTransformer):
            lastout = self.outfc1(lastout)
            lastout = torch.tanh(lastout)
        if not isinstance(self.cell, (RGCN, RelTransformer, GRGAT, ResRGAT, GatedGCNNet, LRGAT, DeepRGCN)):
            lastout = self.dropout(lastout)
        pred = self.outlin(lastout)
        # endregion
        return pred


class ClassificationAccuracy(torch.nn.Module):
    def forward(self, probs, target):
        _, pred = probs.max(-1)
        same = pred == target
        ret = same.float().sum() / same.size(0)
        return ret


class GradAnalyzer(object):
    def __init__(self, model, **kw):
        super(GradAnalyzer, self).__init__(**kw)
        self.model = model
        self.data = []      # data per epoch
        self.inpgrads = []     # input grads per epoch
        self.current_epoch_data = None
        self.current_epoch_inpgrads = None

    def on_batch_start_args(self, batch=None, batch_number=None, current_epoch=None):
        """ Store batch data in self.data """
        if self.current_epoch_data is None:
            self.current_epoch_data = batch
        else:
            self.current_epoch_data = [torch.cat([a.cpu(), b.cpu()], 0) for a, b in zip(self.current_epoch_data, batch)]

    def on_epoch_end(self):
        self.data.append([e.cpu().numpy() for e in self.current_epoch_data])
        self.inpgrads.append(self.current_epoch_inpgrads.cpu().numpy())
        self.current_epoch_data = None
        self.current_epoch_inpgrads = None

    def on_before_optim_step(self):
        """ Get the input gradients from model and store them in self.grads """
        inpgradnorms = self.model.gradprobes[0].grad.norm(2, -1).cpu()
        if self.current_epoch_inpgrads is None:
            self.current_epoch_inpgrads = inpgradnorms
        else:
            self.current_epoch_inpgrads = torch.cat([self.current_epoch_inpgrads, inpgradnorms], 0)

    def finalize(self):
        allgradnorms = np.stack(self.inpgrads, 0)
        meangradnorms = allgradnorms.mean(1)    # (epoch, seqlen)
        meangradnorms_struct = []
        for i in range(meangradnorms.shape[0]):
            b = []
            for j in range(meangradnorms.shape[1]):
                b.append(float(np.log(meangradnorms[i, j])/np.log(10)))
            meangradnorms_struct.append(b)
        out = {"meangradnorms": meangradnorms_struct}
        out = ujson.dumps(out)
        return out

    @classmethod
    def reload(cls, serializedresults:str):
        d = ujson.loads(serializedresults)
        d["meangradnorms"] = np.asarray(d["meangradnorms"])
        return d


def run_training(run_train_epoch=None,
                 run_valid_epoch=None,
                 run_grad_epoch=None,
                 max_epochs=1, gradanalinterval=-1, validinter=1,
                 print_on_valid_only=False, check_stop=tuple()):
    """

    :param run_train_epoch:     function that performs an epoch of training. must accept current_epoch and max_epochs. Tip: use functools.partial
    :param run_valid_epoch:     function that performs an epoch of testing. must accept current_epoch and max_epochs. Tip: use functools.partial
    :param max_epochs:
    :param validinter:
    :param print_on_valid_only:
    :return:
    """
    tt = q.ticktock(":-")
    validinter_count = 0
    current_epoch = 0
    stop_training = current_epoch >= max_epochs
    while stop_training is not True:
        tt.tick()
        if current_epoch % gradanalinterval == 0:
            run_grad_epoch(current_epoch=current_epoch, max_epochs=max_epochs)
        ttmsg = run_train_epoch(current_epoch=current_epoch, max_epochs=max_epochs)
        ttmsg = "Ep. {}/{} -- {}".format(
            f"{{:>{len(str(max_epochs))}}}".format(current_epoch + 1),
            max_epochs,
            ttmsg)
        validepoch = False
        if run_valid_epoch is not None and validinter_count % validinter == 0:
            ttmsg_v = run_valid_epoch(current_epoch=current_epoch, max_epochs=max_epochs)
            ttmsg += " -- " + ttmsg_v
            validepoch = True
        validinter_count += 1
        if not print_on_valid_only or validepoch:
            tt.tock(ttmsg)
        current_epoch += 1
        stop_training = any([e() for e in check_stop])
        stop_training = stop_training or (current_epoch >= max_epochs)


def run(lr=0.001,  # 0.001
        dropout=0.3,
        dropoutemb=-1.,
        zoneout=0.0,
        embdim=20,  # 20 (changed to 50 default)
        hdim=100,  # 50
        numheads=2,
        epochs=100,  # 100
        seqlen=10,  # 10
        numlayers=-1,
        npery=20,
        within=-1,
        batsize=20,
        cuda=False,
        gpu=0,
        cell="sggnn",  # "rgcn", "ggnn", "sggnn", "rgat", "rvgat"
        innercell="dgru",  # "gru" or "dgru" or "sdgru" or "dsdgru" or "sum" or "sumrelu", "linrelu"
        seed=123456,
        noattention=False,
        usevallin=False,
        only_update=False,
        relmode="gatedcatpremap",
        gradnorm=3,
        smoothing=0.1,
        cosinelr=False,
        warmup=0,
        gradanalinterval=10,
        patience=-1,
        no_skip=False,
        skipatt=False,
        nodamper=False,
        shared=False,
        usegate=False,
        usesgru=False,
        noresmsg=False,
        version="cr",
        ):
    # if noattention or nodgru or nonovallin:
    #     dropout = ablationdropout
    # embdim = hdim
    if dropoutemb < 0:
        dropoutemb = dropout
    settings = locals().copy()
    q.pp_dict(settings)

    wandb.init(project="sggnn_recall_small_baselines", config=settings, reinit=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if cuda is False:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda", gpu)


    ds = ConditionalRecallDataset(maxlen=seqlen, NperY=npery, within=within)

    numsteps = seqlen + 1 if numlayers < 0 else numlayers
    _cell = cell
    if cell == "rgcn":
        cell = RGCN(hdim, numsteps, numrels=5, dropout=dropout, shared=False)
    if cell == "sharedrgcn":
        cell = RGCN(hdim, numsteps, numrels=5, dropout=dropout, shared=True)
    elif cell == "deepgcn":
        cell = DeepRGCN(hdim, numlayers=numsteps, numrels=5, dropout=dropout, residual=not no_skip)
    if cell == "ggnn":
        cell = GGNN(hdim, numlayers=1, numrepsperlayer=numsteps, numrels=5, dropout=dropout, use_dgru=False)
    # elif cell == "sggnn":
    #     # numsteps += 1
    #     cell = SGGNNCell(hdim, numrels=5, dropout=dropout, zoneout=zoneout, numheads=numheads, cell=innercell,
    #                      no_attention=noattention, usevallin=usevallin, relmode=relmode)
    elif cell == "graphlstm":
        cell = RelGraphLSTM(hdim, dropout=dropout, numrepsperlayer=numsteps, numrels=5, relmode=relmode, rdim=hdim)
    elif cell == "grgat":
        cell = GRGAT(hdim, numlayers=1, numrepsperlayer=numsteps, numrels=5, numheads=numheads, dropout=dropout, relmode=relmode,
                            rdim=hdim, cell=innercell, norel=only_update, noattention=only_update,
                     skipatt=skipatt, nodamper=nodamper, usevallin=usevallin)
    # elif cell == "rvgat":
    #     cell = RVGAT(hdim, dropout=dropout, numheads=numheads, cell=innercell, usevallin=usevallin, relmode=relmode)
    elif cell == "resrgat":
        if shared:
            numlayers = 1
            numrepsperlayer = numsteps
        else:
            numlayers = numsteps
            numrepsperlayer = 1
        cell = ResRGAT(hdim, numlayers=numlayers, numrepsperlayer=numrepsperlayer,
                       numrels=5, numheads=numheads, dropout=dropout, dropout_act=0.,
                       rdim=hdim, usevallin=usevallin, no_resmsg=noresmsg,
                       skipatt=skipatt, use_gate=usegate, use_sgru=usesgru,
                       cat_rel=True)
    elif cell == "lrgat":
        if shared:
            numlayers = 1
            numrepsperlayer = numsteps
        else:
            numlayers = numsteps
            numrepsperlayer = 1
        cell = LRGAT(hdim, numlayers=numlayers, numrepsperlayer=numrepsperlayer,
                       numrels=5, numheads=numheads, dropout=dropout, usevallin=usevallin)
    elif cell == "reltransformer":
        _numlayers = _numrepsperlayer = 1
        if shared:
            _numrepsperlayer = numsteps
        else:
            _numlayers = numsteps
        cell = RelTransformer(hdim, numlayers=_numlayers, numrepsperlayer=_numrepsperlayer,
                              numrels=5, numheads=numheads, dropout=dropout, rdim=hdim, relmode=relmode)
    elif cell == "rmgat":
        cell = RMGAT(hdim, numrels=5, numsteps=numsteps, dropout=dropout, numheads=numheads, act=innercell)
    elif cell == "rgat":
        dropout = 0.1
        # dropout = 0.
        # hdim = 200
        # cells = [RGATCell(hdim, hdim, num_heads=numheads, feat_drop=0., attn_drop=0., residual=True) for _ in range(numsteps)]
        cell = RGATCell(hdim, hdim, num_heads=numheads, feat_drop=0, attn_drop=0., residual=True)
        cells = [cell] * numsteps
        cell = RGATCellStack(cells, feat_drop=dropout)
        # cell = RGATCell(hdim, hdim, num_heads=numheads, feat_drop=dropout, attn_drop=dropout, residual=True)
    elif cell == "gatedgcn":
        netparams = {
            "num_atom_type": ds.encoder.vocab.number_of_ids(),
            "num_bond_type": 5,
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


    m = SeqGGNN(ds.encoder.vocab, embdim, cell, emb_dropout=dropoutemb, dropout=dropout, numsteps=numsteps,
                maxlen=seqlen+2, add_self_edge=_cell in {"resrgat", "lrgat", "grgat", "rgcn", "sharedrgcn", "rmgat"} and not only_update)
    print(m)
    numparams = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Total number of parameters: {numparams}")

    # dl = ds.dataloader("train", batsize=batsize, shuffle=True)
    # batch = iter(dl).next()
    # print(batch)
    # y = m(batch[0])
    # print(y.size())
    if smoothing > 0:
        loss = q.SmoothedCELoss(smoothing=smoothing)
    else:
        loss = torch.nn.CrossEntropyLoss(reduction="mean")
    acc = ClassificationAccuracy()

    optim = torch.optim.Adam(m.parameters(), lr=lr)
    losses = [q.MetricWrapper(loss, "CE"), q.MetricWrapper(acc, "acc")]
    vlosses = [q.MetricWrapper(loss, "CE"), q.MetricWrapper(acc, "acc")]

    clipgradnorm = lambda: torch.nn.utils.clip_grad_norm_(m.parameters(), gradnorm)
    if cosinelr:
        # lrsched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)
        lrsched = q.sched.Linear(steps=warmup) >> q.sched.Cosine(steps=epochs-warmup) >> 0
    else:
        lrsched = q.sched.Linear(steps=warmup) >> 1
    lrsched = lrsched.get_lr_sched(optim)

    # gradanalyzer = GradAnalyzer(m)

    # clipgradnorm = lambda: None
    # run gradient analysis before training
    _optim = torch.optim.Adam(m.parameters(), lr=0)
    _losses = [q.MetricWrapper(loss, "CE"), q.MetricWrapper(acc, "acc")]
    # _gradbatch = partial(q.train_batch,
    #                      on_start_args=[gradanalyzer.on_batch_start_args],
    #                      on_before_optim_step=[clipgradnorm, gradanalyzer.on_before_optim_step])
    # _on_end = [gradanalyzer.on_epoch_end]
    _gradbatch = partial(q.train_batch,
                         on_start_args=[],
                         on_before_optim_step=[clipgradnorm])
    _on_end = []
    _gradepoch = partial(q.train_epoch, model=m, dataloader=ds.dataloader("valid", batsize=batsize, shuffle=False),
                         optim=_optim, losses=_losses, device=device, _train_batch=_gradbatch, on_end=_on_end)
    # q.run_training(_gradepoch, max_epochs=epochs)

    trainbatch = partial(q.train_batch,
                         on_before_optim_step=[clipgradnorm])
    eyt = q.EarlyStopper(vlosses[1], patience=patience, more_is_better=True, min_epochs=20, remember_f=lambda: deepcopy(m))

    def wandb_logger():
        d = {}
        for name, loss in zip(["CE", "acc"], losses):
            d["train_"+name] = loss.get_epoch_error()
        for name, loss in zip(["CE", "acc"], vlosses):
            d["valid_"+name] = loss.get_epoch_error()
        wandb.log(d)

    on_end = [lambda: lrsched.step()]
    on_end_v = [lambda: wandb_logger(), lambda: eyt.on_epoch_end()]
    checkstop = [lambda: eyt.check_stop()]
    trainepoch = partial(q.train_epoch, model=m, dataloader=ds.dataloader("train", batsize=batsize, shuffle=True),
                         optim=optim, losses=losses, device=device, _train_batch=trainbatch, on_end=on_end)
    validepoch = partial(q.test_epoch, model=m, dataloader=ds.dataloader("valid", batsize=batsize, shuffle=False),
                         losses=vlosses, device=device, on_end=on_end_v)

    run_training(trainepoch, validepoch, _gradepoch,
                 max_epochs=epochs, gradanalinterval=gradanalinterval,
                 check_stop=checkstop)

    if eyt.remembered is not None and patience >= 0:
        print("reloading best")
        m = eyt.remembered

    tlosses = [q.MetricWrapper(loss, "CE"), q.MetricWrapper(acc, "acc")]
    q.test_epoch(model=m, dataloader=ds.dataloader("test", batsize=batsize, shuffle=False),
                         losses=tlosses, device=device)

    settings.update({"train_acc": losses[1].get_epoch_error()})
    settings.update({"valid_acc": vlosses[1].get_epoch_error()})
    settings.update({"test_acc": tlosses[1].get_epoch_error()})
    wandb.config.update(settings)
    print(ujson.dumps(settings, indent=4))

    # # finalize gradient analysis and store results in output
    # # print(len(pickle.dumps(gradanalyzer.finalize())))
    # settings["gradient_analysis"] = gradanalyzer.finalize()


    # run analysis
    eval_inps, eval_outs = q.eval_loop(m, dataloader=ds.dataloader("valid", batsize, shuffle=False), device=device, lastisgold=True)
    inp_seqs = torch.cat(eval_inps[0], 0)
    golds = torch.cat(eval_inps[1], 0)
    preds = torch.cat(eval_outs[0], 0)

    rand = "".join(random.choice(string.ascii_letters) for i in range(6))
    with open(f"sggnn_recall_grad_s{seqlen}_{_cell}.{rand}.preds", "wb") as f:
        pickle.dump({"settings": settings,
                    "inps": inp_seqs.cpu().numpy(),
                    "golds": golds.cpu().numpy(),
                    "preds": preds.cpu().numpy()}, f)

    return settings


# region GRU baseline
class RNNModel(torch.nn.Module):
    def __init__(self, vocab, embdim, hdim, dropout=0., **kw):
        super(RNNModel, self).__init__(**kw)
        self.vocab = vocab
        self.embdim, self.hdim = embdim, hdim

        self.emb = torch.nn.Embedding(vocab.number_of_ids(), self.embdim)
        maxlen = 100
        self.emb_grad_var = torch.nn.Parameter(torch.zeros(100, maxlen, self.embdim))
        self.init_state_grad_var = torch.nn.Parameter(torch.zeros(100, self.hdim))
        self.outlin = torch.nn.Linear(self.hdim, vocab.number_of_ids())

        self.gru = GRUEncoder(self.embdim, self.hdim, dropout=dropout, bidirectional=False)

        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")
        self.acc = ClassificationAccuracy()

    def forward(self, x, gold):

        embs = self.emb(x)
        embs = embs.detach() + self.emb_grad_var[:embs.size(0), :embs.size(1), :]
        initstate = self.init_state_grad_var[:x.size(0), :]
        # initstate = torch.zeros(x.size(0), self.hdim, device=x.device)
        encs, finalenc = self.gru(embs, initstate[None])
        final = finalenc[-1][0]
        out = self.outlin(final)

        if self.training:
            final.sum().backward(create_graph=True)
            stategrad = self.init_state_grad_var.grad[:x.size(0), :]
            inpgrad = self.emb_grad_var.grad[:x.size(0), :x.size(1), :]
            gradloss = -inpgrad[:, -2:-1].norm(2, -1).mean()
            self.init_state_grad_var.grad = None
            self.emb_grad_var.grad = None
            self.zero_grad()


        embs = self.emb(x)
        initstate = torch.zeros(x.size(0), self.hdim, device=x.device)
        encs, finalenc = self.gru(embs, initstate[None])
        final = finalenc[-1][0]
        out = self.outlin(final)

        loss = self.loss(out, gold)
        acc = self.acc(out, gold)

        if self.training:
            return loss, loss, gradloss, acc
        else:
            return loss, acc

    def parameters(self):
        for name, param in self.named_parameters():
            if name not in ("emb_grad_var", "init_state_grad_var"):
                yield param


def run_gru(lr=0.001,
        embdim=20,
        hdim=100,
        epochs=100,
        seqlen=15,
        batsize=20,
        cuda=False,
        dropout=.2,
        gpu=0,
        npery=20,
        ):
    if cuda is False:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda", gpu)
    ds = ConditionalRecallDataset(maxlen=seqlen, NperY=npery)
    print(f"{ds.N} examples in total.")

    m = RNNModel(ds.encoder.vocab, embdim, hdim, dropout=dropout)

    # dl = ds.dataloader("train", batsize=batsize, shuffle=True)
    # batch = iter(dl).next()
    # print(batch)
    # y = m(batch[0])
    # print(y.size())

    optim = torch.optim.Adam(m.parameters(), lr=lr)
    losses = [q.LossWrapper(q.SelectedLinearLoss(0), "loss"), q.LossWrapper(q.SelectedLinearLoss(1), "CE"),
              q.LossWrapper(q.SelectedLinearLoss(2), "gradloss"), q.LossWrapper(q.SelectedLinearLoss(3), "acc")]
    vlosses = [q.LossWrapper(q.SelectedLinearLoss(0), "CE"), q.LossWrapper(q.SelectedLinearLoss(1), "acc")]

    trainepoch = partial(q.train_epoch, model=m, dataloader=ds.dataloader("train", batsize=batsize, shuffle=True), optim=optim, losses=losses, device=device)
    validepoch = partial(q.test_epoch, model=m, dataloader=ds.dataloader("valid", batsize=batsize, shuffle=False), losses=vlosses, device=device)

    q.run_training(trainepoch, validepoch, max_epochs=epochs)

# endregion



def run_experiment(npery=20,
                   within=-1,
                   cell="grgat",
                   innercell="default",
                   relmode="default",
                   usevallin=False,
                   noattention=False,
                   numheads=-1,
                   seqlen=-1,
                   numlayers=-1,
                   cuda=False,
                   gpu=0,
                   seed=-1,
                   epochs=-1,
                   dropout=-1.,
                   dropoutemb=-1.,
                   hdim=-1,
                   embdim=20,
                   gradanalinterval=5,
                   patience=-1,
                   lr=-1.,
                   only_update=False,
                   no_skip=False,
                   skipatt=False,
                   nodamper=False,
                   shared=False,
                   usegate=False,
                   usesgru=False,
                   version="cr",
                   noresmsg=False,
                   ):

    settings = locals().copy()

    ranges = {
        # "dropout": [0., .25, .5],
        "dropout": [0., 0.25],
        # "dropoutemb": [0.1],
        # "epochs": [120],
        "epochs": [200],
        "batsize": [20],
        # "hdim": [150, 300],
        # "hdim": [20, 40, 80],
        "hdim": [100, 200],
        "numheads": [2],
        # "lr": [0.0001, 0.001, 0.0005],
        "lr": [0.0005],
        # "seed": [349899875, 768504099, 87646464],
        # "seed": [87646464],
        "seed": [42, 87646464, 456852],
        "seqlen": [5, 10, 15],
    }
    if cell == "gatedgcn":
        ranges["dropout"] = [0., 0.1, 0.2, 0.4]
        ranges["epochs"] = [200]
        ranges["hdim"] = [50, 150]
        ranges["lr"] = [0.001, 0.005, 0.0005, 0.0001]

    if cell == "sggnn":
        ranges["innercell"] = ["dgru"]
        ranges["epochs"] = [50]
    elif cell == "grgat" or cell == "resrgat" or cell == "lrgat":
        ranges["innercell"] = ["none"]
        ranges["epochs"] = [50]
    elif cell == "rgcn" or cell == "sharedrgcn":
        ranges["innercell"] = ["default"]
    elif cell == "ggnn":
        ranges["innercell"] = ["gru"]
    elif cell == "rmgat":
        ranges["innercell"] = ["none", "relu"]
    elif cell == "reltransformer":
        ranges["shared"] = [True, False]
        # ranges["dropoutemb"] = [0., 0.25, 0.5]
        ranges["dropout"] = [0., 0.25, 0.5]
        ranges["lr"] = [0.0001]

    for k in ranges:
        if k in settings:
            if isinstance(settings[k], str) and settings[k] != "default":
                ranges[k] = [settings[k]]
            elif isinstance(settings[k], (int, float)) and settings[k] >= 0:
                ranges[k] = [settings[k]]
            else:
                pass
                # raise Exception(f"something wrong with setting '{k}'")
            del settings[k]

    def checkconfig(spec):
        return True

    print(__file__)
    celln = cell
    if cell == "sggnn" or cell == "grgat":
        celln = cell + "-" + innercell
    p = __file__ + f".{celln}.x{npery}.s{seqlen}"
    q.run_experiments_random(
        run, ranges, path_prefix=p, check_config=checkconfig, **settings)


if __name__ == '__main__':
    # try_multihead_self_attention()
    # q.argprun(run)
    q.argprun(run_experiment)
    # q.argprun(run_gru)

# for npery=50: -cuda -gpu 0 -seqlen 15 -cell rgcn/ggnn -lr 0.0005 -npery 50 -epochs 200 -patience 200