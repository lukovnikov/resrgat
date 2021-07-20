import collections
import math
import random
from functools import partial

import qelos as q
import torch
import numpy as np
import re
import dgl
import ujson
from dgl.nn.pytorch import edge_softmax
from dgl.nn.pytorch.utils import Identity
from torch.nn import init

from torch.utils.data import DataLoader

from ldgnn.cells import SGGNNCell, GGNNCell, RGATCell, RVGAT, RMGATCell, RMGAT
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


def gen_data(maxlen=10, NperY=2):
    # random.seed(74850303)
    letters = [chr(x) for x in range(97, 123)]
    numbers = [chr(x) for x in range(48, 58)]
    uppercase = [chr(x) for x in range(65, 90)]
    # print(letters)
    # print(numbers)
    # print(uppercase)

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
            position = random.choice(list(range(maxlen-1)))    # position at which y occurs
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
    def __init__(self, maxlen=10, NperY=10, **kw):
        super(ConditionalRecallDataset, self).__init__(**kw)
        self.data = {}
        self.NperY, self.maxlen = NperY, maxlen
        self._seqs, self._ys = gen_data(self.maxlen, self.NperY)
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
                 use_self_loop=False, skip=False, shared=False):
        super(RGCN, self).__init__()
        self.hdim = hdim
        self.rdim = self.hdim if rdim is None else rdim
        self.num_rels = numrels
        self.num_bases = num_bases
        self.numlayers = numlayers
        self.use_self_loop = use_self_loop

        # create rgcn layers
        self.layers = torch.nn.ModuleList()
        layer = RelGraphConv(self.hdim, self.hdim, self.num_rels, "simple",
                         self.num_bases, activation=torch.relu, self_loop=True,
                         dropout=dropout, rdim=self.rdim, aggr="mean")
        for idx in range(self.numlayers):
            if not shared:
                layer = RelGraphConv(self.hdim, self.hdim, self.num_rels, "simple",
                             self.num_bases, activation=torch.relu, self_loop=True,
                             dropout=dropout, rdim=self.rdim, aggr="mean")
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

    def forward(self, g, step=0):
        _h = g.ndata["h"]
        r = g.edata["id"]
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
    def __init__(self, vocab, embdim, cell, numsteps=10, maxlen=10, emb_dropout=0., add_self_edge=True, **kw):
        super(SeqGGNN, self).__init__(**kw)
        self.vocab = vocab
        self.cell = cell
        self.hdim = self.cell.hdim
        self.numsteps = numsteps
        self.embdim = embdim
        self.maxlen = maxlen
        self.add_self_edge = add_self_edge

        self.emb = torch.nn.Embedding(vocab.number_of_ids(), self.embdim)
        self.emb_adapter = torch.nn.Linear(embdim, self.hdim, bias=False) if embdim != self.hdim else Identity()
        self.posemb = torch.nn.Embedding(maxlen, self.embdim)
        outlindim = self.hdim
        self.outlin = torch.nn.Linear(outlindim, vocab.number_of_ids())
        self.emb_dropout = torch.nn.Dropout(emb_dropout)

    def forward(self, x):
        # region create graph
        g = dgl.DGLGraph()
        g.add_nodes(x.size(0) * x.size(1))
        embs = self.emb(x)
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
        self.cell.init_node_states(g, _embs.size(0), device=_embs.device)

        xlen = x.size(1)
        for i in range(x.size(0)):
            if self.add_self_edge:  # self edges
                for j in range(xlen):
                    g.add_edge(i * xlen + j, i * xlen + j, {"id": torch.tensor([3], device=x.device)})
            for j in range(xlen-1):
                g.add_edge(i * xlen + j, i * xlen + j + 1, {"id": torch.tensor([1], device=x.device)})
                g.add_edge(i * xlen + j + 1, i * xlen + j, {"id": torch.tensor([2], device=x.device)})

        # endregion

        # run updates
        self.cell.reset_dropout()
        for step in range(self.numsteps):
            self.cell(g, step=step)

        # region extract predictions
        if False: #isinstance(self.cell, DSSGGNNCell):
            out = torch.cat([g.ndata["h"], g.ndata["c"]], -1)
        else:
            out = g.ndata["h"]
        out = out.view(embs.size(0), embs.size(1), -1)

        lastout = out[:, -1, :]
        pred = self.outlin(lastout)
        # endregion
        return pred


class ClassificationAccuracy(torch.nn.Module):
    def forward(self, probs, target):
        _, pred = probs.max(-1)
        same = pred == target
        ret = same.float().sum() / same.size(0)
        return ret


def run(lr=0.001,  # 0.001
        dropout=0.3,
        dropoutemb=0.0,
        zoneout=0.0,
        embdim=20,  # 20 (changed to 50 default)
        hdim=100,  # 50
        numheads=2,
        epochs=100,  # 100
        seqlen=10,  # 10
        npery=20,
        batsize=20,
        cuda=False,
        gpu=0,
        cell="sggnn",  # "rgcn", "ggnn", "sggnn", "rgat", "rvgat"
        innercell="dgru",  # "gru" or "dgru" or "sdgru" or "dsdgru" or "sum" or "sumrelu", "linrelu"
        seed=123456,
        noattention=False,
        usevallin=False,
        relmode="gatedcatpremap",
        gradnorm=3,
        smoothing=0.1,
        cosinelr=False,
        warmup=0,
        ):
    # if noattention or nodgru or nonovallin:
    #     dropout = ablationdropout
    # embdim = hdim
    settings = locals().copy()
    print(ujson.dumps(settings, indent=4))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if cuda is False:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda", gpu)

    numsteps = seqlen + 1
    if cell == "rgcn":
        cell = RGCN(hdim, numsteps, dropout=dropout, shared=False)
    if cell == "sharedrgcn":
        cell = RGCN(hdim, numsteps, dropout=dropout, shared=True)
    if cell == "ggnn":
        cell = GGNNCell(hdim, dropout=dropout)
    elif cell == "sggnn":
        # numsteps += 1
        cell = SGGNNCell(hdim, dropout=dropout, zoneout=zoneout, numheads=numheads, cell=innercell,
                         no_attention=noattention, usevallin=usevallin, relmode=relmode)
    elif cell == "rvgat":
        cell = RVGAT(hdim, dropout=dropout, numheads=numheads, cell=innercell, usevallin=usevallin, relmode=relmode)
    elif cell == "rmgat":
        cell = RMGAT(hdim, numsteps=numsteps, dropout=dropout, numheads=numheads, act=innercell)
    elif cell == "rgat":
        dropout = 0.1
        # dropout = 0.
        # hdim = 200
        # cells = [RGATCell(hdim, hdim, num_heads=numheads, feat_drop=0., attn_drop=0., residual=True) for _ in range(numsteps)]
        cell = RGATCell(hdim, hdim, num_heads=numheads, feat_drop=0, attn_drop=0., residual=True)
        cells = [cell] * numsteps
        cell = RGATCellStack(cells, feat_drop=dropout)
        # cell = RGATCell(hdim, hdim, num_heads=numheads, feat_drop=dropout, attn_drop=dropout, residual=True)

    ds = ConditionalRecallDataset(maxlen=seqlen, NperY=npery)

    m = SeqGGNN(ds.encoder.vocab, embdim, cell, emb_dropout=dropoutemb, numsteps=numsteps, maxlen=seqlen+2)
    print(m)

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
    losses = [q.LossWrapper(loss, "CE"), q.LossWrapper(acc, "acc")]
    vlosses = [q.LossWrapper(loss, "CE"), q.LossWrapper(acc, "acc")]

    clipgradnorm = lambda: torch.nn.utils.clip_grad_norm_(m.parameters(), gradnorm)
    if cosinelr:
        # lrsched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)
        lrsched = q.sched.Linear(steps=warmup) >> q.sched.Cosine(steps=epochs-warmup) >> 0
    else:
        lrsched = q.sched.Linear(steps=warmup) >> 1
    lrsched = lrsched.get_lr_sched(optim)
    # clipgradnorm = lambda: None
    trainbatch = partial(q.train_batch, on_before_optim_step=[clipgradnorm])
    on_end = [lambda: lrsched.step()]
    trainepoch = partial(q.train_epoch, model=m, dataloader=ds.dataloader("train", batsize=batsize, shuffle=True),
                         optim=optim, losses=losses, device=device, _train_batch=trainbatch, on_end=on_end)
    validepoch = partial(q.test_epoch, model=m, dataloader=ds.dataloader("valid", batsize=batsize, shuffle=False),
                         losses=vlosses, device=device)

    q.run_training(trainepoch, validepoch, max_epochs=epochs)

    tlosses = [q.LossWrapper(loss, "CE"), q.LossWrapper(acc, "acc")]
    q.test_epoch(model=m, dataloader=ds.dataloader("test", batsize=batsize, shuffle=False),
                         losses=tlosses, device=device)

    settings.update({"train_acc": losses[1].get_epoch_error()})
    settings.update({"valid_acc": vlosses[1].get_epoch_error()})
    settings.update({"test_acc": tlosses[1].get_epoch_error()})
    print(ujson.dumps(settings, indent=4))
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
                   cell="sggnn",
                   innercell="dgru",
                   usevallin=False,
                   noattention=False,
                   seqlen=5,
                   cuda=False,
                   gpu=0,
                   seed=123456,
                   ):
    if innercell == "default":
        if cell == "sggnn":
            innercell = "dgru"
        elif cell == "rmgat":
            innercell = "none"
    if cell == "sggnn" and innercell == "dgru" and usevallin is False and noattention is False:
        if seqlen > 0 and seqlen < 7:
            ranges = {
                "dropout": [0, .15, .3],
                "epochs": [50, 100],
                "hdim": [50, 100]
            }
        elif seqlen >= 7 and seqlen < 15:
            ranges = {
                "dropout": [.15, .3],
                "epochs": [100, 150],
                "hdim": [100, 200]
            }
        elif seqlen >= 15:
            ranges = {
                "dropout": [.15, .3],
                "epochs": [100, 150],
                "hdim": [200, 420]
            }
    elif cell == "rgcn" or cell == "sharedrgcn":
        if seqlen < 6:
            ranges = {
                "dropout": [.15, .3, .4],
                "epochs": [120, 200],
                "hdim": [80, 100, 150, 200]
            }
        else:
            ranges = {
                "dropout": [.15, .3, .4],
                "epochs": [120, 200],
                "hdim": [100, 200, 300]
            }
    elif cell == "ggnn":
        if seqlen < 4:
            ranges = {
                "dropout": [.25, .45],
                "epochs": [70, 120],
                "hdim": [80, 120]
            }
        elif seqlen < 8:
            ranges = {
                "dropout": [.3, .45],
                "epochs": [100, 160],
                "hdim": [100, 160]
            }
        else:
            ranges = {
                "dropout": [.15, .3, .45],
                "epochs": [100, 160],
                "hdim": [80, 120, 200]
            }
    elif cell == "rmgat":
        if seqlen < 6:
            ranges = {
                "dropout": [.25, .45],
                "epochs": [50, 100],
                "hdim": [80, 120]
            }
        elif seqlen >= 6:
            ranges = {
                "dropout": [.1, .2, .3, .5],
                "epochs": [100, 140],
                "hdim": [100, 102, 140, 142, 200, 198]
            }
        dropoutemb = .1
    if npery < 50:
        batsize = 20
    else:
        batsize = 100
    print(__file__)
    celln = cell
    if cell == "sggnn":
        celln = cell + "-" + innercell
    p = __file__ + f".{celln}.x{npery}.s{seqlen}.xps"
    q.run_experiments(run, ranges, path=p, pmtf=lambda x: x["test_acc"] > .9,
                      npery=npery, cell=cell, innercell=innercell, seqlen=seqlen,
                      cuda=cuda, gpu=gpu, batsize=batsize, seed=seed, dropoutemb=dropoutemb,
                      usevallin=usevallin, noattention=noattention)


def run_experiments(npery=20,
                    cell="rmgat",
                    innercell="default",
                    usevallin=False,
                    noattention=False,
                    cuda=False,
                    gpu=0,
                    ):
    seqlens = [7, 10]
    if cell == "sggnn" and innercell == "dgru" and usevallin is False and noattention is False:
        if npery < 50:
            seqlens = [3, 5, 7, 10]
        else:
            seqlens = [20, 30, 40]
    for seqlen in seqlens:
        run_experiment(npery=npery, seqlen=seqlen, cell=cell, innercell=innercell, usevallin=usevallin, noattention=noattention,
                       cuda=cuda, gpu=gpu)



if __name__ == '__main__':
    # try_multihead_self_attention()
    q.argprun(run)
    # q.argprun(run_experiments)
    # q.argprun(run_gru)