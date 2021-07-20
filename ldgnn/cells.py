import math
import numpy as np

import dgl
import torch
from dgl.nn.pytorch import edge_softmax
from dgl.nn.pytorch.utils import Identity
from torch.nn import init
from torch.nn.init import calculate_gain

from ldgnn.nn import DSDGRUCell, SimpleDGRUCell, AttentionDropout, WindowHeadAttention, MultiHeadAttention_Isolated, \
    DGRUCell, SumAct, LinearSkip, GRUCell, NIGRUCell, MultiHeadAttention, forward_mix_backward_sum

# RELMATINITMULT = 0.2
RELMATINITMULT = 0.75
# RELMATINITMULT = 1.


def xavier_uniform_for_relmats_(relmat, gain=1.):
    # return torch.nn.init.xavier_uniform_(relmat, gain=gain)
    fanin, fanout = relmat.size(1), relmat.size(2)
    std = gain \
          * math.sqrt(2.0 / float(fanin + fanout))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return torch.nn.init._no_grad_uniform_(relmat, -a, a)


class RGATCell(torch.nn.Module):
    r"""Apply `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`__
    over an input signal.

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}

    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:

    .. math::
        \alpha_{ij}^{l} & = \mathrm{softmax_i} (e_{ij}^{l})

        e_{ij}^{l} & = \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    num_heads : int
        Number of heads in Multi-Head Attention.
    feat_drop : float, optional
        Dropout rate on feature, defaults: ``0``.
    attn_drop : float, optional
        Dropout rate on attention weight, defaults: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope.
    residual : bool, optional
        If True, use residual connection.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    """
    def __init__(self,
                 in_feats=None,
                 out_feats=None,
                 num_heads=2,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=torch.nn.functional.elu,
                 numrels=1,
                 mode="mult",       # "mult" or "add"
                 ):
        super(RGATCell, self).__init__()
        raise Exception("probably want to use RMGAT for comparison purposes")
        self.numheads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.mode = mode
        self.hdim = out_feats

        self.relvectors = torch.nn.Parameter(torch.randn(numrels, self._in_feats))
        init.kaiming_uniform_(self.relvectors, a=math.sqrt(5))

        self.fc = torch.nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.kfc = torch.nn.Linear(in_feats*2, out_feats * num_heads, bias=False)
        self.attn_l = torch.nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = torch.nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = torch.nn.Dropout(feat_drop)
        self.register_buffer("dropout_mask", torch.ones(self.hdim))
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope)
        if residual:
            if in_feats != out_feats:
                self.res_fc = torch.nn.Linear(in_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.xavier_normal_(self.fc.weight, gain=gain)
        torch.nn.init.xavier_normal_(self.kfc.weight, gain=gain)
        torch.nn.init.xavier_normal_(self.attn_l, gain=gain)
        torch.nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, torch.nn.Linear):
            torch.nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def message_func(self, edges):
        relvecs = self.relvectors[edges.data["id"]]
        hs = edges.src["h"]
        hs = self.dropout_mask[None, :] * self.feat_drop(hs)
        msg = torch.cat([hs, relvecs], -1)
        msg = self.kfc(msg)
        hs = self.fc(hs)
        return {"msg": msg, "hs": hs}

    def reduce_func(self, nodes):
        queries = self.fc(self.feat_drop(nodes.data["h"]) * self.dropout_mask[None, :])
        context = nodes.mailbox["msg"]  # bsd - !! fc() has already been done in message !!
        queries = queries.view(queries.size(0), self.numheads, -1)  # bhl
        context = context.view(context.size(0), context.size(1), self.numheads, -1).transpose(1, 2)  # bhsl
        if self.mode == "mult":
            weights = torch.einsum("bhd,bhsd->bhs", queries, context) / np.sqrt(context.size(-1))
        elif self.mode == "add":
            pass
        weights = self.leaky_relu(weights)
        alphas = torch.softmax(weights, -1)  # bhs
        alphas = self.attn_drop(alphas)

        values = nodes.mailbox["hs"]
        values = values.view(values.size(0), values.size(1), self.numheads, -1).transpose(1, 2)
        red = torch.einsum("bhs,bhsd->bhd", alphas, values)
        red = red.sum(1)
        return {"red": red}

    def apply_node_func(self, nodes):
        h = nodes.data["h"]
        h = self.feat_drop(h) * self.dropout_mask[None, :]
        rst = nodes.data["red"]
        if self.res_fc is not None:
            resval = self.res_fc(h)
            rst = rst + resval
        if self.activation:
            rst = self.activation(rst)
        return {"h": rst}

    def forward(self, g, step=0):
        g.update_all(self.message_func, self.reduce_func, self.apply_node_func)

    def _forward(self, g, feat):
        r"""Compute graph attention network layer.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        """
        g = g.local_var()
        h = self.feat_drop(feat)
        feat = self.fc(h).view(-1, self.numheads, self._out_feats)
        el = (feat * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat * self.attn_r).sum(dim=-1).unsqueeze(-1)
        g.ndata.update({'ft': feat, 'el': el, 'er': er})
        # compute edge attention
        g.apply_edges(dgl.function.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(g.edata.pop('e'))
        # compute softmax
        g.edata['a'] = self.attn_drop(edge_softmax(g, e))
        # message passing
        g.update_all(dgl.function.u_mul_e('ft', 'a', 'm'),
                     dgl.function.sum('m', 'ft'))
        rst = g.ndata['ft']
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h).view(h.shape[0], -1, self._out_feats)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst


str2act = {
    "none": Identity(),
    "relu": torch.nn.ReLU(),
    "lrelu": torch.nn.LeakyReLU(.1),
    "celu": torch.nn.CELU(),
    "tanh": torch.nn.Tanh(),
    "sigm": torch.nn.Sigmoid(),
}


class RMGAT(torch.nn.Module):
    def __init__(self, hdim, numsteps=1, dropout=0., numrels=1, numheads=2, act="none", rdim=None, useshared=False, **kw):
        super(RMGAT, self).__init__(**kw)
        if useshared:
            numsteps = 1
        self.layers = torch.nn.ModuleList([
            RMGATCell(hdim, dropout=dropout, numrels=numrels,
                      numheads=numheads, act=act, rdim=rdim, **kw)
            for _ in range(numsteps)])
        self.hdim = hdim

    def init_node_states(self, g, batsize, device):
        # for layer in self.layers:
        self.layers[0].init_node_states(g, batsize, device)

    def reset_dropout(self):
        for layer in self.layers:
            layer.reset_dropout()

    def forward(self, g, step=None):
        if step is None:
            ret = g
            for layer in self.layers:
                ret = layer(ret)
        else:
            step = min(step, len(self.layers) - 1)
            layer = self.layers[step]
            ret = layer(g, step=step)
        return ret


class RMGATCell(torch.nn.Module):
    useGasV = True
    """ Babylon Health style R-GAT"""
    def __init__(self, hdim, dropout=0., numrels=1, numheads=2, act="none", rdim=None, **kw):
        super(RMGATCell, self).__init__(**kw)
        self.hdim, self.dropout, self.numrels = hdim, dropout, numrels
        self.rdim = self.hdim if rdim is None else rdim
        self.numheads = numheads
        self.act = str2act[act]
        self.relmats_G = torch.nn.Parameter(torch.zeros(self.numrels, self.hdim, self.hdim))
        self.relmats_Q = torch.nn.Parameter(torch.zeros(self.numrels, self.hdim, self.hdim))
        self.relmats_K = torch.nn.Parameter(torch.zeros(self.numrels, self.hdim, self.hdim))
        if not self.useGasV:
            self.relmats_V = torch.nn.Parameter(torch.zeros(self.numrels, self.hdim, self.hdim))
        xavier_uniform_for_relmats_(self.relmats_G, RELMATINITMULT*torch.nn.init.calculate_gain("relu"))
        xavier_uniform_for_relmats_(self.relmats_Q, RELMATINITMULT*torch.nn.init.calculate_gain("relu"))
        xavier_uniform_for_relmats_(self.relmats_K, RELMATINITMULT*torch.nn.init.calculate_gain("relu"))
        if not self.useGasV:
            xavier_uniform_for_relmats_(self.relmats_V, RELMATINITMULT*torch.nn.init.calculate_gain("relu"))
        # init.kaiming_uniform_(self.relmats_G, a=math.sqrt(5))
        # init.kaiming_uniform_(self.relmats_Q, a=math.sqrt(5))
        # init.kaiming_uniform_(self.relmats_K, a=math.sqrt(5))
        # init.kaiming_uniform_(self.relmats_V, a=math.sqrt(5))
        self.dropout = torch.nn.Dropout(dropout)

    def mha(self, query, key, val):
        queries = self.dropout(query)
        context = self.dropout(key)    # bsd
        queries = queries.view(queries.size(0), self.numheads, -1)  # bhl
        context = context.view(context.size(0), context.size(1), self.numheads, -1).transpose(1, 2)     # bhsl
        weights = torch.einsum("bhd,bhsd->bhs", queries, context) / np.sqrt(context.size(-1))
        weights = self.attn_dropout(weights)
        alphas = torch.softmax(weights, -1)      # bhs

        values = val
        values = self.dropout(values)
        values = values.view(values.size(0), values.size(1), self.numheads, -1).transpose(1, 2)
        red = torch.einsum("bhs,bhsd->bhd", alphas, values)
        red = red.view(red.size(0), -1)
        return red

    def message_func(self, edges):
        src_hs = edges.src["h"]
        tgt_hs = edges.dst["h"]
        # do transformation by relation-specific G's
        relmats = self.relmats_G[edges.data["id"]]
        src_hs = torch.einsum("bh,bhd->bd", src_hs, relmats)
        tgt_hs = torch.einsum("bh,bhd->bd", tgt_hs, relmats)
        # do transformation by relation-specific Q's and K's
        relmats_Q = self.relmats_Q[edges.data["id"]]
        relmats_K = self.relmats_K[edges.data["id"]]
        src_hs = self.dropout(src_hs)
        tgt_hs = self.dropout(tgt_hs)
        src_hs = torch.einsum("bh,bhd->bd", src_hs, relmats_K)
        tgt_hs = torch.einsum("bh,bhd->bd", tgt_hs, relmats_Q)
        # do multi-head attention score computation
        src_hs = src_hs.view(src_hs.size(0), self.numheads, -1)     # bhl
        tgt_hs = tgt_hs.view(tgt_hs.size(0), self.numheads, -1)     # bhl
        logits = (src_hs * tgt_hs).sum(-1) / np.sqrt(src_hs.size(-1))  # bh
        if not self.useGasV:
            # do value transformation of source nodes
            relmats_V = self.relmats_V[edges.data["id"]]
        else:
            relmats_V = relmats
        values = edges.src["h"]
        values = self.dropout(values)
        values = torch.einsum("bh,bhd->bd", values, relmats_V)
        values = values.view(values.size(0), self.numheads, -1)     # bhl
        return {"att_score": logits, "values": values}

    def reduce_func(self, nodes):
        logits = nodes.mailbox["att_score"]     # batsize x numneighbours x numheads
        values = nodes.mailbox["values"]        # batsize x numneighbours x numheads x dim
        logits = logits.transpose(1, 2)         # bhs
        values = values.transpose(1, 2)         # bhsd
        alphas = torch.softmax(logits, -1)      # bhs
        # red = torch.einsum("bhs,bhsd->bhd", alphas, values)     # bhd
        red = torch.einsum("bhs,bhsd->bhd", alphas, values)     # bhd
        red = red.view(red.size(0), -1)
        return {"red": red}

    def apply_node_func(self, nodes):
        red = nodes.data["red"]
        h = self.act(red)
        return {"h": h}

    def forward(self, g, step=0):
        g.update_all(self.message_func, self.reduce_func, self.apply_node_func)

    def reset_dropout(self):
        pass
        # device = self.dropout_mask.device
        # ones = torch.ones(self.hdim, device=device)
        # self.dropout_mask = self.dropout(ones)
        # self.dropout_red_mask = self.dropout_red(ones)

    def init_node_states(self, g, batsize, device):
        g.ndata["red"] = torch.zeros(batsize, self.hdim, device=device)


class BRGATCell(torch.nn.Module):
    def __init__(self, hdim, dropout=0., numrels=1, **kw):
        super(BRGATCell, self).__init__(**kw)
        self.hdim, self.dropout, self.numrels = hdim, dropout, numrels


    def init_node_states(self, g, batsize, device):
        g.ndata["red"] = torch.zeros(batsize, self.hdim, device=device)


class GGNN(torch.nn.Module):
    def __init__(self, hdim, numlayers=1, numrepsperlayer=1, numrels=1, dropout=0., rdim=None, use_dgru=False, **kw):
        super(GGNN, self).__init__(**kw)
        self.hdim = hdim
        self.layers = torch.nn.ModuleList([
            GGNNCell(hdim, numrels=numrels, dropout=dropout,
                      rdim=rdim, use_dgru=use_dgru, **kw)
            for _ in range(numlayers)
        ])
        self.numrepsperlayer = numrepsperlayer

    def reset_dropout(self):
        for layer in self.layers:
            layer.reset_dropout()

    def init_node_states(self, g, batsize, device):
        self.layers[0].init_node_states(g, batsize, device)

    def forward(self, g, step=None):
        if step is None:
            for layer in self.layers:
                for _ in range(self.numrepsperlayer):
                    layer(g)
        else:
            # assert(step is not None)
            _step = step/self.numrepsperlayer
            _step = math.floor(_step)
            _step = min(_step, len(self.layers) - 1)
            layer = self.layers[_step]
            layer(g, step=None)
        return g


class GGNNCell(torch.nn.Module):       # RELATIONS: adding vectors
    def __init__(self, hdim, dropout=0., numrels=1, rdim=None, use_dgru=False, **kw):
        super(GGNNCell, self).__init__(**kw)
        self.hdim = hdim
        self.rdim = self.hdim if rdim is None else rdim
        if use_dgru:
            self.node_gru = DGRUCell(self.hdim, self.hdim, dropout=dropout)
        else:
            self.node_gru = GRUCell(self.hdim, self.hdim, dropout=dropout)

        # self.rellin = torch.nn.Linear(self.rdim, self.rdim, bias=False)
        if self.hdim != self.rdim:
            self.relproj_in = torch.nn.Linear(self.hdim, self.rdim, bias=False)
            self.relproj_out = torch.nn.Linear(self.rdim, self.hdim, bias=False)
        # self.relvectors = torch.nn.Parameter(torch.randn(numrels, self.hdim))
        # init.kaiming_uniform_(self.relvectors, a=math.sqrt(5))
        self.relmats = torch.nn.Parameter(torch.randn(numrels, self.rdim, self.rdim))
        xavier_uniform_for_relmats_(self.relmats, RELMATINITMULT*torch.nn.init.calculate_gain("relu"))

        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer("dropout_mask", torch.ones(self.hdim))

    def reset_dropout(self):
        if hasattr(self.node_gru, "reset_dropout"):
            self.node_gru.reset_dropout()
        device = self.dropout_mask.device
        ones = torch.ones(self.hdim, device=device)
        self.dropout_mask = self.dropout(ones).clamp_max(1)

    def init_node_states(self, g, batsize, device):
        g.ndata["red"] = torch.zeros(batsize, self.hdim, device=device)

    def message_func(self, edges):
        hs = edges.src["h"]
        relmats = self.relmats[edges.data["id"]]

        if self.rdim != self.hdim:
            _hs = self.relproj_in(hs)
        else:
            _hs = hs

        _hs = torch.einsum("bh,bhd->bd", _hs, relmats)

        if self.rdim != self.hdim:
            _hs = self.relproj_out(_hs)
            msg = hs + _hs
        else:
            msg = hs
        # relvecs = self.relvectors[edges.data["id"]]
        # msg = edges.src["h"]
        # msg = self.rellin(msg)
        # msg = msg + relvecs
        # msg = self.relgru(relvecs, msg)
        return {"msg": msg}

    def reduce_func(self, nodes):
        red = nodes.mailbox["msg"].mean(1)
        # red = nodes.mailbox["msg"].sum(1)
        return {"red": red}

    def apply_node_func(self, nodes):
        # h = self.node_gru(self.dropout(nodes.data["red"]), nodes.data["h"] * self.dropout_mask[None, :])
        h = self.node_gru(nodes.data["red"], nodes.data["h"])
        return {"h": h}

    def forward(self, g, step=0):
        g.update_all(self.message_func, self.reduce_func, self.apply_node_func)


class RelGraphLSTM(torch.nn.Module):
    def __init__(self, hdim, numlayers=1, numrepsperlayer=1, numrels=1, dropout=0.,
                 rdim=None, relmode="default", **kw):
        super(RelGraphLSTM, self).__init__(**kw)
        self.hdim = hdim
        self.layers = torch.nn.ModuleList([
            RelGraphLSTMCell(hdim, numrels=numrels, dropout=dropout,
                      rdim=rdim, relmode=relmode, **kw)
            for _ in range(numlayers)
        ])
        self.numrepsperlayer = numrepsperlayer

    def reset_dropout(self):
        for layer in self.layers:
            layer.reset_dropout()

    def init_node_states(self, g, batsize, device):
        self.layers[0].init_node_states(g, batsize, device)

    def forward(self, g, step=None):
        if step is None:
            for layer in self.layers:
                for _ in range(self.numrepsperlayer):
                    g = layer(g)
        else:
            # assert(step is not None)
            _step = step/self.numrepsperlayer
            _step = math.floor(_step)
            _step = min(_step, len(self.layers) - 1)
            layer = self.layers[_step]
            g = layer(g, step=None)
        return g


class RelGraphLSTMCell(torch.nn.Module):
    def __init__(self, hdim, numrels=1, dropout=0, rdim=None, relmode="default", **kw):
        super(RelGraphLSTMCell, self).__init__(**kw)
        self.hdim = hdim
        self.rdim = self.hdim if rdim is None else rdim
        self.numrels = numrels

        self.relmode = relmode
        if self.relmode == "gatedcatmap" or self.relmode == "gated":
            indim = self.hdim + (self.hdim if self.cat_tgt else 0) + (self.rdim if self.cat_rel else 0)
            self.msg_block = GatedCatMap(indim, self.hdim, zdim=self.zdim, dropout=dropout)
            self.relvectors = torch.nn.Parameter(torch.randn(numrels, self.rdim))
            init.kaiming_uniform_(self.relvectors, a=math.sqrt(5))
        elif self.relmode == "default" or self.relmode == "addrelu" or self.relmode == "addmul":
            self.relvectors = torch.nn.Parameter(torch.randn(numrels, self.rdim))
            self.relvectors2 = torch.nn.Parameter(torch.randn(numrels, self.rdim))
            init.kaiming_uniform_(self.relvectors, a=math.sqrt(5))
            init.kaiming_uniform_(self.relvectors2, a=math.sqrt(5))
            self.relvectors2.data = self.relvectors2.data - 3
            self.lrelu = torch.nn.LeakyReLU(0.1)
        elif self.relmode == "baseline":
            self.relproj_in = torch.nn.Linear(self.hdim, self.rdim, bias=False) if self.rdim != self.hdim else None
            self.relproj_out = torch.nn.Linear(self.rdim, self.hdim, bias=False) if self.rdim != self.hdim else None
            self.relmats = torch.nn.Parameter(torch.randn(numrels, self.rdim, self.rdim))
            # init.uniform_(self.relmats, -0.1, 0.1)
            # init.xavier_uniform_(self.relmats, gain=torch.nn.init.calculate_gain("relu"))
            xavier_uniform_for_relmats_(self.relmats, gain=RELMATINITMULT * torch.nn.init.calculate_gain('relu'))
            self.cat_rel = False
        else:
            raise Exception(f"unknown relmode '{relmode}'")

        self.dropout = torch.nn.Dropout(dropout)

        self.ln_att = torch.nn.LayerNorm(hdim)
        self.ln_csum = torch.nn.LayerNorm(hdim)
        self.ln_hsum = torch.nn.LayerNorm(hdim)

        self.inp_lin = torch.nn.Linear(hdim*2, hdim)
        self.out_lin = torch.nn.Linear(hdim*2, hdim)
        self.upd_lin = torch.nn.Linear(hdim*2, hdim)
        self.fgt_lin = torch.nn.Linear(hdim*2, hdim)

    def reset_dropout(self):
        pass

    def init_node_states(self, g, batsize, device):
        g.ndata["hsum"] = torch.zeros(batsize, self.hdim, device=device)
        g.ndata["csum"] = torch.zeros(batsize, self.hdim, device=device)
        # g.ndata["c"] = torch.zeros(batsize, self.hdim, device=device)

    def message_func(self, edges):
        hs = edges.src["h"]
        cs = edges.src["c"]
        tgt_hs = edges.dst["h"]

        if self.relmode in ("rescatmap", "gatedcatmap", "gated", "residual"):
            if "emb" in edges.data:  # if "emb" in edata, then use those
                relvecs = edges.data["emb"]
            else:
                relvecs = self.relvectors[edges.data["id"]]
            inps = [hs, relvecs]
            hs = self.msg_block(*inps)
            msg = hs

        elif self.relmode in ("default", "addrelu"):
            if "emb" in edges.data:  # if "emb" in edata, then use those
                relvecs = edges.data["emb"]
            else:
                relvecs = self.relvectors[edges.data["id"]]
            hs = hs + relvecs
            hs = self.lrelu(hs)
            cs = cs + relvecs
            cs = self.lrelu(cs)
            msg = hs

        elif self.relmode == "addmul":
            if "emb" in edges.data:  # if "emb" in edata, then use those
                relvecs = edges.data["emb"]
            else:
                relvecs = self.relvectors[edges.data["id"]]
            relvecs2 = torch.sigmoid(self.relvectors2[edges.data["id"]])
            hs = hs * (1 - relvecs2) + relvecs * relvecs2
            msg = hs

        elif self.relmode == "baseline":
            assert False, "this code must be fixed first"
            relmats = self.relmats[edges.data["id"]]
            _hs = self.relproj_in(hs) if self.relproj_in is not None else hs
            _hs = torch.einsum("bh,bhd->bd", _hs, relmats)
            if self.relproj_out is not None:
                _hs = self.relproj_out(_hs)
                msg = hs + _hs
            else:
                msg = _hs

        fgt = self.fgt_lin(torch.cat([hs, tgt_hs], -1))
        fgt = torch.sigmoid(fgt)
        cs = fgt * cs

        return {"msg": msg, "hs": hs, "cs": cs}

    def reduce_func(self, nodes):  # there were no dropouts at all in here !!!
        hsum = nodes.mailbox["hs"].sum(1)
        csum = nodes.mailbox["cs"].sum(1)
        return {"hsum": hsum, "csum": csum}

    def apply_node_func(self, nodes):
        h = nodes.data["h"]
        c = nodes.data["c"]
        hsum = nodes.data["hsum"]
        hsum = self.ln_hsum(hsum)
        csum = nodes.data["csum"]
        csum = self.ln_csum(csum)
        gate_inp = torch.cat([h, hsum], -1)
        gate_inp = self.dropout(gate_inp)
        inp_gate = torch.sigmoid(self.inp_lin(gate_inp))
        out_gate = torch.sigmoid(self.out_lin(gate_inp))
        upd_c = torch.tanh(self.upd_lin(gate_inp))

        new_c = upd_c * inp_gate + csum
        new_h = new_c * out_gate
        return {"h": new_h, "c": new_c}

    def forward(self, g, step=0):
        g.update_all(self.message_func, self.reduce_func, self.apply_node_func)
        return g



class RVGAT(torch.nn.Module):
    dropout_in_reduce = False
    def __init__(self, hdim, dropout=0., dropout_red=-1, numrels=1, numheads=4,
                 usevallin=False, relmode="prelu", cell="none", **kw):
        super(RVGAT, self).__init__(**kw)
        self.hdim = hdim

        if cell == "none":
            self.node_cell = Identity()
        elif cell == "relu":
            self.node_cell = torch.nn.ReLU()
        elif cell == "lrelu":
            self.node_cell = torch.nn.LeakyReLU(0.1)
        elif cell == "linskip":
            self.node_cell = LinearSkip(dim=self.hdim, act=torch.nn.ReLU(), bias=True, dropout=dropout)

        self.relmode = relmode
        self.relvectors = torch.nn.Parameter(torch.randn(numrels, self.hdim))
        # self.self_relvector = torch.nn.Parameter(torch.randn(self.hdim))
        # self.relvectors_add = torch.nn.Parameter(torch.zeros(numrels, self.hdim))
        self.relvectors_mul = torch.nn.Parameter((torch.rand_like(self.relvectors)-0.5)*1. + 3.)
        init.kaiming_uniform_(self.relvectors, a=math.sqrt(5))
        # init.uniform_(self.self_relvector, -0.1, 0.1)
        # init.kaiming_uniform_(self.relvectors_add, a=math.sqrt(5))
        self.relmats = torch.nn.Parameter(torch.randn(numrels, self.hdim, self.hdim))
        init.uniform_(self.relmats, -0.1, 0.1)

        if relmode == "gatedcatpremap" or relmode == "gatedcatpostmap":
            self.catblock = GatedCatMapBlock(self.hdim, dropout=0., zoneout=0.)
        elif relmode == "catpremap" or relmode == "catpostmap":
            # self.catblock = CatMapBlock(self.hdim)
            self.catblock = SimpleCatMapBlock(self.hdim, dropout=0.)

        self.attention = WindowHeadAttention(self.hdim, self.hdim * 2, self.hdim, self.hdim, numheads=numheads,
                                             dropout=0., attn_dropout=0.,
                                             usevallin=usevallin)

        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer("dropout_mask", torch.ones(self.hdim))
        dropout_red = dropout if dropout_red < 0 else dropout_red
        self.dropout_red = torch.nn.Dropout(dropout_red)
        self.register_buffer("dropout_red_mask", torch.ones(self.hdim))

        # ablations
        self.usevallin = usevallin

        self.layernorm = torch.nn.LayerNorm(self.hdim)

    def reset_dropout(self):
        device = self.dropout_mask.device
        ones = torch.ones(self.hdim, device=device)
        self.dropout_mask = self.dropout(ones)
        self.dropout_red_mask = self.dropout_red(ones)

    def init_node_states(self, g, batsize, device):
        g.ndata["red"] = torch.zeros(batsize, self.hdim, device=device)

    def message_func(self, edges):
        hs = edges.src["h"]
        relvecs = self.relvectors[edges.data["id"]]
        # relvecs_add = self.relvectors_add[edges.data["id"]]
        relvecs_mul = self.relvectors_mul[edges.data["id"]]

        if self.relmode in ("catpremap", "gatedcatpremap"):
            hs = self.catblock(hs, relvecs)
        # if self.relmode == "add":
        #     hs = hs + relvecs_add
        elif self.relmode == "original" or self.relmode == "originalmul":
            hs = hs + relvecs
        elif self.relmode == "prelu":
            hs = hs + relvecs
            hs = torch.max(hs, torch.zeros_like(hs)) \
                 + torch.sigmoid(relvecs_mul) * torch.min(hs, torch.zeros_like(hs))
            # hs = torch.nn.functional.prelu(hs + relvecs, relvecs_add)
        elif self.relmode == "mix":
            hs = hs * torch.sigmoid(relvecs_mul) + relvecs * (1-torch.sigmoid(relvecs_mul))

        msg = torch.cat([hs, relvecs], -1)

        if self.relmode == "originalmul":
            hs = hs * torch.sigmoid(relvecs_mul)
        elif self.relmode in ("catpostmap", "gatedcatpostmap"):
            hs = self.catblock(hs, relvecs)
        # elif self.relmode == "addmul":
        #     hs = (hs + relvecs_add) * torch.sigmoid(relvecs_mul)
        return {"msg": msg, "hs": hs}

    def reduce_func(self, nodes):       # there were no dropouts at all in here !!!
        queries = nodes.data["h"]
        keys, values = nodes.mailbox["msg"], nodes.mailbox["hs"]

        # cat_to_keys = torch.cat([nodes.data["h"], self.self_relvector[None, :].repeat(keys.size(0), 1)], -1)
        # keys = torch.cat([keys, cat_to_keys[:, None, :]], 1)
        values = torch.cat([values, nodes.data["h"][:, None, :]], 1)
        if self.dropout_in_reduce:
            queries = queries * self.dropout_red_mask[None, :]
            keys = keys * torch.cat([self.dropout_red_mask[None, None, :], self.dropout_red_mask[None, None, :]], -1)
            # if self.usevallin:
            #     pass
            #     # values = self.dropout_red(values)
            #
            # else:
            #     values = values * self.dropout_mask[None, None, :].clamp_max(1)
        red = self.attention(queries, keys, values)
        return {"red": red}

    def apply_node_func(self, nodes):
        if self.dropout_in_reduce:
            # inp = nodes.data["red"]
            inp = (nodes.data["red"] + nodes.data["h"]) / 2
            h = self.node_cell(inp)
        else:
            # inp = nodes.data["red"]
            inp = (nodes.data["red"] + nodes.data["h"]) / 2      # this is skip before node cell
            if self.usevallin and not isinstance(self.node_cell, LinearSkip):
                inp = inp * self.dropout_mask[None, :]
            else:
                inp = inp# * self.dropout_mask[None, :].clamp_max(1)
            h = self.node_cell(inp)
        return {"h": h}

    def forward(self, g, step=0):
        g.update_all(self.message_func, self.reduce_func, self.apply_node_func)


class CatMapBlock(torch.nn.Module):
    def __init__(self, dim, **kw):
        super(CatMapBlock, self).__init__(**kw)
        self.dim = dim
        self.lin = torch.nn.Linear(self.dim * 2, self.dim)

    def forward(self, h, x):
        _h = torch.cat([h, x], -1)
        _h = self.lin(_h)
        ret = _h + h
        return ret


class SimpleCatMapBlock(torch.nn.Module):
    def __init__(self, dim, dropout=0., **kw):
        super(SimpleCatMapBlock, self).__init__(**kw)
        self.dim = dim
        self.linA = torch.nn.Linear(self.dim * 2, self.dim * 4)
        self.act = torch.nn.CELU()
        self.linB = torch.nn.Linear(self.dim * 4, self.dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, a, b):
        h = torch.cat([a, b], -1)
        _h = self.linA(self.dropout(h))
        _h = self.act(_h)
        cand = self.linB(_h)
        ret = cand + a
        return ret


class GatedCatMapBlock(torch.nn.Module):
    def __init__(self, dim, dropout=0., zoneout=0., act=None, odim=None, **kw):
        super(GatedCatMapBlock, self).__init__(**kw)
        self.dim = dim
        self.odim = self.dim if odim is None else odim
        self.linA = torch.nn.Linear(self.dim * 2, self.dim * 4)
        self.act = torch.nn.CELU() if act is None else act
        self.linB = torch.nn.Linear(self.dim * 4, self.dim)
        self.linMix = torch.nn.Linear(self.dim * 4, self.dim)
        self.linMix.bias.data.fill_(3.)
        self.dropout = torch.nn.Dropout(dropout)
        self.zoneout = zoneout

    def forward(self, h, x):
        _h = torch.cat([h, x], -1)
        _cand = self.linA(self.dropout(_h))
        _cand = self.act(_cand)
        cand = self.linB(_cand)
        mix = torch.sigmoid(self.linMix(_cand))
        ret = h * mix + cand * (1 - mix)

        # zoneout
        if self.zoneout > 0 and self.training:
            zo_rand = torch.rand_like(ret)
            a = zo_rand <= self.zoneout
            ret = h * a.float() + ret * (~a).float()
        return ret


class GatedCatMap(torch.nn.Module):
    def __init__(self, indim, odim, dropout=0., zoneout=0., activation=None, zdim=None, usegradskip=False, use_layernorm=True, **kw):
        super(GatedCatMap, self).__init__(**kw)
        self.dim = indim
        self.odim = odim
        self.zdim = self.dim * 1. if zdim is None else zdim

        self.activation = torch.nn.CELU() if activation is None else activation

        self.linA = torch.nn.Linear(self.dim, self.zdim)
        self.linB = torch.nn.Linear(self.zdim, self.odim)
        self.linMix = torch.nn.Linear(self.zdim, self.odim)
        self.linMix.bias.data.fill_(3.)

        self.dropout = torch.nn.Dropout(dropout)
        self.zoneout = zoneout

        self.ln = None
        if use_layernorm:
            self.ln = torch.nn.LayerNorm(indim)

        self.usegradskip = usegradskip

    def forward(self, *inps):
        h = inps[0]
        _h = torch.cat(inps, -1)
        if self.ln is not None:
            _h = self.ln(_h)
        _h = self.dropout(_h)
        _cand = self.linA(_h)
        _cand = self.activation(_cand)
        cand = self.linB(self.dropout(_cand))
        mix = torch.sigmoid(self.linMix(_cand))
        if self.usegradskip:
            ret = forward_mix_backward_sum(h, cand, mix)
        else:
            ret = h * mix + cand * (1 - mix)

        # zoneout
        if self.zoneout > 0 and self.training:
            zo_rand = torch.rand_like(ret)
            a = zo_rand <= self.zoneout
            ret = h * a.float() + ret * (~a).float()
        return ret


class ResidualEdgeUpdate(torch.nn.Module):
    def __init__(self, indim, odim, dropout=0., act_dropout=0., activation=None, zdim=None, nodamper=False, **kw):
        super(ResidualEdgeUpdate, self).__init__(**kw)
        self.dim = indim
        self.odim = odim
        self.zdim = self.dim * 1. if zdim is None else zdim

        self.activation = torch.nn.CELU() if activation is None else activation

        self.linA = torch.nn.Linear(self.dim, self.zdim)
        self.linB = torch.nn.Linear(self.zdim, self.odim)

        self.dropout = torch.nn.Dropout(dropout)
        self.act_dropout = torch.nn.Dropout(act_dropout)
        self.ln = torch.nn.LayerNorm(indim)
        self.ln2 = torch.nn.LayerNorm(odim)

        self.damper = torch.nn.Parameter(torch.zeros(self.odim) - 2)
        self.nodamper = nodamper

    def forward(self, *inps):
        h = inps[0]
        x = torch.cat(inps, -1)
        x = self.ln(x)
        # _h = self.dropout(_h)
        x = self.linA(x)
        x = self.activation(x)
        x = self.dropout(x)
        # x = self.act_dropout(x)
        x = self.linB(x)
        if not self.nodamper:
            x = torch.sigmoid(self.damper) * x
        ret = x + h
        # ret = self.ln2(ret)
        return ret


class ResidualNodeUpdate(ResidualEdgeUpdate):
    def forward(self, *inps):
        h = inps[0]
        _h = h
        # _h = torch.cat(inps, -1)
        _h = self.ln(_h)
        # _h = self.dropout(_h)
        _cand = self.linA(_h)
        _cand = self.activation(_cand)
        _cand = self.dropout(_cand)
        # _cand = self.act_dropout(_cand)
        cand = self.linB(_cand)
        if not self.nodamper:
            cand = torch.sigmoid(self.damper) * cand
        ret = cand + h
        # ret = self.ln2(ret)
        return ret


class DeepRGCN(torch.nn.Module):
    def __init__(self, hdim, numlayers=1, numrels=1, dropout=0.,
                 residual=True, **kw):
        super(DeepRGCN, self).__init__(**kw)
        self.hdim = hdim
        self.layers = torch.nn.ModuleList([
            DeepGCNCell(hdim, numrels=numrels, dropout=dropout, **kw)
            for _ in range(numlayers)
        ])
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(hdim) for _ in range(numlayers)])
        self.dropout = torch.nn.Dropout(dropout)
        self.residual = residual
        self.numlayers = numlayers

    def reset_dropout(self):
        for layer in self.layers:
            layer.reset_dropout()

    def init_node_states(self, g, batsize, device):
        self.layers[0].init_node_states(g, batsize, device)

    def forward(self, g, step=None):
        assert(step is None)
        g = self.layers[0](g)
        for layernr in range(1, self.numlayers):
            h = g.ndata["h"]
            _h = self.norms[layernr-1](h)
            _h = torch.relu(_h)
            _h = self.dropout(_h)
            g.ndata["h"] = _h

            g = self.layers[layernr](g)

            if self.residual:
                _h = g.ndata["h"]
                _h = h + _h
                g.ndata["h"] = _h

        h = g.ndata["h"]
        h = self.norms[-1](h)
        h = self.dropout(h)
        g.ndata["h"] = h
            # _step = step/self.numrepsperlayer
            # _step = math.floor(_step)
            # _step = min(_step, len(self.layers) - 1)
            # layer = self.layers[_step]
            # norm = self.norms[_step]
            #
            # g = layer(g, step=None)
            # g = norm
            # g = layer(g, step=None)
        return g


class DeepGCNCell(torch.nn.Module):
    def __init__(self, dim, dropout=0., numrels=5, residual=True, **kw):
        super(DeepGCNCell, self).__init__(**kw)
        self.lin = torch.nn.Linear(dim, dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.residual = residual
        self.ln = torch.nn.LayerNorm(dim)
        self.hdim = dim

        self.relvectors = torch.nn.Parameter(torch.randn(numrels, self.hdim))
        init.kaiming_uniform_(self.relvectors, a=math.sqrt(5))

    def reset_dropout(self):
        pass

    def init_node_states(self, g, batsize, device):
        g.ndata["red"] = torch.zeros(batsize, self.hdim, device=device)

    def message_func(self, edges):
        hs = edges.src["h"]
        if "emb" in edges.data:
            relvecs = edges.data["emb"]
        else:
            relvecs = self.relvectors[edges.data["id"]]
        msg = hs + relvecs
        # msg = self.ln(msg)
        msg = torch.relu(msg)
        return {"msg": msg, "hs": hs}

    def reduce_func(self, nodes):  # there were no dropouts at all in here !!!
        msg, hs = nodes.mailbox["msg"], nodes.mailbox["hs"]
        degree = hs.size(1)
        red = msg.mean(1)
        # deg_inv_sqrt = degree ** -0.5
        # if deg_inv_sqrt == float('inf'):
        #     deg_inv_sqrt = 0
        # red = red * deg_inv_sqrt
        return {"red": red, "degree": torch.ones_like(red[:, 0])*degree}

    def apply_node_func(self, nodes):
        h = nodes.data["h"]
        red = nodes.data["red"]
        # degree = nodes.data["degree"]
        # # _h =
        # deg_inv_sqrt = degree.pow(-0.5)
        # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # red = red * deg_inv_sqrt[:, None]
        # _h = h + red
        red = self.lin(red)
        return {"h": red}

    def forward(self, g, step=0):
        g.update_all(self.message_func, self.reduce_func, self.apply_node_func)
        return g


GATE_BIAS = 0
class SGRU(torch.nn.Module):
    def __init__(self, dim, bias=True, dropout=0., gate_bias=GATE_BIAS, gate_bias_skip=0, gate_bias_hskip=0, use_layernorm=True, **kw):
        super(SGRU, self).__init__(**kw)
        self.dim, self.bias = dim, bias
        self.gateW = torch.nn.Linear(dim * 2, dim * 5, bias=bias)
        self.gateU = torch.nn.Linear(dim * 2, dim, bias=bias)
        self.sm = torch.nn.Softmax(-1)
        self.dropout = torch.nn.Dropout(dropout)
        # self.register_buffer("dropout_mask", torch.ones(self.dim * 2))
        self.gate_bias = gate_bias
        self.gate_bias_skip = gate_bias_skip
        self.gate_bias_hskip = gate_bias_hskip

        # apply custom bias biases
        self.gateW.bias.data[2*dim:3*dim] += self.gate_bias_hskip
        self.gateW.bias.data[3*dim:4*dim] += self.gate_bias_skip
        self.gateW.bias.data[4*dim:5*dim] -= self.gate_bias

        self.ln = None
        self.ln2 = None
        if use_layernorm:
            self.ln = torch.nn.LayerNorm(dim * 2)
            self.ln2 = torch.nn.LayerNorm(dim * 2)

    def reset_dropout(self):
        pass

    def forward(self, x, h):
        inp = torch.cat([x, h], 1)
        if self.ln is not None:
            inp = self.ln(inp)
        inp = self.dropout(inp)
        # inp = inp * self.dropout_mask[None, :]
        gates = self.gateW(inp)
        gates = list(gates.chunk(5, 1))
        rx = torch.sigmoid(gates[0])
        rh = torch.sigmoid(gates[1])
        z_gates = gates[2:5]
        # z_gates[0] = z_gates[0] + self.gate_bias_hskip
        # z_gates[1] = z_gates[1] + self.gate_bias_skip       # apply gate bias on the skip connection
        # z_gates[2] = z_gates[2] - self.gate_bias            # apply gate bias on the update
        z = torch.softmax(torch.stack(z_gates, -1), -1)
        inp = torch.cat([x * rx, h * rh], 1)
        if self.ln2 is not None:
            inp = self.ln2(inp)
        inp = self.dropout(inp)
        # inp = inp * self.dropout_mask[None, :]
        u = self.gateU(inp)
        u = torch.tanh(u)
        h_new = torch.stack([x, h, u], 2) * z
        h_new = h_new.sum(-1)
        return h_new


class ResRGAT(torch.nn.Module):
    def __init__(self, hdim, numlayers=1, numrepsperlayer=1, numrels=1, numheads=4, dropout=0., dropout_attn=0., dropout_act=0.,
                 dropout_red=0., rdim=None, usevallin=False, norel=False, use_gate=False, use_sgru=False, use_noderes=False,
                 cat_rel=True, cat_tgt=False, no_msg=False, no_resmsg=False,
                 skipatt=False, **kw):
        super(ResRGAT, self).__init__(**kw)
        self.hdim = hdim
        self.norel = norel
        self.layers = torch.nn.ModuleList([
            ResRGATCell(hdim, numrels=numrels, numheads=numheads,
                      dropout=dropout, dropout_red=dropout_red, dropout_attn=dropout_attn, dropout_act=dropout_act,
                      rdim=rdim, usevallin=usevallin, norel=norel,
                      cat_rel=cat_rel, cat_tgt=cat_tgt, use_gate=use_gate, use_sgru=use_sgru, use_noderes=use_noderes,
                      skipatt=skipatt, no_msg=no_msg, no_resmsg=no_resmsg, **kw)
            for _ in range(numlayers)
        ])
        self.numrepsperlayer = numrepsperlayer

    def reset_dropout(self):
        for layer in self.layers:
            layer.reset_dropout()

    def init_node_states(self, g, batsize, device):
        self.layers[0].init_node_states(g, batsize, device)

    def forward(self, g, step=None):
        if step is None:
            for layer in self.layers:
                for _ in range(self.numrepsperlayer):
                    g = layer(g)
        else:
            # assert(step is not None)
            _step = step/self.numrepsperlayer
            _step = math.floor(_step)
            _step = min(_step, len(self.layers) - 1)
            layer = self.layers[_step]
            g = layer(g, step=None)
        return g


class ResRGATCell(torch.nn.Module):
    def __init__(self, hdim, numrels=1, numheads=4, dropout=0., dropout_attn=0., dropout_act=0.,
                 dropout_red=0., rdim=None, usevallin=False, norel=False,
                 cat_rel=True, cat_tgt=False, use_gate=False, use_sgru=False,
                 skipatt=False, no_msg=False, no_resmsg=False, use_noderes=False, **kw):
        super(ResRGATCell, self).__init__(**kw)
        self.cat_rel, self.cat_tgt = cat_rel, cat_tgt
        self.hdim = hdim
        self.rdim = self.hdim if rdim is None else rdim
        indim = self.hdim + (self.hdim if self.cat_tgt else 0) + (self.rdim if True else 0)
        self.zdim = hdim
        self.numrels = numrels
        self.norel = norel
        self.skipatt = skipatt
        self.use_gate = use_gate
        self.use_sgru = use_sgru

        self.use_noderes = use_noderes

        self.no_msg = no_msg
        self.no_resmsg = no_resmsg

        if norel:
            self.cat_rel = False
        else:
            self.activation = torch.nn.CELU()

            self.linA = torch.nn.Linear(indim, self.zdim)
            self.linB = torch.nn.Linear(self.zdim, self.hdim)

            self.ln = torch.nn.LayerNorm(indim)
            self.ln2 = torch.nn.LayerNorm(self.hdim)
            self.relvectors = torch.nn.Parameter(torch.randn(numrels, self.rdim))
            init.kaiming_uniform_(self.relvectors, a=math.sqrt(5))

        self.linGate = None
        if self.use_gate:
            self.linGate = torch.nn.Linear(self.zdim, self.hdim)
            self.linGate.bias.data.fill_(3.)

        if self.use_sgru:
            self.sgru = SGRU(self.hdim, dropout=dropout, gate_bias_hskip=2)

        self.attention = MultiHeadAttention(self.hdim, (self.hdim + self.rdim) if self.cat_rel else self.hdim,
                                            self.hdim, self.hdim, numheads=numheads, use_layernorm=False,
                                            dropout=0., attn_dropout=dropout_attn, usevallin=usevallin)

        self.dropout = torch.nn.Dropout(dropout)
        self.dropout_red = torch.nn.Dropout(dropout_red)
        self.dropout_act = torch.nn.Dropout(dropout_act)

        # ablations
        self.usevallin = usevallin

        self.ln_att = torch.nn.LayerNorm(hdim)

    def reset_dropout(self):
        pass

    def init_node_states(self, g, batsize, device):
        g.ndata["red"] = torch.zeros(batsize, self.hdim, device=device)

    def message_func(self, edges):
        hs = edges.src["h"]
        if self.no_msg is True:
            return {"msg": hs, "hs": hs}
        if self.norel:
            relvecs = torch.zeros(hs.size(0), self.rdim)
        elif "emb" in edges.data:             # if "emb" in edata, then use those
            relvecs = edges.data["emb"]
        else:
            relvecs = self.relvectors[edges.data["id"]]
        inps = [hs, relvecs]
        if self.cat_tgt:    # False by default
            inps.append(edges.dst["h"])

        # residual update
        x = torch.cat(inps, -1)
        x = self.dropout_act(x)
        x = self.ln(x)
        x = self.linA(x)
        x = self.activation(x)
        # x = self.dropout_act(x)
        _x = self.linB(x)
        _x = self.dropout(_x)
        if self.use_gate:
            g = self.linGate(x)
            g = torch.sigmoid(g)
            if self.no_resmsg is True:
                hs = _x * (1 - g)
            else:
                hs = hs * g + _x * (1 - g)
        else:
            if self.no_resmsg is True:
                hs = _x
            else:
                hs = hs + _x
        # if not self.skipatt:
        #     hs = self.ln2(hs)

        if self.cat_rel:  # True by default
            msg = torch.cat([hs, relvecs], -1)
        else:
            msg = hs

        return {"msg": msg, "hs": hs}

    def reduce_func(self, nodes):  # there were no dropouts at all in here !!!
        queries = nodes.data["h"]
        keys, values = nodes.mailbox["msg"], nodes.mailbox["hs"]
        red = self.attention(queries, keys, values)
        if self.skipatt:
            red = red + nodes.data["h"]
            red = self.ln_att(red)
        return {"red": red}

    def apply_node_func(self, nodes):
        assert not (self.use_sgru and self.use_noderes)
        if self.use_sgru:
            h = self.sgru(nodes.data["red"], nodes.data["h"])
        elif self.use_noderes:
            h = nodes.data["red"] + nodes.data["h"]
        else:
            h = nodes.data["red"]
        return {"h": h}

    def forward(self, g, step=0):
        g.update_all(self.message_func, self.reduce_func, self.apply_node_func)
        return g


class LRGAT(torch.nn.Module):
    def __init__(self, hdim, numlayers=1, numrepsperlayer=1, numrels=1, numheads=4, dropout=0., dropout_attn=0., dropout_act=0.,
                 dropout_red=0., rdim=None, usevallin=False, norel=False, use_gate=False, use_sgru=False,
                 cat_rel=True, cat_tgt=False,
                 skipatt=False, **kw):
        super(LRGAT, self).__init__(**kw)
        self.hdim = hdim
        self.norel = norel
        self.layers = torch.nn.ModuleList([
            LRGATCell(hdim, numrels=numrels, numheads=numheads,
                      dropout=dropout, dropout_attn=dropout_attn, rdim=rdim, usevallin=usevallin, residual_vallin=True, norel=norel, **kw)
            for _ in range(numlayers)
        ])
        self.numrepsperlayer = numrepsperlayer

    def reset_dropout(self):
        for layer in self.layers:
            layer.reset_dropout()

    def init_node_states(self, g, batsize, device):
        self.layers[0].init_node_states(g, batsize, device)

    def forward(self, g, step=None):
        if step is None:
            for layer in self.layers:
                for _ in range(self.numrepsperlayer):
                    g = layer(g)
        else:
            # assert(step is not None)
            _step = step/self.numrepsperlayer
            _step = math.floor(_step)
            _step = min(_step, len(self.layers) - 1)
            layer = self.layers[_step]
            g = layer(g, step=None)
        return g


class LRGATCell(torch.nn.Module):
    def __init__(self, hdim, numrels=1, numheads=4, dropout=0., dropout_attn=0., rdim=None,
                 usevallin=False, residual_vallin=False, norel=False, **kw):
        super(LRGATCell, self).__init__(**kw)
        self.hdim = hdim
        self.rdim = self.hdim if rdim is None else rdim
        self.zdim = hdim
        self.numrels = numrels
        self.norel = norel

        if not norel:
            self.relvectors = torch.nn.Parameter(torch.randn(numrels, self.rdim))
            init.kaiming_uniform_(self.relvectors, a=math.sqrt(5))

        self.gru = GRUCell(self.hdim, dropout=dropout)

        self.attention = MultiHeadAttention(self.hdim, self.hdim,
                                            self.hdim, self.hdim, numheads=numheads, use_layernorm=False,
                                            dropout=0., attn_dropout=dropout_attn, usevallin=usevallin, residual_vallin=residual_vallin)

        self.dropout = torch.nn.Dropout(dropout)

        self.ln_att = torch.nn.LayerNorm(hdim)

    def reset_dropout(self):
        pass

    def init_node_states(self, g, batsize, device):
        g.ndata["red"] = torch.zeros(batsize, self.hdim, device=device)

    def message_func(self, edges):
        hs = edges.src["h"]
        if self.norel:
            relvecs = torch.zeros(hs.size(0), self.rdim)
        elif "emb" in edges.data:             # if "emb" in edata, then use those
            relvecs = edges.data["emb"]
        else:
            relvecs = self.relvectors[edges.data["id"]]
        hs = hs + relvecs
        # hs = torch.nn.functional.leaky_relu(hs, 0.1)
        msg = hs

        return {"msg": msg, "hs": hs}

    def reduce_func(self, nodes):  # there were no dropouts at all in here !!!
        queries = nodes.data["h"]
        keys, values = nodes.mailbox["msg"], nodes.mailbox["hs"]
        red = self.attention(queries, keys, values)
        return {"red": red}

    def apply_node_func(self, nodes):
        h = self.gru(nodes.data["h"], nodes.data["red"])
        h = h + nodes.data["h"]
        return {"h": h}

    def forward(self, g, step=0):
        g.update_all(self.message_func, self.reduce_func, self.apply_node_func)
        return g


class LRTM(torch.nn.Module):
    def __init__(self, hdim, numlayers=1, numrepsperlayer=1, numrels=1, numheads=4, dropout=0., norel=False, **kw):
        super(LRTM, self).__init__(**kw)
        self.hdim = hdim
        self.norel = norel
        self.layers = torch.nn.ModuleList([
            LRTMCell(hdim, numrels=numrels, numheads=numheads, dropout=dropout, norel=norel, **kw)
            for _ in range(numlayers)
        ])
        self.numrepsperlayer = numrepsperlayer

    def reset_dropout(self):
        for layer in self.layers:
            layer.reset_dropout()

    def init_node_states(self, g, batsize, device):
        self.layers[0].init_node_states(g, batsize, device)

    def forward(self, g, step=None):
        if step is None:
            for layer in self.layers:
                for _ in range(self.numrepsperlayer):
                    g = layer(g)
        else:
            # assert(step is not None)
            _step = step/self.numrepsperlayer
            _step = math.floor(_step)
            _step = min(_step, len(self.layers) - 1)
            layer = self.layers[_step]
            g = layer(g, step=None)
        return g


class FFN(torch.nn.Module):
    GATE_BIAS = -2.5
    def __init__(self, hdim, indim=None, zdim=None, use_gate=False, dropout=0., act_dropout=0., **kw):
        super(FFN, self).__init__(**kw)
        self.hdim = hdim
        self.indim = indim if indim is not None else self.hdim
        self.zdim = zdim if zdim is not None else self.hdim
        self.use_gate = use_gate

        self.ln_inp = torch.nn.LayerNorm(self.indim)
        # self.ln_att = torch.nn.LayerNorm(self.hdim)
        # self.ln_final = torch.nn.LayerNorm(self.hdim)

        self.dropout = torch.nn.Dropout(dropout)
        self.act_dropout = torch.nn.Dropout(act_dropout)

        self.act_fn = torch.nn.CELU()

        self.fc1 = torch.nn.Linear(self.indim, self.zdim)
        if self.use_gate:
            self.fc2 = torch.nn.Linear(self.zdim, self.hdim * 2)
            self.fc2.bias.data[self.hdim:] = self.GATE_BIAS
            # self.fcg = torch.nn.Linear(self.zdim, self.hdim)
            # self.fcg.bias.data.fill_(-2.5)
        else:
            self.fc2 = torch.nn.Linear(self.zdim, self.hdim)

    def forward(self, x):
        z = x
        z = self.ln_inp(z)
        z = self.fc1(z)
        z = self.act_fn(z)
        z = self.act_dropout(z)
        z = self.fc2(z)
        if self.use_gate:
            x, r = torch.chunk(z, 2, -1)
            x = torch.sigmoid(r) * self.dropout(x)
        else:
            x = self.dropout(z)
        return x


class LRTMCell(torch.nn.Module):   # Light Relational Transformer
    def __init__(self, hdim, numrels=1, numheads=4, dropout=0., attn_dropout=0., act_dropout=0.,
                 norel=False, use_gate=True, use_sgru=False, skipatt=False, **kw):
        super(LRTMCell, self).__init__(**kw)
        self.hdim = hdim
        self.rdim = self.hdim
        self.zdim = self.hdim
        self.numrels = numrels
        self.norel = norel
        self.skipatt = skipatt
        self.use_gate = use_gate

        self.attention = MultiHeadAttention(self.hdim, self.hdim,
                                            self.hdim, self.hdim, numheads=numheads, use_layernorm=False,
                                            dropout=0., attn_dropout=attn_dropout,
                                            usevallin=False)
        # self.attention_lin = torch.nn.Linear(self.hdim, self.hdim)

        self.use_sgru = use_sgru
        if self.use_sgru:
            self.sgru = DGRUCell(self.hdim, dropout=dropout)
        else:
            self.update_fn = FFN(self.hdim, dropout=dropout, act_dropout=act_dropout, use_gate=self.use_gate)

        if not self.norel:
            # self.msg_ln = torch.nn.LayerNorm(self.hdim, elementwise_affine=False)
            self.relvectors = torch.nn.Embedding(self.numrels, self.hdim*3)
            torch.nn.init.kaiming_uniform_(self.relvectors.weight.data[:, :self.hdim], a=math.sqrt(5))
            torch.nn.init.ones_(self.relvectors.weight.data[:, self.hdim:self.hdim*2])
            torch.nn.init.zeros_(self.relvectors.weight.data[:, self.hdim*2:])

            # message function network from ResRGAT
            self.msg_fn = FFN(indim=self.hdim + self.rdim, hdim=self.hdim, use_gate=self.use_gate, dropout=dropout, act_dropout=act_dropout)

    def reset_dropout(self):
        pass

    def init_node_states(self, g, batsize, device):
        g.ndata["red"] = torch.zeros(batsize, self.hdim, device=device)

    def message_func(self, edges):
        hs = edges.src["h"]
        if self.norel:
            relvecs = torch.zeros(hs.size(0), self.rdim)
        elif "emb" in edges.data:             # if "emb" in edata, then use those
            relvecs = edges.data["emb"]
        else:
            relvecs = self.relvectors(edges.data["id"])
        relvecs = relvecs[:, :self.hdim]
        inps = [hs, relvecs]

        # residual update
        x = torch.cat(inps, -1)
        _x = self.msg_fn(x)
        hs = hs + _x
        msg = hs

        return {"msg": msg, "hs": hs}

    def simple_message_func(self, edges):
        hs = edges.src["h"]
        if self.norel:
            return {"msg": hs, "hs": hs}
        if self.norel:
            relvecs = torch.ones(hs.size(0), self.rdim)
            relvecs2 = torch.zeros(hs.size(0), self.rdim)
            relvecs_add = torch.ones(hs.size(0), self.rdim)
        else:
            if "emb" in edges.data:  # if "emb" in edata, then use those
                relvecs = edges.data["emb"]
            else:
                relvecs = self.relvectors(edges.data["id"])
            relvecs_add, relvecs, relvecs2 = torch.chunk(relvecs, 3, -1)

        _hs = hs
        hs = hs + relvecs_add
        hs = self.msg_ln(hs)
        hs = hs * relvecs + relvecs2
        # hs = hs + relvecs2
        hs = torch.nn.functional.leaky_relu(hs, 0.25)

        hs = _hs + hs       # residual

        msg = hs
        return {"msg": msg, "hs": hs}

    def reduce_func(self, nodes):  # there were no dropouts at all in here !!!
        queries = nodes.data["h"]
        keys, values = nodes.mailbox["msg"], nodes.mailbox["hs"]
        red = self.attention(queries, keys, values)
        return {"red": red}

    def apply_node_func(self, nodes):
        if self.use_sgru:
            h = self.sgru(nodes.data["h"], nodes.data["red"])
        else:
            h = nodes.data["h"]
            summ = nodes.data["red"]

            if self.skipatt:
                summ = summ + h
                summ = self.ln_att(summ)

            z = summ
            x = self.update_fn(z)
            h = h + x
            # h = summ + x

        return {"h": h}

    def forward(self, g, step=0):
        g.update_all(self.message_func, self.reduce_func, self.apply_node_func)
        return g


class LessOldLRTMCell(torch.nn.Module):   # Light Relational Transformer
    def __init__(self, hdim, numrels=1, numheads=4, dropout=0., attn_dropout=0., act_dropout=0.,
                 norel=False, use_gate=True, use_sgru=False, skipatt=False, **kw):
        super(LessOldLRTMCell, self).__init__(**kw)
        self.hdim = hdim
        self.rdim = self.hdim
        self.zdim = self.hdim
        self.numrels = numrels
        self.norel = norel
        self.skipatt = skipatt
        self.use_gate = use_gate

        self.attention = MultiHeadAttention(self.hdim, self.hdim,
                                            self.hdim, self.hdim, numheads=numheads, use_layernorm=False,
                                            dropout=0., attn_dropout=attn_dropout,
                                            usevallin=False)
        # self.attention_lin = torch.nn.Linear(self.hdim, self.hdim)

        self.use_sgru = use_sgru
        if self.use_sgru:
            self.sgru = DGRUCell(self.hdim, dropout=dropout)
        else:
            self.ln_att = torch.nn.LayerNorm(self.hdim)
            self.ln_final = torch.nn.LayerNorm(self.hdim)

            self.dropout = torch.nn.Dropout(dropout)
            self.act_dropout = torch.nn.Dropout(act_dropout)

            self.act_fn = torch.nn.CELU()

            self.fc1 = torch.nn.Linear(self.hdim, self.zdim)
            if self.use_gate:
                self.fc2 = torch.nn.Linear(self.zdim, self.hdim*2)
                self.fc2.bias.data[self.hdim:] = -2.5
                # self.fcg = torch.nn.Linear(self.zdim, self.hdim)
                # self.fcg.bias.data.fill_(-2.5)
            else:
                self.fc2 = torch.nn.Linear(self.zdim, self.hdim)

        if not self.norel:
            self.msg_ln = torch.nn.LayerNorm(self.hdim, elementwise_affine=False)
            self.relvectors = torch.nn.Embedding(self.numrels, self.hdim*3)
            torch.nn.init.kaiming_uniform_(self.relvectors.weight.data[:, :self.hdim], a=math.sqrt(5))
            torch.nn.init.ones_(self.relvectors.weight.data[:, self.hdim:self.hdim*2])
            torch.nn.init.zeros_(self.relvectors.weight.data[:, self.hdim*2:])

            # message function network from ResRGAT
            self.activation = torch.nn.CELU()

            self.linA = torch.nn.Linear(self.hdim + self.hdim, self.zdim)
            self.linB = torch.nn.Linear(self.zdim, self.hdim)

            self.ln1 = torch.nn.LayerNorm(self.hdim + self.hdim)
            self.ln2 = torch.nn.LayerNorm(self.hdim)

    def reset_dropout(self):
        pass

    def init_node_states(self, g, batsize, device):
        g.ndata["red"] = torch.zeros(batsize, self.hdim, device=device)

    def message_func(self, edges):
        hs = edges.src["h"]
        if self.norel:
            relvecs = torch.zeros(hs.size(0), self.rdim)
        elif "emb" in edges.data:             # if "emb" in edata, then use those
            relvecs = edges.data["emb"]
        else:
            relvecs = self.relvectors(edges.data["id"])
        relvecs = relvecs[:, :self.hdim]
        inps = [hs, relvecs]

        # residual update
        x = torch.cat(inps, -1)
        x = self.ln1(x)
        x = self.linA(x)
        x = self.activation(x)
        # x = self.dropout_act(x)
        _x = self.linB(x)
        # _x = self.ln2(_x)
        _x = self.dropout(_x)
        hs = hs + _x
        msg = hs

        return {"msg": msg, "hs": hs}

    def simple_message_func(self, edges):
        hs = edges.src["h"]
        if self.norel:
            return {"msg": hs, "hs": hs}
        if self.norel:
            relvecs = torch.ones(hs.size(0), self.rdim)
            relvecs2 = torch.zeros(hs.size(0), self.rdim)
            relvecs_add = torch.ones(hs.size(0), self.rdim)
        else:
            if "emb" in edges.data:  # if "emb" in edata, then use those
                relvecs = edges.data["emb"]
            else:
                relvecs = self.relvectors(edges.data["id"])
            relvecs_add, relvecs, relvecs2 = torch.chunk(relvecs, 3, -1)

        _hs = hs
        hs = hs + relvecs_add
        hs = self.msg_ln(hs)
        hs = hs * relvecs + relvecs2
        # hs = hs + relvecs2
        hs = torch.nn.functional.leaky_relu(hs, 0.25)

        hs = _hs + hs       # residual

        msg = hs
        return {"msg": msg, "hs": hs}

    def reduce_func(self, nodes):  # there were no dropouts at all in here !!!
        queries = nodes.data["h"]
        keys, values = nodes.mailbox["msg"], nodes.mailbox["hs"]
        red = self.attention(queries, keys, values)
        return {"red": red}

    def apply_node_func(self, nodes):
        if self.use_sgru:
            h = self.sgru(nodes.data["h"], nodes.data["red"])
        else:
            h = nodes.data["h"]
            summ = nodes.data["red"]

            if self.skipatt:
                summ = summ + h
                summ = self.ln_att(summ)

            z = summ
            z = self.ln_final(z)
            z = self.fc1(z)
            z = self.act_fn(z)
            z = self.act_dropout(z)
            z = self.fc2(z)
            if self.use_gate:
                x, r = torch.chunk(z, 2, -1)
                x = torch.sigmoid(r) * self.dropout(x)
            else:
                x = self.dropout(z)
            # h = h + x
            h = summ + x

        return {"h": h}

    def forward(self, g, step=0):
        g.update_all(self.message_func, self.reduce_func, self.apply_node_func)
        return g


class OldLRTMCell(torch.nn.Module):   # Light Relational Transformer
    def __init__(self, hdim, numrels=1, numheads=4, dropout=0., attn_dropout=0., act_dropout=0.,
                 norel=False, use_gate=True, **kw):
        super(LRTMCell, self).__init__(**kw)
        self.hdim = hdim
        self.rdim = self.hdim
        self.zdim = self.hdim * 2
        self.numrels = numrels
        self.norel = norel
        self.skipatt = False
        self.use_gate = use_gate

        self.attention = MultiHeadAttention(self.hdim, self.hdim,
                                            self.hdim, self.hdim, numheads=numheads, use_layernorm=False,
                                            dropout=0., attn_dropout=attn_dropout,
                                            usevallin=False, residual_vallin=False)
        # self.attention_lin = torch.nn.Linear(self.hdim, self.hdim)
        self.ln_att = torch.nn.LayerNorm(self.hdim)
        self.ln_final = torch.nn.LayerNorm(self.hdim)

        self.dropout = torch.nn.Dropout(dropout)
        self.act_dropout = torch.nn.Dropout(act_dropout)

        self.act_fn = torch.nn.CELU()

        self.fc1 = torch.nn.Linear(self.hdim, self.zdim)
        self.fc2 = torch.nn.Linear(self.zdim, self.hdim)
        if self.use_gate:
            self.fcg = torch.nn.Linear(self.zdim, self.hdim)
            self.fcg.bias.data.fill_(-2.5)

        if not self.norel:
            self.relvectors_add = torch.nn.Parameter(torch.randn(self.numrels, self.rdim))
            torch.nn.init.kaiming_uniform_(self.relvectors_add, a=math.sqrt(5))
            self.msg_ln = torch.nn.LayerNorm(self.hdim, elementwise_affine=True)
            self.relvectors = torch.nn.Embedding(self.numrels, self.hdim)
            self.relvectors2 = torch.nn.Embedding(self.numrels, self.hdim)
            torch.nn.init.ones_(self.relvectors.weight)
            torch.nn.init.zeros_(self.relvectors2.weight)

    def reset_dropout(self):
        pass

    def init_node_states(self, g, batsize, device):
        g.ndata["red"] = torch.zeros(batsize, self.hdim, device=device)

    def message_func(self, edges):
        hs = edges.src["h"]
        if self.norel:
            return {"msg": hs, "hs": hs}
        relvecs_add = None
        relvecs = None
        relvecs2 = None
        if self.norel:
            relvecs = torch.ones(hs.size(0), self.rdim)
            relvecs2 = torch.zeros(hs.size(0), self.rdim)
            relvecs_add = torch.ones(hs.size(0), self.rdim)
        elif "emb" in edges.data:  # if "emb" in edata, then use those
            relvecs = edges.data["emb"]
            if "emb2" in edges.data:
                relvecs2 = edges.data["emb2"]
            if "emb_add" in edges.data:
                relvecs_add = edges.data["emb_add"]
        if relvecs is None:
            relvecs = self.relvectors(edges.data["id"])
        if relvecs2 is None:
            relvecs2 = self.relvectors2(edges.data["id"])
        if relvecs_add is None:
            relvecs_add = self.relvectors_add[edges.data["id"]]

        hs = hs + relvecs_add
        hs = self.msg_ln(hs)
        # hs = hs * relvecs + relvecs2
        hs = hs + relvecs2
        # hs = torch.nn.functional.leaky_relu(hs, 0.25)

        msg = hs
        return {"msg": msg, "hs": hs}

    def reduce_func(self, nodes):  # there were no dropouts at all in here !!!
        queries = nodes.data["h"]
        keys, values = nodes.mailbox["msg"], nodes.mailbox["hs"]
        red = self.attention(queries, keys, values)
        return {"red": red}

    def apply_node_func(self, nodes):
        h = nodes.data["h"]
        summ = nodes.data["red"]

        if self.skipatt:
            h = h + summ
            h = self.ln_att(h)
        else:
            h = summ

        z = self.fc1(h)
        z = self.act_fn(z)
        z = self.act_dropout(z)
        x = self.fc2(z)
        x = self.dropout(x)
        if self.use_gate:
            r = self.fcg(self.dropout(z))
            x = torch.sigmoid(r) * x
        h = h + x
        h = self.ln_final(h)

        return {"h": h}

    def forward(self, g, step=0):
        g.update_all(self.message_func, self.reduce_func, self.apply_node_func)
        return g


class RelTransformer(torch.nn.Module):
    def __init__(self, hdim, numlayers=1, numrepsperlayer=1, numrels=1, numheads=4, dropout=0.,
                 rdim=None, relmode="default", norel=False, innercell="default", **kw):
        super(RelTransformer, self).__init__(**kw)
        self.hdim = hdim
        self.norel = norel
        self.layers = torch.nn.ModuleList([
            RelTransformerCell(hdim, numrels=numrels, numheads=numheads, dropout=dropout,
                      rdim=rdim, relmode=relmode, norel=norel, innercell=innercell, **kw)
            for _ in range(numlayers)
        ])
        self.numrepsperlayer = numrepsperlayer

    def reset_dropout(self):
        for layer in self.layers:
            layer.reset_dropout()

    def init_node_states(self, g, batsize, device):
        self.layers[0].init_node_states(g, batsize, device)

    def forward(self, g, step=None):
        if step is None:
            for layer in self.layers:
                for _ in range(self.numrepsperlayer):
                    g = layer(g)
        else:
            # assert(step is not None)
            _step = step/self.numrepsperlayer
            _step = math.floor(_step)
            _step = min(_step, len(self.layers) - 1)
            layer = self.layers[_step]
            g = layer(g, step=None)
        return g


class RelTransformerCell(torch.nn.Module):   # same as SGGNN but without all the ablations
    def __init__(self, hdim, numrels=1, numheads=4, dropout=0., attn_dropout=0., act_dropout=0.,
                 rdim=None, relmode="default", norel=False, innercell="default", **kw):
        super(RelTransformerCell, self).__init__(**kw)
        self.hdim = hdim
        self.rdim = self.hdim if rdim is None else rdim
        self.zdim = self.hdim * 2
        self.numrels = numrels
        self.norel = norel

        self.attention = MultiHeadAttention(self.hdim, self.hdim,
                                            self.hdim, self.hdim, numheads=numheads, use_layernorm=False,
                                            dropout=0., attn_dropout=attn_dropout, usevallin=innercell != "light")
        self.attention_lin = torch.nn.Linear(self.hdim, self.hdim)
        self.ln_att = torch.nn.LayerNorm(self.hdim)
        self.ln_final = torch.nn.LayerNorm(self.hdim)

        self.dropout = torch.nn.Dropout(dropout)
        self.act_dropout = torch.nn.Dropout(act_dropout)

        self.act_fn = torch.nn.CELU()

        self.fc1 = torch.nn.Linear(self.hdim, self.zdim)
        self.fc2 = torch.nn.Linear(self.zdim, self.hdim)

        self.innercell = innercell

        self.relmode = relmode
        if not norel:
            if self.relmode == "addrelu" or self.relmode == "addmul" or self.relmode == "default":
                self.relvectors = torch.nn.Parameter(torch.randn(numrels, self.rdim))
                self.relvectors2 = torch.nn.Parameter(torch.randn(numrels, self.rdim))
                init.kaiming_uniform_(self.relvectors, a=math.sqrt(5))
                init.kaiming_uniform_(self.relvectors2, a=math.sqrt(5))
                self.relvectors2.data = self.relvectors2.data - 3
                self.lrelu = torch.nn.LeakyReLU(0.25)
            elif self.relmode == "rescatmap" or self.relmode == "residual":
                indim = self.hdim + (self.hdim if self.cat_tgt else 0) + (self.rdim if True else 0)
                self.msg_block = ResidualEdgeUpdate(indim, self.hdim, zdim=self.zdim, dropout=dropout, nodamper=nodamper)
                self.relvectors = torch.nn.Parameter(torch.randn(numrels, self.rdim))
                init.kaiming_uniform_(self.relvectors, a=math.sqrt(5))
            elif self.relmode == "baseline":
                self.relproj_in = torch.nn.Linear(self.hdim, self.rdim, bias=False) if self.rdim != self.hdim else None
                self.relproj_out = torch.nn.Linear(self.rdim, self.hdim, bias=False) if self.rdim != self.hdim else None
                self.relmats = torch.nn.Parameter(torch.randn(numrels, self.rdim, self.rdim))
                # init.uniform_(self.relmats, -0.1, 0.1)
                # init.xavier_uniform_(self.relmats, gain=torch.nn.init.calculate_gain("relu"))
                xavier_uniform_for_relmats_(self.relmats, gain=RELMATINITMULT*torch.nn.init.calculate_gain('relu'))
                self.cat_rel = False
            else:
                raise Exception(f"unknown relmode '{relmode}'")

    def reset_dropout(self):
        pass

    def init_node_states(self, g, batsize, device):
        g.ndata["red"] = torch.zeros(batsize, self.hdim, device=device)

    def message_func(self, edges):
        hs = edges.src["h"]
        if self.norel:
            return {"msg": hs, "hs": hs}
        # msg = self.rellin(msg)

        if self.relmode in ("rescatmap", "residual"):
            if "emb" in edges.data:             # if "emb" in edata, then use those
                relvecs = edges.data["emb"]
            else:
                relvecs = self.relvectors[edges.data["id"]]
            inps = [hs, relvecs]
            hs = self.msg_block(*inps)
            msg = hs

        elif self.relmode in ("addrelu", "default"):
            if "emb" in edges.data:             # if "emb" in edata, then use those
                relvecs = edges.data["emb"]
            else:
                relvecs = self.relvectors[edges.data["id"]]
            hs = hs + relvecs
            hs = self.lrelu(hs)
            msg = hs

        elif self.relmode == "addmul":
            if "emb" in edges.data:             # if "emb" in edata, then use those
                relvecs = edges.data["emb"]
            else:
                relvecs = self.relvectors[edges.data["id"]]
            relvecs2 = torch.sigmoid(self.relvectors2[edges.data["id"]])
            hs = hs * (1 - relvecs2) + relvecs * relvecs2
            msg = hs

        elif self.relmode == "baseline":
            relmats = self.relmats[edges.data["id"]]
            _hs = self.relproj_in(hs) if self.relproj_in is not None else hs
            _hs = torch.einsum("bh,bhd->bd", _hs, relmats)
            if self.relproj_out is not None:
                _hs = self.relproj_out(_hs)
                msg = hs + _hs
            else:
                msg = _hs

        return {"msg": msg, "hs": hs}

    def reduce_func(self, nodes):  # there were no dropouts at all in here !!!
        queries = nodes.data["h"]
        keys, values = nodes.mailbox["msg"], nodes.mailbox["hs"]
        red = self.attention(queries, keys, values)
        return {"red": red}

    def apply_node_func(self, nodes):
        h = nodes.data["h"]
        summ = nodes.data["red"]
        if self.innercell != "light":
            summ = self.attention_lin(summ)
            summ = self.dropout(summ)
            h = summ + h
            h = self.ln_att(h)

        x = self.fc1(h)
        x = self.act_fn(x)
        x = self.act_dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        h = h + x
        h = self.ln_final(h)

        return {"h": h}

    def forward(self, g, step=0):
        g.update_all(self.message_func, self.reduce_func, self.apply_node_func)
        return g



class SimpleRGAT(torch.nn.Module):
    def __init__(self, hdim, numlayers=1, numrepsperlayer=1, numrels=1, numheads=4, dropout=0.,
                 rdim=None, residual=True, **kw):
        super(SimpleRGAT, self).__init__(**kw)
        self.hdim = hdim
        self.layers = torch.nn.ModuleList([
            SimpleRGATCell(hdim, numrels=numrels, numheads=numheads, dropout=dropout,
                      rdim=rdim, residual=residual, **kw)
            for _ in range(numlayers)
        ])
        self.numrepsperlayer = numrepsperlayer
        self.residual = residual

    def reset_dropout(self):
        for layer in self.layers:
            layer.reset_dropout()

    def init_node_states(self, g, batsize, device):
        self.layers[0].init_node_states(g, batsize, device)

    def forward(self, g, step=None):
        if step is None:
            for layer in self.layers:
                for _ in range(self.numrepsperlayer):
                    g = layer(g)
        else:
            # assert(step is not None)
            _step = step/self.numrepsperlayer
            _step = math.floor(_step)
            _step = min(_step, len(self.layers) - 1)
            layer = self.layers[_step]
            g = layer(g, step=None)
        return g


class SimpleRGATCell(torch.nn.Module):   # same as SGGNN but without all the ablations
    def __init__(self, hdim, numrels=1, numheads=4, dropout=0., attn_dropout=0., act_dropout=0.,
                 rdim=None, residual=True, **kw):
        super(SimpleRGATCell, self).__init__(**kw)
        self.hdim = hdim
        self.rdim = self.hdim if rdim is None else rdim
        self.zdim = self.hdim * 2
        self.numrels = numrels

        self.attention = MultiHeadAttention(self.hdim, self.hdim,
                                            self.hdim, self.hdim, numheads=numheads, use_layernorm=False,
                                            dropout=0., attn_dropout=attn_dropout, usevallin=True)
        self.attention_lin = torch.nn.Linear(self.hdim, self.hdim)
        self.ln_att = torch.nn.LayerNorm(self.hdim)
        self.ln_final = torch.nn.LayerNorm(self.hdim)

        self.dropout = torch.nn.Dropout(dropout)
        self.act_dropout = torch.nn.Dropout(act_dropout)

        self.act_fn = torch.nn.CELU()

        self.fc1 = torch.nn.Linear(self.hdim, self.zdim)
        self.fc2 = torch.nn.Linear(self.zdim, self.hdim)

        self.relvectors = torch.nn.Parameter(torch.randn(numrels, self.rdim))
        init.kaiming_uniform_(self.relvectors, a=math.sqrt(5))
        self.lrelu = torch.nn.LeakyReLU(0.25)

        self.residual = residual

    def reset_dropout(self):
        pass

    def init_node_states(self, g, batsize, device):
        g.ndata["red"] = torch.zeros(batsize, self.hdim, device=device)

    def message_func(self, edges):
        hs = edges.src["h"]

        if "emb" in edges.data:             # if "emb" in edata, then use those
            relvecs = edges.data["emb"]
        else:
            relvecs = self.relvectors[edges.data["id"]]
        hs = hs + relvecs
        hs = self.lrelu(hs)
        msg = hs
        return {"msg": msg, "hs": hs}

    def reduce_func(self, nodes):  # there were no dropouts at all in here !!!
        queries = nodes.data["h"]
        keys, values = nodes.mailbox["msg"], nodes.mailbox["hs"]
        red = self.attention(queries, keys, values)
        return {"red": red}

    def apply_node_func(self, nodes):
        h = nodes.data["h"]
        summ = nodes.data["red"]
        x = summ
        x = self.act_fn(x)
        x = self.dropout(x)
        if self.residual:
            h = h + x
        else:
            h = x
        return {"h": h}

    def forward(self, g, step=0):
        g.update_all(self.message_func, self.reduce_func, self.apply_node_func)
        return g


class GRGAT(torch.nn.Module):
    def __init__(self, hdim, numlayers=1, numrepsperlayer=1, numrels=1, numheads=4, dropout=0.,
                 dropout_red=0., dropout_attn=0., rdim=None, usevallin=False, relmode="default", norel=False,
                 cell="none", cat_rel=True, cat_tgt=False, usegradskip=False, noattention=False,
                 skipatt=False, nodamper=False, **kw):
        super(GRGAT, self).__init__(**kw)
        self.hdim = hdim
        self.norel = norel
        self.layers = torch.nn.ModuleList([
            GRGATCell(hdim, numrels=numrels, numheads=numheads, dropout=dropout, dropout_red=dropout_red, dropout_attn=dropout_attn,
                      rdim=rdim, usevallin=usevallin, relmode=relmode, cell=cell, norel=norel,
                      cat_rel=cat_rel, cat_tgt=cat_tgt, usegradskip=usegradskip, noattention=noattention,
                      skipatt=skipatt, nodamper=nodamper, **kw)
            for _ in range(numlayers)
        ])
        self.numrepsperlayer = numrepsperlayer

    def reset_dropout(self):
        for layer in self.layers:
            layer.reset_dropout()

    def init_node_states(self, g, batsize, device):
        self.layers[0].init_node_states(g, batsize, device)

    def forward(self, g, step=None):
        if step is None:
            for layer in self.layers:
                for _ in range(self.numrepsperlayer):
                    g = layer(g)
        else:
            # assert(step is not None)
            _step = step/self.numrepsperlayer
            _step = math.floor(_step)
            _step = min(_step, len(self.layers) - 1)
            layer = self.layers[_step]
            g = layer(g, step=None)
        return g


class GRGATCell(torch.nn.Module):   # same as SGGNN but without all the ablations
    def __init__(self, hdim, numrels=1, numheads=4, dropout=0.,
                 dropout_red=0., dropout_attn=0., rdim=None, usevallin=False, relmode="default", norel=False,
                 cell="none", cat_rel=True, cat_tgt=False, usegradskip=False, noattention=False,
                 skipatt=False, nodamper=False, aggregator="default", **kw):
        super(GRGATCell, self).__init__(**kw)
        if cell == "resrgat":
            relmode = "residual"
            cell = "none"
            nodamper = True
            skipatt = False
            usevallin = False
        elif cell == "reltransformer":
            relmode = "addrelu"
            cell = "res"
            skipatt = True
            nodamper = True
            usevallin = True
        elif cell == "reltransformer-mm":
            relmode = "baseline"
            cell = "res"
            skipatt = True
            nodamper = True
            usevallin = True
        self.hdim = hdim
        self.rdim = self.hdim if rdim is None else rdim
        self.zdim = self.hdim
        self.numrels = numrels
        self.norel = norel
        self.skipatt = skipatt

        self.cat_rel, self.cat_tgt = cat_rel, cat_tgt

        if cell == "none":
            self.node_cell = None
        elif cell == "relu":
            self.node_cell = torch.nn.ReLU()
        elif cell == "celu":
            self.node_cell = torch.nn.CELU()
        elif cell == "tanh":
            self.node_cell = torch.nn.Tanh()
        elif cell == "gate":
            self.node_cell = GatedCatMap(self.hdim, self.hdim, zdim=self.zdim, dropout=dropout, usegradskip=usegradskip)
        elif cell == "gru":
            self.node_cell = GRUCell(self.hdim, dropout=dropout)
        elif cell == "dgru" or cell == "default":
            self.node_cell = DGRUCell(self.hdim, dropout=dropout, usegradskip=usegradskip)
        elif cell == "res":
            self.node_cell = ResidualNodeUpdate(self.hdim, self.hdim, zdim=self.zdim, dropout=dropout, nodamper=nodamper)
        else:
            raise Exception(f"unknown cell type: {cell}")

        self.relmode = relmode
        if norel:
            self.cat_rel = False
        else:
            if self.relmode == "default" or self.relmode == "gatedcatmap" or self.relmode == "gated":
                indim = self.hdim + (self.hdim if self.cat_tgt else 0) + (self.rdim if self.cat_rel else 0)
                self.msg_block = GatedCatMap(indim, self.hdim, zdim=self.zdim, dropout=dropout)
                self.relvectors = torch.nn.Parameter(torch.randn(numrels, self.rdim))
                init.kaiming_uniform_(self.relvectors, a=math.sqrt(5))
            elif self.relmode == "addrelu" or self.relmode == "addmul":
                self.relvectors = torch.nn.Parameter(torch.randn(numrels, self.rdim))
                self.relvectors2 = torch.nn.Parameter(torch.randn(numrels, self.rdim))
                init.kaiming_uniform_(self.relvectors, a=math.sqrt(5))
                init.kaiming_uniform_(self.relvectors2, a=math.sqrt(5))
                self.relvectors2.data = self.relvectors2.data - 3
                self.lrelu = torch.nn.LeakyReLU(0.1)
            elif self.relmode == "rescatmap" or self.relmode == "residual":
                indim = self.hdim + (self.hdim if self.cat_tgt else 0) + (self.rdim if True else 0)
                self.msg_block = ResidualEdgeUpdate(indim, self.hdim, zdim=self.zdim, dropout=dropout, nodamper=nodamper)
                self.relvectors = torch.nn.Parameter(torch.randn(numrels, self.rdim))
                init.kaiming_uniform_(self.relvectors, a=math.sqrt(5))
            elif self.relmode == "baseline":
                self.relproj_in = torch.nn.Linear(self.hdim, self.rdim, bias=False) if self.rdim != self.hdim else None
                self.relproj_out = torch.nn.Linear(self.rdim, self.hdim, bias=False) if self.rdim != self.hdim else None
                self.relmats = torch.nn.Parameter(torch.randn(numrels, self.rdim, self.rdim))
                # init.uniform_(self.relmats, -0.1, 0.1)
                # init.xavier_uniform_(self.relmats, gain=torch.nn.init.calculate_gain("relu"))
                xavier_uniform_for_relmats_(self.relmats, gain=RELMATINITMULT*torch.nn.init.calculate_gain('relu'))
                self.cat_rel = False
            else:
                raise Exception(f"unknown relmode '{relmode}'")

        if aggregator not in ("default", "attention", "att"):
            noattention = True
        if not noattention:
            self.attention = MultiHeadAttention(self.hdim, (self.hdim + self.rdim) if self.cat_rel else self.hdim,
                                                self.hdim, self.hdim, numheads=numheads, use_layernorm=False,
                                                dropout=0., attn_dropout=dropout_attn, usevallin=usevallin)

        self.dropout = torch.nn.Dropout(dropout)
        self.dropout_red = torch.nn.Dropout(dropout_red)

        # ablations
        self.usevallin = usevallin
        self.noattention = noattention
        if noattention:
            self.cat_rel = False

        self.aggr = aggregator

        self.ln_att = torch.nn.LayerNorm(hdim)

    def reset_dropout(self):
        if hasattr(self.node_cell, "reset_dropout"):
            self.node_cell.reset_dropout()

    def init_node_states(self, g, batsize, device):
        g.ndata["red"] = torch.zeros(batsize, self.hdim, device=device)

    def message_func(self, edges):
        hs = edges.src["h"]
        if self.norel:
            return {"msg": hs, "hs": hs}
        # msg = self.rellin(msg)

        if self.relmode in ("default", "rescatmap", "gatedcatmap", "gated", "residual"):
            if "emb" in edges.data:             # if "emb" in edata, then use those
                relvecs = edges.data["emb"]
            else:
                relvecs = self.relvectors[edges.data["id"]]
            inps = [hs, relvecs]
            if self.cat_tgt:    # False by default
                inps.append(edges.dst["h"])
            hs = self.msg_block(*inps)

            if self.cat_rel:  # True by default
                msg = torch.cat([hs, relvecs], -1)
            else:
                msg = hs

        elif self.relmode == "addrelu":
            if "emb" in edges.data:             # if "emb" in edata, then use those
                relvecs = edges.data["emb"]
            else:
                relvecs = self.relvectors[edges.data["id"]]
            hs = hs + relvecs
            hs = self.lrelu(hs)
            if self.cat_rel:  # True by default
                msg = torch.cat([hs, relvecs], -1)
            else:
                msg = hs

        elif self.relmode == "addmul":
            if "emb" in edges.data:             # if "emb" in edata, then use those
                relvecs = edges.data["emb"]
            else:
                relvecs = self.relvectors[edges.data["id"]]
            relvecs2 = torch.sigmoid(self.relvectors2[edges.data["id"]])
            hs = hs * (1 - relvecs2) + relvecs * relvecs2
            if self.cat_rel:  # True by default
                msg = torch.cat([hs, relvecs], -1)
            else:
                msg = hs

        elif self.relmode == "baseline":
            relmats = self.relmats[edges.data["id"]]
            _hs = self.relproj_in(hs) if self.relproj_in is not None else hs
            _hs = torch.einsum("bh,bhd->bd", _hs, relmats)
            if self.relproj_out is not None:
                _hs = self.relproj_out(_hs)
                msg = hs + _hs
            else:
                msg = _hs

        return {"msg": msg, "hs": hs}

    def reduce_func(self, nodes):  # there were no dropouts at all in here !!!
        if self.noattention:
            red = nodes.mailbox["hs"].mean(1)
        elif self.aggr == "max":
            red, _ = nodes.mailbox["hs"].max(1)
        else:
            queries = nodes.data["h"]
            keys, values = nodes.mailbox["msg"], nodes.mailbox["hs"]
            red = self.attention(queries, keys, values)
            if self.skipatt:
                red = red + nodes.data["h"]
                red = self.ln_att(red)
        return {"red": red}

    def apply_node_func(self, nodes):
        if self.node_cell is None:
            h = nodes.data["red"]
        elif not isinstance(self.node_cell, (DGRUCell, GRUCell, ResidualNodeUpdate)):
            h = self.node_cell(nodes.data["red"])
        else:
            h = self.node_cell(nodes.data["red"],  # * self.dropout_mask[None, :].clamp_max(1),
                               nodes.data["h"])  # * self.dropout_mask[None, :].clamp_max(1))
        return {"h": h}

    def forward(self, g, step=0):
        g.update_all(self.message_func, self.reduce_func, self.apply_node_func)
        return g


def kaiming_uniform_(tensor, a=0, fan=-1, nonlinearity='leaky_relu'):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only
        used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
    """
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


class GraphLSTMGNNCell(torch.nn.Module):
    def __init__(self, hdim, numrels=5, dropout=0., **kw):
        super(GraphLSTMGNNCell, self).__init__(**kw)
        self.Wi = torch.nn.Linear(hdim, hdim, bias=True)
        self.Wf = torch.nn.Linear(hdim, hdim, bias=True)
        self.Wo = torch.nn.Linear(hdim, hdim, bias=True)
        self.Wc = torch.nn.Linear(hdim, hdim, bias=True)
        self.Ui = torch.nn.Parameter(torch.zeros(numrels, hdim, hdim))
        self.Uf = torch.nn.Parameter(torch.zeros(numrels, hdim, hdim))
        self.Uo = torch.nn.Parameter(torch.zeros(numrels, hdim, hdim))
        self.Uc = torch.nn.Parameter(torch.zeros(numrels, hdim, hdim))
        self.hdim = hdim

        self.dropout = torch.nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_uniform_(self.Ui, a=math.sqrt(5), fan=self.hdim)
        kaiming_uniform_(self.Uf, a=math.sqrt(5), fan=self.hdim)
        kaiming_uniform_(self.Uo, a=math.sqrt(5), fan=self.hdim)
        kaiming_uniform_(self.Uc, a=math.sqrt(5), fan=self.hdim)

    def reset_dropout(self):
        pass

    def init_node_states(self, g, batsize, device):
        g.ndata["red"] = torch.zeros(batsize, self.hdim, device=device)
        g.ndata["c"] = torch.zeros_like(g.ndata["h"])

    def forward(self, g:dgl.DGLGraph, step=0):
        g.update_all(self.message_func, self.reduce_func, self.apply_node_func)

    def message_func(self, edges):
        src_hs = edges.src["h"]
        tgt_hs = edges.tgt["h"]
        src_cs = edges.src["c"]
        edge_ids = edges.data["id"]

        src_hs = self.dropout(src_hs)
        tgt_hs = self.dropout(tgt_hs)

        edge_Uis = self.Ui.index_select(0, edge_ids)
        edge_Ufs = self.Uf.index_select(0, edge_ids)
        edge_Uos = self.Uo.index_select(0, edge_ids)
        edge_Ucs = self.Uc.index_select(0, edge_ids)

        msg_uis = torch.einsum("bd,dk->bk", src_hs, edge_Uis)
        msg_ufs = torch.einsum("bd,dk->bk", src_hs, edge_Ufs)
        msg_uos = torch.einsum("bd,dk->bk", src_hs, edge_Uos)
        msg_ucs = torch.einsum("bd,dk->bk", src_hs, edge_Ucs)

        msg_f = torch.sigmoid(self.Wf(tgt_hs) + msg_ufs)
        msg_c = src_cs * msg_f
        return {
            "msg_c": msg_c,
            "msg_gi": msg_uis,
            "msg_go": msg_uos,
            "msg_gc": msg_ucs
        }

    def reduce_func(self, nodes):
        msg_c = nodes.mailbox["msg_c"].sum(1)
        msg_gi = nodes.mailbox["msg_gi"].sum(1)
        msg_go = nodes.mailbox["msg_go"].sum(1)
        msg_gc = nodes.mailbox["msg_gc"].sum(1)
        return {
            "msg_c": msg_c,
            "msg_gi": msg_gi,
            "msg_go": msg_go,
            "msg_gc": msg_gc
        }

    def apply_node_func(self, nodes):
        gi = torch.sigmoid(self.Wi(nodes.ndata["h"]) + nodes.ndata["msg_gi"])
        go = torch.sigmoid(self.Wo(nodes.ndata["h"]) + nodes.ndata["msg_go"])
        gc = torch.tanh(self.Wc(nodes.ndata["h"]) + nodes.ndata["msg_gc"])
        c = nodes.ndata["msg_c"] + gi * gc
        h = go * torch.tanh(c)
        return {
            "c": c,
            "h": h
        }


class SGGNNCell(torch.nn.Module):
    dropout_in_reduce = False
    dropout_value_in_reduce = False
    isolated_attention = False
    def __init__(self, hdim, dropout=0., dropout_red=-1, rdim=None, zoneout=0., numrels=1, numheads=4,
                 no_attention=False, usevallin=False, cell="dgru",  # "dgru", "gru", "dsdgru", "sdgru", "sum"
                 relmode="original", gumbelmix=0., windowhead=False, # "original" or "originalmul" or "add" or "mul" or "addmul" or "muladd" or "gru"
                 **kw):
        super(SGGNNCell, self).__init__(**kw)
        self.hdim = hdim
        self.rdim = self.hdim if rdim is None else rdim
        if cell == "gru":
            self.node_cell = GRUCell(self.hdim, dropout_rec=dropout)
        elif cell == "dgru":
            self.node_cell = DGRUCell(self.hdim, dropout_rec=dropout, zoneout=zoneout)
        elif cell == "sdgru":
            self.node_cell = SimpleDGRUCell(self.hdim, dropout_rec=dropout)
        elif cell == "dsdgru":
            self.node_cell = DSDGRUCell(self.hdim, dropout_rec=dropout)
        elif cell == "nigru":
            self.node_cell = NIGRUCell(self.hdim, dropout_rec=dropout)

        self.dualstate = cell=="dsdgru"
        # self.rellin = torch.nn.Linear(self.hdim, self.hdim, bias=False)
        self.relmode = relmode
        if self.rdim != self.hdim:
            assert(self.relmode == "matmul")
        self.relvectors = torch.nn.Parameter(torch.randn(numrels, self.hdim))
        init.kaiming_uniform_(self.relvectors, a=math.sqrt(5))
        # self.relvectors_mul = torch.nn.Parameter(torch.ones(numrels, self.hdim) * 5)
        self.relvectors_mul = torch.nn.Parameter((torch.rand_like(self.relvectors)-0.5)*1. + 3.)
        self.relvectors_mul.data = self.relvectors_mul.data + 3
        if "matmul" in self.relmode:
            self.relproj_in = torch.nn.Linear(self.hdim, self.rdim, bias=False) if self.rdim != self.hdim else None
            self.relproj_out = torch.nn.Linear(self.rdim, self.hdim, bias=False) if self.rdim != self.hdim else None
            self.relmats = torch.nn.Parameter(torch.randn(numrels, self.rdim, self.rdim))
            # init.uniform_(self.relmats, -0.1, 0.1)
            torch.nn.init.xavier_uniform_(self.relmats, gain=torch.nn.init.calculate_gain('relu'))
        # init.kaiming_uniform_(self.relvectors_mul, a=math.sqrt(5))

        if relmode == "gatedcatpremap" or relmode == "gatedcatpostmap":
            self.catblock = GatedCatMapBlock(self.hdim, dropout=0., zoneout=zoneout)
        elif relmode == "catpremap" or relmode == "catpostmap":
            # self.catblock = CatMapBlock(self.hdim)
            self.catblock = SimpleCatMapBlock(self.hdim, dropout=0.)

        self.self_relvector = torch.nn.Parameter(torch.randn(self.hdim))
        init.uniform_(self.self_relvector, -0.1, 0.1)

        self.relgru = torch.nn.GRUCell(self.hdim, self.hdim)
        self.relgru.bias_hh.data[self.hdim:self.hdim*2] = 3
        self.relgru.bias_ih.data[self.hdim:self.hdim*2] = 3

        if windowhead:
            self.attention = WindowHeadAttention(self.hdim, self.hdim * 2, self.hdim, self.hdim, stride=2, windowsize=int(self.hdim/numheads),
                                 dropout=0., attn_dropout=0.,
                                 usevallin=usevallin)
        else:
            MHA = MultiHeadAttention if not self.isolated_attention else MultiHeadAttention_Isolated
            self.attention = MHA(self.hdim, self.hdim * 2, self.hdim, self.hdim, numheads=numheads,
                                 dropout=0., attn_dropout=0.,
                                 usevallin=usevallin, gumbelmix=gumbelmix)

        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer("dropout_mask", torch.ones(self.hdim))
        dropout_red = dropout if dropout_red < 0 else dropout_red
        self.dropout_red = torch.nn.Dropout(dropout_red)
        self.register_buffer("dropout_red_mask", torch.ones(self.hdim))

        # ablations
        self.no_attention = no_attention
        self.usevallin = usevallin

        self.layernorm = torch.nn.LayerNorm(self.hdim)

    def reset_dropout(self):
        if hasattr(self.node_cell, "reset_dropout"):
            self.node_cell.reset_dropout()
        device = self.dropout_mask.device
        ones = torch.ones(self.hdim, device=device)
        self.dropout_mask = self.dropout(ones)
        self.dropout_red_mask = self.dropout_red(ones)

    def init_node_states(self, g, batsize, device):
        g.ndata["red"] = torch.zeros(batsize, self.hdim, device=device)
        if self.dualstate:
            g.ndata["c"] = g.ndata["h"]

    def message_func(self, edges):
        hs = edges.src["h"]
        # msg = self.rellin(msg)
        relvecs = self.relvectors[edges.data["id"]]
        relvecs_mul = self.relvectors_mul[edges.data["id"]]
        if self.relmode in ("catpremap", "gatedcatpremap"):
            hs = self.catblock(hs, relvecs)
        # if self.relmode == "matmul":
        elif self.relmode == "matmul":
            relmats = self.relmats[edges.data["id"]]
            _hs = self.relproj_in(hs) if self.relproj_in is not None else hs
            _hs = torch.einsum("bh,bhd->bd", _hs, relmats)
            if self.relproj_out is not None:
                _hs = self.relproj_out(_hs)
                hs = _hs + hs
            else:
                hs = _hs
        if self.relmode == "matmulskip":
            relmats = self.relmats[edges.data["id"]]
            _hs = self.relproj_in(hs) if self.relproj_in is not None else hs
            _hs = torch.einsum("bh,bhd->bd", _hs, relmats)
            _hs = self.relproj_out(_hs) if self.relproj_out is not None else _hs
            hs = _hs + hs
        elif self.relmode == "add":
            hs = hs + relvecs
        elif self.relmode == "addmul":
            hs = hs + relvecs
            hs = hs * torch.sigmoid(relvecs_mul)
        elif self.relmode == "gru":
            hs = self.relgru(relvecs, hs)
        elif self.relmode == "original" or self.relmode == "originalmul":
            hs = hs + relvecs
        elif self.relmode == "lrelu":   # might want to move leaky relu and prelu back here from before the cat (prelu seemed to work ok here)
            hs = hs + relvecs
        elif self.relmode == "prelu":
            hs = hs + relvecs
            hs = torch.max(hs, torch.zeros_like(hs)) \
                 + torch.sigmoid(relvecs_mul) * torch.min(hs, torch.zeros_like(hs))
            # hs = torch.nn.functional.prelu(hs + relvecs, relvecs_add)
        elif self.relmode == "mix":
            hs = hs * torch.sigmoid(relvecs_mul) + relvecs * (1-torch.sigmoid(relvecs_mul))
        else:
            pass

        if self.no_attention:
            msg = hs
        else:
            msg = torch.cat([hs, relvecs], -1)

        if self.relmode in ("catpostmap", "gatedcatpostmap"):
            hs = self.catblock(hs, relvecs)
        # if self.relmode == "originalmul":
        elif self.relmode == "originalmul":
            hs = hs * torch.sigmoid(relvecs_mul)
        elif self.relmode == "lrelu":
            hs = torch.nn.functional.leaky_relu(hs, .2)
        elif self.relmode == "mul":
            hs = hs * torch.sigmoid(relvecs_mul)
        elif self.relmode == "addmul":
            hs = (hs + relvecs) * torch.sigmoid(relvecs_mul)
        return {"msg": msg, "hs": hs}

    def reduce_func(self, nodes):       # there were no dropouts at all in here !!!
        if self.no_attention:
            red = nodes.mailbox["hs"].mean(1)
            if self.dropout_value_in_reduce:
                red = red * self.dropout_red_mask[None, :]#.clamp_max(1.)
        else:
            if self.dualstate:
                queries = nodes.data["c"]
            else:
                queries = nodes.data["h"]
            keys, values = nodes.mailbox["msg"], nodes.mailbox["hs"]
            if self.dropout_in_reduce:
                queries = queries * self.dropout_red_mask[None, :]
                keys = keys * torch.cat([self.dropout_red_mask[None, None, :], self.dropout_red_mask[None, None, :]], -1)
                # if self.usevallin:
                #     pass
                #     # values = self.dropout_red(values)
                #
                # else:
                #     values = values * self.dropout_mask[None, None, :].clamp_max(1)
            if self.dropout_value_in_reduce:
                values = values * self.dropout_mask[None, None, :]#.clamp_max(1)
            red = self.attention(queries, keys, values)
        return {"red": red}

    def apply_node_func(self, nodes):
        # if self.self_in_att:
        #     h = self.node_cell(nodes.data["red"])
        #     # h = self.layernorm(h)
        #     return {"h": h}
        if self.dualstate:
            h, c = self.node_cell(nodes.data["red"],# * self.dropout_mask[None, :].clamp_max(1),
                                  nodes.data["c"]) # * self.dropout_mask[None, :].clamp_max(1))
            # h = self.layernorm(h)
            return {"h": h, "c": c}
        else:
            h = self.node_cell(nodes.data["red"], # * self.dropout_mask[None, :].clamp_max(1),
                               nodes.data["h"])# * self.dropout_mask[None, :].clamp_max(1))
            # h = self.layernorm(h)
            return {"h": h}

    def forward(self, g, step=0):
        g.update_all(self.message_func, self.reduce_func, self.apply_node_func)


"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""


class MLPReadout(torch.nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [torch.nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(torch.nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = torch.nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = torch.relu(y)
        y = self.FC_layers[self.L](y)
        return y


class GatedGCNNet(torch.nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        emb_dim = net_params["emb_dim"]
        hidden_dim = net_params['hidden_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.pos_enc = net_params['pos_enc']
        self.shared = net_params["shared"] if "shared" in net_params else False
        self.hdim = hidden_dim
        if self.pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_pos_enc = torch.nn.Linear(pos_enc_dim, hidden_dim)

        self.embedding_h = torch.nn.Embedding(num_atom_type, emb_dim) #hidden_dim)
        self.emb_adapter = torch.nn.Linear(emb_dim, hidden_dim) if emb_dim != hidden_dim else Identity()

        if self.edge_feat:
            self.embedding_e = torch.nn.Embedding(num_bond_type, hidden_dim)
        else:
            self.embedding_e = torch.nn.Linear(1, hidden_dim)

        self.in_feat_dropout = torch.nn.Dropout(in_feat_dropout)

        self.layers = torch.nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                   self.batch_norm, self.residual) for _ in range(n_layers)])
        self.numrepsperlayer = 1

        if self.shared:
            self.numrepsperlayer = len(self.layers)
            self.layers = torch.nn.ModuleList([self.layers[0]])

        self.MLP_layer = MLPReadout(hidden_dim, 1)  # 1 out dim since regression problem

    def init_node_states(self, g, batsize, device):
        pass

    def forward(self, g, h, e, h_pos_enc=None):

        # input embedding
        h = self.embedding_h(h)
        h = self.emb_adapter(h)
        h = self.in_feat_dropout(h)
        if self.pos_enc:
            h_pos_enc = self.embedding_pos_enc(h_pos_enc.float())
            h = h + h_pos_enc
        if not self.edge_feat:  # edge feature set to 1
            e = torch.ones(e.size(0), 1).to(self.device)
        e = self.embedding_e(e)

        # convnets
        for conv in self.layers:
            for _ in range(self.numrepsperlayer):
                h, e = conv(g, h, e)
        g.ndata['h'] = h

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        elif self.readout == "none":
            pass
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
        if self.readout == "none":
            return g.ndata["h"]
        else:
            return self.MLP_layer(hg)

    def loss(self, scores, targets):
        # loss = torch.nn.MSELoss()(scores,targets)
        loss = torch.nn.L1Loss()(scores, targets)
        return loss


class GatedGCNLayer(torch.nn.Module):
    """
        Param: []
    """

    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual

        if input_dim != output_dim:
            self.residual = False

        self.A = torch.nn.Linear(input_dim, output_dim, bias=True)
        self.B = torch.nn.Linear(input_dim, output_dim, bias=True)
        self.C = torch.nn.Linear(input_dim, output_dim, bias=True)
        self.D = torch.nn.Linear(input_dim, output_dim, bias=True)
        self.E = torch.nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = torch.nn.BatchNorm1d(output_dim)
        self.bn_node_e = torch.nn.BatchNorm1d(output_dim)
        # self.bn_node_h = torch.nn.LayerNorm(output_dim)
        # self.bn_node_e = torch.nn.LayerNorm(output_dim)

    def message_func(self, edges):
        Bh_j = edges.src['Bh']
        e_ij = edges.data['Ce'] + edges.src['Dh'] + edges.dst['Eh']  # e_ij = Ce_ij + Dhi + Ehj
        edges.data['e'] = e_ij
        return {'Bh_j': Bh_j, 'e_ij': e_ij}

    def reduce_func(self, nodes):
        Ah_i = nodes.data['Ah']
        Bh_j = nodes.mailbox['Bh_j']
        e = nodes.mailbox['e_ij']
        sigma_ij = torch.sigmoid(e)  # sigma_ij = sigmoid(e_ij)
        # h = Ah_i + torch.mean( sigma_ij * Bh_j, dim=1 ) # hi = Ahi + mean_j alpha_ij * Bhj
        h = Ah_i + torch.sum(sigma_ij * Bh_j, dim=1) / (torch.sum(sigma_ij,
                                                                  dim=1) + 1e-6)  # hi = Ahi + sum_j eta_ij/sum_j' eta_ij' * Bhj <= dense attention
        return {'h': h}

    def forward(self, g, h, e):

        h_in = h  # for residual connection
        e_in = e  # for residual connection

        g.ndata['h'] = h
        g.ndata['Ah'] = self.A(h)
        g.ndata['Bh'] = self.B(h)
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h)
        g.edata['e'] = e
        g.edata['Ce'] = self.C(e)
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata['h']  # result of graph convolution
        e = g.edata['e']  # result of graph convolution

        if self.batch_norm:
            h = self.bn_node_h(h)  # batch normalization
            e = self.bn_node_e(e)  # batch normalization

        h = torch.relu(h)  # non-linear activation
        e = torch.relu(e)  # non-linear activation

        if self.residual:
            h = h_in + h  # residual connection
            e = e_in + e  # residual connection

        h = torch.dropout(h, self.dropout, self.training)
        e = torch.dropout(e, self.dropout, self.training)

        return h, e

    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                                            self.in_channels,
                                                            self.out_channels)
