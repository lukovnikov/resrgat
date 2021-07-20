import dgl
import ujson
import os
import re
from typing import Set

import torch
import numpy as np
from dgl import ALL
from dgl.frame import Frame
from torch.autograd import Function

from ldgnn.vocab import Vocab
import qelos as q


class ForwardMixBackwardSum(Function):
    @staticmethod
    def forward(ctx, a, b, mix):
        ctx.save_for_backward(a, b, mix)
        output = a * mix + b * (1 - mix)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        a, b, mix = ctx.saved_tensors
        grad_a = grad_out
        grad_b = grad_out

        grad_mix = grad_out * a - grad_out * b
        return grad_a, grad_b, grad_mix


forward_mix_backward_sum = ForwardMixBackwardSum.apply


class ForwardMixBackwardSumMulti(Function):
    @staticmethod
    def forward(ctx, a, mix):
        ctx.save_for_backward(a, mix)
        output = (a * mix).sum(-1)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        a, mix = ctx.saved_tensors

        reps = [1] * grad_out.dim()
        reps.append(a.size(-1))
        grad_a = grad_out.unsqueeze(-1).repeat(*reps)

        grad_mix = grad_a * a
        return grad_a, grad_mix


forward_mix_backward_sum_multi = ForwardMixBackwardSumMulti.apply


class AttentionDropout(torch.nn.Module):
    def __init__(self, p=0., dim=-1, **kw):
        super(AttentionDropout, self).__init__(**kw)
        self.p = p
        self.dropout = torch.nn.Dropout(self.p)
        self.dim = dim

    def forward(self, weights):
        # prevent from all weights becoming zero
        if self.training and self.p > 0:
            ps = torch.rand_like(weights)
            maxs, _ = ps.max(-1)
            ts = torch.min(maxs, torch.ones_like(maxs) * self.p)
            mask = (ps >= ts.unsqueeze(-1)).float()
            weights = mask * weights
            # mask = self.dropout(torch.ones_like(weights)).clamp_max(1)
            # alldropped = mask.sum(self.dim, keepdim=True) == 0
            # # select one entry at random and use it when alldropped == True
            # addmask = torch.zeros_like(mask)
            # rand_select = torch.randint(0, weights.size(self.dim), alldropped.size(),
            #                             device=weights.device)
            # addmask = addmask.scatter(self.dim, rand_select, 1)
            # mask = torch.where(alldropped, addmask, mask)
        return weights


class MultiHeadAttention(torch.nn.Module):
    GUMBEL_TEMP = 1.
    def __init__(self, querydim, keydim=None, valdim=None, hdim=None,
                 use_layernorm=False, dropout=0., attn_dropout=0., numheads=1, usevallin=True,
                 residual_vallin=False,
                 gumbelmix=0., **kw):
        super(MultiHeadAttention, self).__init__(**kw)
        self.querydim = querydim
        self.hdim = querydim if hdim is None else hdim
        self.valdim = querydim if valdim is None else valdim
        self.keydim = querydim if keydim is None else keydim
        self.usevallin = usevallin
        self.residual_vallin = residual_vallin
        self.numheads = numheads
        assert(self.hdim // self.numheads == self.hdim / self.numheads)

        self.query_lin = torch.nn.Linear(self.querydim, self.hdim, bias=False)
        self.key_lin = torch.nn.Linear(self.keydim, self.hdim, bias=False)
        self.value_lin = torch.nn.Linear(self.valdim, self.hdim, bias=False) if usevallin is True else None

        self.dropout = torch.nn.Dropout(dropout)
        self.attn_dropout = torch.nn.Dropout(attn_dropout)

        if use_layernorm:
            self.ln_query = torch.nn.LayerNorm(querydim)
            self.ln_key = torch.nn.LayerNorm(keydim)
            if self.value_lin is not None:
                self.ln_value = torch.nn.LayerNorm(valdim)
        else:
            self.ln_query = None
            self.ln_key = None
            self.ln_value = None

        self.gumbelmix = gumbelmix

    def forward(self, query, key, val):
        if self.ln_query is not None:
            query = self.ln_query(query)
        if self.ln_key is not None:
            key = self.ln_key(key)
        if self.ln_value is not None:
            val = self.ln_value(val)
        queries = self.query_lin(query)
        context = self.key_lin(key)    # bsd
        queries = queries.view(queries.size(0), self.numheads, -1)  # bhl
        context = context.view(context.size(0), context.size(1), self.numheads, -1).transpose(1, 2)     # bhsl
        weights = torch.einsum("bhd,bhsd->bhs", queries, context) / np.sqrt(context.size(-1))
        alphas = torch.softmax(weights, -1)      # bhs
        if q.v(self.gumbelmix) > 0. and self.training:
            alphas_gumbel = torch.nn.functional.gumbel_softmax(weights, tau=self.GUMBEL_TEMP, hard=False, dim=-1)
            alphas = alphas_gumbel * q.v(self.gumbelmix) + alphas * (1 - q.v(self.gumbelmix))

        alphas = self.attn_dropout(alphas)
        values = val
        if self.value_lin is not None:
            values = self.value_lin(values)
            if self.residual_vallin:
                values = values + val
        values = values.view(values.size(0), values.size(1), self.numheads, -1).transpose(1, 2)
        red = torch.einsum("bhs,bhsd->bhd", alphas, values)
        red = red.view(red.size(0), -1)
        return red


class WindowHeadAttention(torch.nn.Module):
    def __init__(self, querydim, keydim=None, valdim=None, hdim=None, stride=1,
                 dropout=0., attn_dropout=0., windowsize=5, usevallin=True, **kw):
        super(WindowHeadAttention, self).__init__(**kw)
        self.querydim = querydim
        self.hdim = querydim if hdim is None else hdim
        self.valdim = querydim if valdim is None else valdim
        self.keydim = querydim if keydim is None else keydim
        self.usevallin = usevallin
        self.windowsize = windowsize
        assert(self.windowsize < self.hdim)
        self.stride = stride

        self.query_lin = torch.nn.Linear(self.querydim, self.hdim*self.stride, bias=False)
        self.key_lin = torch.nn.Linear(self.keydim, self.hdim*self.stride, bias=False)
        self.value_lin = torch.nn.Linear(self.valdim, self.hdim, bias=False) if usevallin is True else None

        self.dropout = torch.nn.Dropout(dropout)
        self.attn_dropout = torch.nn.Dropout(attn_dropout)

        self.register_buffer("window", torch.ones(self.windowsize))

    def forward(self, query, key, val):
        queries = self.query_lin(self.dropout(query))# bd
        context = self.key_lin(self.dropout(key))    # bsd
        prods = queries[:, None, :] * context       # bsd
        _prods = torch.cat([prods, prods[:, :, :self.windowsize+1]], -1)
        _prods = self.attn_dropout(_prods)
        __prods = _prods.view(-1, _prods.size(-1))[:, None, :]

        weights = torch.conv1d(__prods, self.window[None, None, :], stride=self.stride) \
                  / torch.sqrt(self.window.sum())
        weights = weights[:, 0, :self.hdim]
        weights = weights.view(_prods.size(0), _prods.size(1), self.hdim)
        weights = weights.transpose(1, 2)       # bds

        alphas = torch.softmax(weights, -1)      # bds

        values = val
        if self.value_lin is not None:
            values = self.value_lin(self.dropout(values))
        values = values.transpose(1, 2)     # bds
        red = (values * alphas).sum(-1)     # bd
        return red


class MultiHeadAttention_Isolated(torch.nn.Module):
    def __init__(self, querydim, keydim=None, valdim=None, hdim=None,
                 dropout=0., numheads=1, usevallin=True, **kw):
        super(MultiHeadAttention_Isolated, self).__init__(**kw)
        self.querydim = querydim
        self.hdim = querydim if hdim is None else hdim
        self.valdim = querydim if valdim is None else valdim
        self.keydim = querydim if keydim is None else keydim
        self.usevallin = usevallin
        self.numheads = numheads
        assert(self.hdim // self.numheads == self.hdim / self.numheads)
        assert(self.querydim // self.numheads == self.querydim / self.numheads)
        assert(self.keydim // self.numheads == self.keydim / self.numheads)

        self.query_lin = torch.nn.Parameter(torch.zeros(self.querydim//self.numheads, self.numheads, self.hdim // self.numheads))
        self.key_lin = torch.nn.Parameter(torch.zeros(self.keydim//self.numheads, self.numheads, self.hdim // self.numheads))
        self.value_lin = torch.nn.Parameter(torch.zeros(self.valdim//self.numheads, self.numheads, self.hdim // self.numheads)) \
            if usevallin is True else None

        self.attn_dropout = AttentionDropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.query_lin, a=np.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.key_lin, a=np.sqrt(5))
        if self.value_lin is not None:
            torch.nn.init.kaiming_uniform_(self.value_lin, a=np.sqrt(5))

    def forward(self, query, key, val):
        queries = query.view(query.size(0), self.numheads, -1)          # bhl
        keys = key.view(key.size(0), key.size(1), self.numheads, -1)    # bshl
        queries = torch.einsum("bhl,lhk->bhk", queries, self.query_lin)
        keys = torch.einsum("bshl,lhk->bhsk", keys, self.key_lin)
        weights = torch.einsum("bhk,bhsk->bhs", queries, keys) / np.sqrt(keys.size(-1))
        weights = self.attn_dropout(weights)
        alphas = torch.softmax(weights, -1)      # bhs

        values = val.view(val.size(0), val.size(1), self.numheads, -1).transpose(1, 2)  # bhsl
        if self.value_lin is not None:
            values = torch.einsum("bhsl,lhk->bhsk", values, self.value_lin)
        red = torch.einsum("bhs,bhsk->bhk", alphas, values)
        red = red.view(red.size(0), -1)
        return red


class TokenEmb(torch.nn.Module):
    def __init__(self, emb:torch.nn.Embedding, adapt_dims=None, rare_token_ids:Set[int]=None, rare_id:int=None, **kw):
        super(TokenEmb, self).__init__(**kw)
        self.emb = emb
        self.rare_token_ids = rare_token_ids
        self.rare_id = rare_id
        self._do_rare()

        self.adapter = None
        if adapt_dims is not None and adapt_dims[0] != adapt_dims[1]:
            self.adapter = torch.nn.Linear(*adapt_dims)

        # self.init_params()

    def init_params(self):
        torch.nn.init.uniform_(self.emb.weight, -0.1, 0.1)
        torch.nn.init.constant_(self.emb.weight[0], 0)

    def _do_rare(self, rare_token_ids:Set[int]=None, rare_id:int=None):
        self.register_buffer("unkmask", None)
        self.rare_token_ids = self.rare_token_ids if rare_token_ids is None else rare_token_ids
        self.rare_id = self.rare_id if rare_id is None else rare_id
        if self.rare_id is not None and self.rare_token_ids is not None:
            # build id mapper
            unkmask = torch.ones(self.emb.num_embeddings)
            save = False
            for id in self.rare_token_ids:
                if id < unkmask.size(0):
                    unkmask[id] = 1
                    save = True
            if save:
                self.register_buffer("unkmask", unkmask)

    def forward(self, x:torch.Tensor):
        numembs = self.emb.num_embeddings
        unkmask = x >= numembs
        if torch.any(unkmask):
            x = x.masked_fill(unkmask, self.rare_id)
        if self.unkmask is not None:
            x = x.masked_fill(self.unkmask[None, :], self.rare_id)
        ret = self.emb(x)
        if self.adapter is not None:
            ret = self.adapter(ret)
        return ret


def load_pretrained_embeddings(emb, D, p="../data/glove/glove300uncased"):
    W = np.load(p + ".npy")
    with open(p + ".words") as f:
        words = ujson.load(f)
        preD = dict(zip(words, range(len(words))))
    # map D's indexes onto preD's indexes
    select = np.zeros(emb.num_embeddings, dtype="int64") - 1
    covered_words = set()
    covered_word_ids = set()
    for k, v in D.items():
        if k in preD:
            select[v] = preD[k]
            covered_words.add(k)
            covered_word_ids.add(v)
    selectmask = select != -1
    select = select * selectmask.astype("int64")
    subW = W[select, :]
    subW = torch.tensor(subW).to(emb.weight.device)
    selectmask = torch.tensor(selectmask).to(emb.weight.device).to(torch.float)
    emb.weight.data = emb.weight.data * (1-selectmask[:, None]) + subW * selectmask[:, None]        # masked set or something else?
    print("done")
    return covered_words, covered_word_ids


class GRUCell(torch.nn.Module):
    def __init__(self, dim, bias=True, dropout=0., dropout_rec=0., use_layernorm=True, long_range_bias=0., **kw):
        super(GRUCell, self).__init__(**kw)
        self.dim, self.bias = dim, bias
        self.long_range_bias = long_range_bias
        self.gateW = torch.nn.Linear(dim * 2, dim * 2, bias=bias)
        self.gateW.bias.data[dim:] = self.gateW.bias.data[dim:] + self.long_range_bias
        self.gateU = torch.nn.Linear(dim * 2, dim, bias=bias)
        self.sm = torch.nn.Softmax(-1)
        self.dropout = torch.nn.Dropout(dropout)
        self.dropout_rec = torch.nn.Dropout(dropout_rec)
        self.register_buffer("dropout_mask", torch.ones(self.dim * 2))
        self.ln = None
        self.ln2 = None
        if use_layernorm:
            self.ln = torch.nn.LayerNorm(dim * 2)
            self.ln2 = torch.nn.LayerNorm(dim * 2)

    def reset_dropout(self):
        device = self.dropout_mask.device
        ones = torch.ones(self.dim * 2, device=device)
        self.dropout_mask = self.dropout_rec(ones)

    def forward(self, x, h):
        inp = torch.cat([x, h], 1)
        if self.ln is not None:
            inp = self.ln(inp)
        inp = self.dropout(inp)
        inp = inp * self.dropout_mask[None, :]
        gates = self.gateW(inp)
        gates = gates.chunk(2, 1)
        r = torch.sigmoid(gates[0])
        z = torch.sigmoid(gates[1])
        inp = torch.cat([x, h * r], 1)
        if self.ln2 is not None:
            inp = self.ln2(inp)
        inp = self.dropout(inp)
        inp = inp * self.dropout_mask[None, :]
        u = self.gateU(inp)
        u = torch.tanh(u)
        h_new = h * z + (1 - z) * u
        return h_new


GATE_BIAS = 0
class DGRUCell(torch.nn.Module):
    zoneout_frac = .7
    def __init__(self, dim, bias=True, dropout=0., dropout_rec=0., zoneout=0., gate_bias=GATE_BIAS, usegradskip=False, use_layernorm=True, **kw):
        super(DGRUCell, self).__init__(**kw)
        self.dim, self.bias = dim, bias
        self.gateW = torch.nn.Linear(dim * 2, dim * 5, bias=bias)
        self.gateU = torch.nn.Linear(dim * 2, dim, bias=bias)
        self.sm = torch.nn.Softmax(-1)
        self.dropout = torch.nn.Dropout(dropout)
        self.dropout_rec = torch.nn.Dropout(dropout_rec)
        self.zoneout = zoneout
        self.register_buffer("dropout_mask", torch.ones(self.dim * 2))
        self.gate_bias = gate_bias
        self.usegradskip = usegradskip

        self.ln = None
        self.ln2 = None
        if use_layernorm:
            self.ln = torch.nn.LayerNorm(dim * 2)
            self.ln2 = torch.nn.LayerNorm(dim * 2)

    def reset_dropout(self):
        device = self.dropout_mask.device
        ones = torch.ones(self.dim * 2, device=device)
        self.dropout_mask = self.dropout_rec(ones)

    def forward(self, x, h):
        inp = torch.cat([x, h], 1)
        if self.ln is not None:
            inp = self.ln(inp)
        inp = self.dropout(inp)
        inp = inp * self.dropout_mask[None, :]
        gates = self.gateW(inp)
        gates = list(gates.chunk(5, 1))
        rx = torch.sigmoid(gates[0])
        rh = torch.sigmoid(gates[1])
        z_gates = gates[2:5]
        z_gates[2] = z_gates[2] - self.gate_bias
        z = torch.softmax(torch.stack(z_gates, -1), -1)
        inp = torch.cat([x * rx, h * rh], 1)
        if self.ln2 is not None:
            inp = self.ln2(inp)
        inp = self.dropout(inp)
        inp = inp * self.dropout_mask[None, :]
        u = self.gateU(inp)
        u = torch.tanh(u)
        if self.usegradskip:
            h_new = forward_mix_backward_sum_multi(torch.stack([x, h, u], -1), z)
        else:
            h_new = torch.stack([x, h, u], 2) * z
            h_new = h_new.sum(-1)

        # do two-way zoneout
        if self.zoneout > 0. and self.training:
            a = self.zoneout * self.zoneout_frac  # until "a", take x
            b = self.zoneout    # between "a" and "b", take h, after "b", take h_new
            zo_rand = torch.rand_like(h_new)
            mix_x = zo_rand <= a
            mix_h = (zo_rand > a) & (zo_rand <= b)
            mix_h_new = zo_rand > b
            h_new = x * mix_x.float() + h * mix_h.float() + h_new * mix_h_new.float()
        return h_new


class NIGRUCell(torch.nn.Module):
    """
    "No-input" GRU cell: ignores state input, treats input as state
    """
    def __init__(self, dim, bias=True, dropout_rec=0., **kw):
        super(NIGRUCell, self).__init__(**kw)
        self.dim, self.bias = dim, bias
        self.gateW = torch.nn.Linear(dim, dim * 2, bias=bias)
        self.gateU = torch.nn.Linear(dim, dim, bias=bias)
        self.sm = torch.nn.Softmax(-1)
        self.dropout_rec = torch.nn.Dropout(dropout_rec)
        self.register_buffer("dropout_mask", torch.ones(self.dim))

    def reset_dropout(self):
        device = self.dropout_mask.device
        ones = torch.ones(self.dim, device=device)
        self.dropout_mask = self.dropout_rec(ones)

    def forward(self, x, *args, **kw):
        inp = x
        inp = inp * self.dropout_mask[None, :]
        gates = self.gateW(inp)
        gates = gates.chunk(2, 1)
        r = torch.sigmoid(gates[0])
        z = torch.sigmoid(gates[1])
        inp = x * r
        inp = inp * self.dropout_mask[None, :]
        u = self.gateU(inp)
        u = torch.tanh(u)
        h_new = z * x + (1 - z) * u
        return h_new


class SimpleDGRUCell(torch.nn.Module):
    def __init__(self, dim, bias=True, dropout_rec=0., **kw):
        super(SimpleDGRUCell, self).__init__(**kw)
        self.dim, self.bias = dim, bias
        self.gateW = torch.nn.Linear(dim * 2, dim * 4, bias=bias)
        self.sm = torch.nn.Softmax(-1)
        self.dropout_rec = torch.nn.Dropout(dropout_rec)
        self.register_buffer("dropout_mask", torch.ones(self.dim * 2))

    def reset_dropout(self):
        device = self.dropout_mask.device
        ones = torch.ones(self.dim * 2, device=device)
        self.dropout_mask = self.dropout_rec(ones)

    def forward(self, x, h):
        inp = torch.cat([x, h], 1)
        inp = inp * self.dropout_mask[None, :]
        gates = self.gateW(inp)
        gates = gates.chunk(4, 1)
        z = torch.softmax(torch.stack(gates[:3], 2), -1)
        u = torch.tanh(gates[-1])
        h_new = torch.stack([x, h, u], 2) * z
        h_new = h_new.sum(-1)
        return h_new


class SumAct(torch.nn.Module):
    def __init__(self, dim=None, act=None, bias=True, **kw):
        super(SumAct, self).__init__(**kw)
        if dim is not None:
            self.lin = torch.nn.Linear(dim*2, dim, bias=bias)
        else:
            self.lin = None
        if act == "relu":
            self.act = torch.nn.ReLU()
        elif act is None:
            self.act = None

    def forward(self, x, h):
        if self.lin is not None:
            ret = self.lin(torch.cat([x, h], -1))
        else:
            ret = (x + h) / 2
        if self.act is not None:
            ret = self.act(ret)
        return ret


class LinearSkip(torch.nn.Module):
    """ A two-layer perceptron with a skip from input to output. """
    def __init__(self, dim=None, hdim=None, act=None, bias=True, useln=False, dropout=0., **kw):
        super(LinearSkip, self).__init__(**kw)
        self.dim = dim
        self.hdim = hdim if hdim is not None else self.dim * 2
        self.linA = torch.nn.Linear(self.dim, self.hdim, bias=bias)
        self.linB = torch.nn.Linear(self.hdim, self.dim, bias=bias)
        self.act = act if act is not None else torch.nn.LeakyReLU()
        self.ln = torch.nn.LayerNorm(self.dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.useln = useln

    def forward(self, x):
        ret = x
        ret = self.dropout(ret)
        ret = self.linA(ret)
        ret = self.act(ret)
        ret = self.linB(ret)
        if self.useln:
            ret = ret + x
            ret = self.ln(ret)
        else:
            ret = (ret + x) / 2
        return ret



class DSDGRUCell(torch.nn.Module):
    def __init__(self, dim, bias=True, dropout_rec=0., **kw):
        super(DSDGRUCell, self).__init__(**kw)
        self.dim, self.bias = dim, bias
        self.gateW = torch.nn.Linear(dim * 2, dim * 8, bias=bias)
        self.gateU = torch.nn.Linear(dim * 2, dim, bias=bias)
        self.sm = torch.nn.Softmax(-1)
        self.dropout_rec = torch.nn.Dropout(dropout_rec)
        self.register_buffer("dropout_mask", torch.ones(self.dim * 2))

    def reset_dropout(self):
        device = self.dropout_mask.device
        ones = torch.ones(self.dim * 2, device=device)
        self.dropout_mask = self.dropout_rec(ones)

    def init_biases(self):      # initialize biases to remember longer in beginning
        pv = 3
        nv = 0
        if self.bias:
            self.gateW.bias.data[self.dim*2:].fill_(nv)
            self.gateW.bias.data[self.dim*3:self.dim*4].fill_(pv)
            self.gateW.bias.data[self.dim*5:self.dim*6].fill_(pv)
        else:
            print("WARNING: no biases in model so this method has no effect. Please create object with bias=True.")

    def forward(self, x, c):
        inp = torch.cat([x, c], 1)
        inp = inp * self.dropout_mask[None, :]
        gates = self.gateW()
        gates = gates.chunk(8, 1)
        rx = torch.sigmoid(gates[0])
        rh = torch.sigmoid(gates[1])
        z_c = torch.softmax(torch.stack(gates[2:5], 2), -1)
        z_o = torch.softmax(torch.stack(gates[5:8], 2), -1)
        inp = torch.cat([x * rx, c * rh], 1)
        inp = inp * self.dropout_mask[None, :]
        u = self.gateU(inp)
        u = torch.tanh(u)
        c_new = torch.stack([x, c, u], 2) * z_c
        c_new = c_new.sum(-1)
        h_new = torch.stack([x, c, u], 2) * z_o
        h_new = h_new.sum(-1)
        return h_new, c_new


def try_dgru_cell():
    xs = torch.nn.Parameter(torch.rand(2, 10, 5))
    _hs = torch.nn.Parameter(torch.rand(2, 10, 5))
    hs = [h[:, 0, :] for h in _hs.split(1, 1)]

    m = [SimpleDGRUCell(5) for _ in range(10)]
    # m = [torch.nn.GRUCell(5, 5) for _ in range(10)]
    for i in range(xs.size(1)):
        x = xs[:, i]
        for j in range(len(hs)):
            h = hs[j]
            y = m[j](x, h)
            x = y
            hs[j] = y
    print(y)
    y.sum().backward()
    print(xs.grad[:, 0].norm())
    print(_hs.grad[:, 0].norm())


class _Encoder(torch.nn.Module):
    def __init__(self, embdim, hdim, num_layers=1, dropout=0., bidirectional=True, **kw):
        super(_Encoder, self).__init__(**kw)
        self.embdim, self.hdim, self.numlayers, self.bidir, self.dropoutp = embdim, hdim, num_layers, bidirectional, dropout
        self.dropout = torch.nn.Dropout(dropout)
        self.create_rnn()

    def forward(self, x, initstate=None, mask=None):
        x = self.dropout(x)

        if mask is not None:
            _x = torch.nn.utils.rnn.pack_padded_sequence(x, mask.sum(-1), batch_first=True, enforce_sorted=False)
        else:
            _x = x

        _outputs, hidden = self.rnn(_x, initstate)

        if mask is not None:
            y, _ = torch.nn.utils.rnn.pad_packed_sequence(_outputs, batch_first=True)
        else:
            y = _outputs

        hidden = (hidden,) if not q.issequence(hidden) else hidden
        hiddens = []
        for _hidden in hidden:
            i = 0
            _hiddens = tuple()
            while i < _hidden.size(0):
                if self.bidir is True:
                    _h = torch.cat([_hidden[i], _hidden[i+1]], -1)
                    i += 2
                else:
                    _h = _hidden[i]
                    i += 1
                _hiddens = _hiddens + (_h,)
            hiddens.append(_hiddens)
        hiddens = tuple(zip(*hiddens))
        return y, hiddens


class RNNEncoder(_Encoder):
    def create_rnn(self):
        self.rnn = torch.nn.RNN(self.embdim, self.hdim, self.numlayers,
                                bias=True, batch_first=True, dropout=self.dropoutp, bidirectional=self.bidir)

    def init_params(self):
        for name, param in self.rnn.named_parameters():
            if 'weight' in name or 'bias' in name:
                param.data.uniform_(-0.1, 0.1)


class GRUEncoder(_Encoder):
    def create_rnn(self):
        self.rnn = torch.nn.GRU(self.embdim, self.hdim, self.numlayers,
                                bias=True, batch_first=True, dropout=self.dropoutp, bidirectional=self.bidir)
        self.init_params()

    def init_params(self):
        for name, param in self.rnn.named_parameters():
            if 'weight' in name or 'bias' in name:
                param.data.uniform_(-0.1, 0.1)


class LSTMEncoder(_Encoder):
    def create_rnn(self):
        self.rnn = torch.nn.LSTM(self.embdim, self.hdim, self.numlayers,
                                bias=True, batch_first=True, dropout=self.dropoutp, bidirectional=self.bidir)
        self.init_params()

    def init_params(self):
        for name, param in self.rnn.named_parameters():
            if 'weight' in name or 'bias' in name:
                param.data.uniform_(-0.1, 0.1)


class BasicGenOutput(torch.nn.Module):
    def __init__(self, h_dim:int, vocab:Vocab=None, dropout:float=0., **kw):
        super(BasicGenOutput, self).__init__(**kw)
        self.gen_lin = torch.nn.Linear(h_dim, vocab.number_of_ids(), bias=True)
        self.sm = torch.nn.Softmax(-1)
        self.logsm = torch.nn.LogSoftmax(-1)
        self.dropout = torch.nn.Dropout(dropout)

        self.vocab = vocab

        # rare output tokens
        self.rare_token_ids = vocab.rare_ids
        if len(self.rare_token_ids) > 0:
            out_mask = torch.ones(self.vocab.number_of_ids())
            for rare_token_id in self.rare_token_ids:
                out_mask[rare_token_id] = 0
            self.register_buffer("out_mask", out_mask)
        else:
            self.register_buffer("out_mask", None)

    def forward(self, x:torch.Tensor, out_mask=None, **kw):
        """
        :param x:           (batsize, hdim)
        :param out_mask:    (batsize, vocsize)
        :param kw:
        :return:
        """
        x = self.dropout(x)
        # - generation probs
        gen_probs = self.gen_lin(x)
        if self.out_mask is not None:
            gen_probs = gen_probs + torch.log(self.out_mask)[None, :]
        if out_mask is not None:
            gen_probs = gen_probs + torch.log(out_mask)
        gen_probs = self.logsm(gen_probs)
        return gen_probs


def local_var(self:dgl.DGLGraph):
    return self.local_var()

# def local_var(self:dgl.BatchedDGLGraph):
#     local_node_frame = FrameRef(Frame(self._node_frame._frame))
#     local_edge_frame = FrameRef(Frame(self._edge_frame._frame))
#     # Use same per-column initializers and default initializer.
#     # If registered, a column (based on key) initializer will be used first,
#     # otherwise the default initializer will be used.
#     sync_frame_initializer(local_node_frame._frame, self._node_frame._frame)
#     sync_frame_initializer(local_edge_frame._frame, self._edge_frame._frame)
#
#     batched_graph = dgl.BatchedDGLGraph([dgl.DGLGraph()], ALL, ALL)
#     batched_graph._graph = self._graph
#     batched_graph._node_frame = local_node_frame
#     batched_graph._edge_frame = local_edge_frame
#     batched_graph._batch_size = self._batch_size
#     batched_graph._batch_num_nodes = self._batch_num_nodes
#     batched_graph._batch_num_edges = self._batch_num_edges
#     return batched_graph


class AttentionReadout(torch.nn.Module):
    # First, apply multihead attention over all nodes within the batched graph
    # Second, apply outnet on per-graph aggregation
    def __init__(self, hdim, outdim=None, numheads=2, outnet="default",
                 _detach_scores=False, **kw):
        super(AttentionReadout, self).__init__(**kw)
        self.hdim, self.numheads = hdim, numheads
        self.outdim = outdim if outdim is not None else hdim
        self.att_vecs = torch.nn.Parameter(torch.zeros(self.hdim, self.numheads))
        if outnet == "default":
            self.outnet = torch.nn.Linear(self.hdim, self.outdim)
        else:
            self.outnet = outnet
        self.reset_parameters()
        self._detach_scores = _detach_scores

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.att_vecs, a=np.sqrt(5))

    def forward(self, x:dgl.DGLGraph, states:torch.Tensor=None):
        x = x.local_var()
        att_scores = torch.einsum("bd,dh->bh", states, self.att_vecs)
        if self._detach_scores:
            att_scores = att_scores.detach()
        x.ndata["_local_weights"] = att_scores
        att_weights = dgl.softmax_nodes(x, "_local_weights")    # bh
        v = states.view(states.size(0), self.numheads, -1)  # bhl
        v = v * att_weights[:, :, None]
        x.ndata["_local_v"] = v.view(v.size(0), -1)
        attn_summ = dgl.sum_nodes(x, "_local_v")
        ret = attn_summ
        ret = self.outnet(ret)
        return ret


def try_attention_readout():
    x = dgl.DGLGraph()
    x.add_nodes(2)
    y = dgl.DGLGraph()
    y.add_nodes(3)
    x = dgl.batch([x, y])
    x.ndata["id"] = torch.arange(0, x.number_of_nodes())

    print(x.ndata["id"])

    hdim = 6
    h = torch.rand(x.number_of_nodes(), hdim)

    attnro = AttentionReadout(hdim, 2)
    z = attnro(x, h)

    print(z)




if __name__ == '__main__':
    # try_dgru_cell()
    try_attention_readout()