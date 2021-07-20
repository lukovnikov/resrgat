from unittest import TestCase

import dgl
import torch
from dgl.nn.pytorch.utils import Identity

from ldgnn.nn import WindowHeadAttention, AttentionReadout, forward_mix_backward_sum, forward_mix_backward_sum_multi


class TestForwardMixBackwardSum(TestCase):
    def test_it(self):
        a = torch.nn.Parameter(torch.rand(5, 4))
        b = torch.nn.Parameter(torch.rand(5, 4))
        mix = torch.nn.Parameter(torch.sigmoid(torch.randn(5, 4)))
        o = forward_mix_backward_sum(a, b, mix)

        o_ref = a * mix + b * (1 - mix)
        self.assertTrue(torch.allclose(o, o_ref))

    def test_grad(self):
        a = torch.nn.Parameter(torch.rand(5, 4))
        a2 = torch.nn.Parameter(a.clone().data)
        a3 = torch.nn.Parameter(a.clone().data)
        b = torch.nn.Parameter(torch.rand(5, 4))
        b2 = torch.nn.Parameter(b.clone().data)
        b3 = torch.nn.Parameter(b.clone().data)
        mix = torch.nn.Parameter(torch.sigmoid(torch.randn(5, 4)))
        mix2 = torch.nn.Parameter(mix.clone().data)
        o = forward_mix_backward_sum(a, b, mix)

        o2 = a2 * mix2 + b2 * (1 - mix2)
        o3 = a3 + b3

        o.sum().backward()
        o2.sum().backward()
        o3.sum().backward()
        self.assertTrue(torch.allclose(o, o2))
        self.assertTrue(torch.allclose(a.grad, a3.grad))
        self.assertTrue(torch.allclose(b.grad, b3.grad))
        self.assertTrue(torch.allclose(mix.grad, mix2.grad))


class TestForwardMixBackwardSumMulti(TestCase):
    def test_it(self):
        a = torch.nn.Parameter(torch.rand(5, 4))
        b = torch.nn.Parameter(torch.rand(5, 4))
        mix = torch.nn.Parameter(torch.sigmoid(torch.randn(5, 4)))
        mix = torch.stack([mix, 1 - mix], -1)
        h = torch.stack([a, b], -1)
        o = forward_mix_backward_sum_multi(h, mix)

        print(h.size(), mix.size())
        print(h)

        o_ref = (h * mix).sum(-1)
        self.assertTrue(torch.allclose(o, o_ref))

    def test_grad(self):
        a = torch.nn.Parameter(torch.rand(5, 4, 2))
        a2 = torch.nn.Parameter(a.clone().data)
        a3 = torch.nn.Parameter(a.clone().data)
        mix = torch.nn.Parameter(torch.sigmoid(torch.randn(5, 4)))
        mix = torch.stack([mix, 1-mix], -1)
        mix = torch.nn.Parameter(mix.clone().data)
        mix2 = torch.nn.Parameter(mix.clone().data)
        o = forward_mix_backward_sum_multi(a, mix)

        o2 = (a2 * mix2).sum(-1)
        o3 = a3.sum(-1)

        o.sum().backward()
        o2.sum().backward()
        o3.sum().backward()
        self.assertTrue(torch.allclose(o, o2))
        self.assertTrue(torch.allclose(a.grad, a3.grad))
        self.assertTrue(torch.allclose(mix.grad, mix2.grad))


class TestMultiHeadAttention(TestCase):
    def test_it(self):
        h = torch.nn.Parameter(torch.randn(3, 20))
        hs = torch.nn.Parameter(torch.randn(3, 5, 20))
        msg = torch.nn.Parameter(torch.randn(3, 5, 20))

        m = WindowHeadAttention(20, 20, 20, numheads=4)
        y = m(h, hs, msg)
        print(y.size())
        y[0, :5].sum().backward()
        print(h.grad[0])
        print(h.grad[1])
        print(hs.grad[0])
        print(hs.grad[1])
        print(msg.grad[0])


class TestAttentionReadout(TestCase):
    def test_it(self):
        print(dgl.__version__)
        x = dgl.DGLGraph()
        x.add_nodes(2)
        y = dgl.DGLGraph()
        y.add_nodes(3)
        x = dgl.batch([x, y])
        x.ndata["id"] = torch.arange(0, x.number_of_nodes())

        print(x.ndata["id"])

        hdim = 6
        h = torch.rand(x.number_of_nodes(), hdim)
        h = torch.nn.Parameter(h)

        attnro = AttentionReadout(hdim, hdim, numheads=2, outnet=Identity(), _detach_scores=True)
        z = attnro(x, h)
        print(z)

        z[:, :hdim//2].sum().backward()
        print(h.grad)
        assert(not torch.allclose(h.grad[:, :hdim//2], torch.zeros_like(h.grad[:, :hdim//2])))
        assert(torch.allclose(h.grad[:, hdim//2:], torch.zeros_like(h.grad[:, hdim//2:])))
        # print(z)
