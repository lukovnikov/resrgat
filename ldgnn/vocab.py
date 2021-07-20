from abc import ABC, abstractmethod
from typing import Union, Callable, List, Dict
import numpy as np

import torch


class _Vocab(object):
    pass


class Vocab(_Vocab):
    padtoken = "@PAD@"
    unktoken = "@UNK@"
    starttoken = "@START@"
    endtoken = "@END@"
    def __init__(self, padid:int=0, unkid:int=1, startid:int=2, endid:int=3, **kw):
        self.D = {self.padtoken: padid, self.unktoken: unkid}
        self.D[self.starttoken] = startid
        self.D[self.endtoken] = endid
        self.counts = {k: np.infty for k in self.D.keys()}
        self.rare_tokens = set()
        self.rare_ids = set()
        self.RD = {v: k for k, v in self.D.items()}
        self.growing = True
        self._tokenmapper = None

    def nextid(self):
        return max(self.D.values()) + 1

    def finalize(self, min_freq:int=0, top_k:int=np.infty, keep_rare=False):
        self.growing = False
        sorted_counts = sorted(self.counts.items(), key=lambda x: x[1], reverse=True)

        if min_freq == 0 and top_k > len(sorted_counts):
            self.rare_tokens = set()
        else:
            if top_k < len(sorted_counts) and sorted_counts[top_k][1] > min_freq:
                i = top_k
            else:
                if top_k < len(sorted_counts):
                    sorted_counts = sorted_counts[:top_k]
                # binary search for min_freq position
                i = 0
                divider = 2
                where = +1
                while True:
                    i += where * len(sorted_counts) // divider
                    if (i == len(sorted_counts)) or (
                            sorted_counts[i][1] <= min_freq - 1 and sorted_counts[i - 1][1] >= min_freq):
                        break  # found
                    elif sorted_counts[i][1] < min_freq:  # go up
                        where = -1
                    elif sorted_counts[i][1] >= min_freq:  # go down
                        where = +1
                    divider *= 2
                    divider = min(divider, len(sorted_counts))
            self.rare_tokens = set([t[0] for t in sorted_counts[i:]])
            if not keep_rare:
                sorted_counts = sorted_counts[:i]

        nextid = max(self.D.values()) + 1
        for token, cnt in sorted_counts:
            if token not in self.D:
                self.D[token] = nextid
                nextid += 1

        self.RD = {v: k for k, v in self.D.items()}
        if keep_rare:
            self.rare_ids = set([self[rare_token] for rare_token in self.rare_tokens])

    def add_token(self, token, seen:Union[int,bool]=True):
        assert(self.growing)
        if token not in self.counts:
            self.counts[token] = 0
        if seen > 0:
            self.counts[token] += float(seen)

    def __getitem__(self, item:str) -> int:
        if self._tokenmapper is not None:
            item = self._tokenmapper(self, item)
        if item not in self.D:
            assert(self.unktoken in self.D)
            item = self.unktoken
        id = self.D[item]
        return id

    def __call__(self, item:int) -> str:
        return self.RD[item]

    def number_of_ids(self, last_nonrare=True):
        if not last_nonrare:
            return max(self.D.values()) + 1
        else:
            return max(set(self.D.values()) - self.rare_ids) + 1

    def reverse(self):
        return {v: k for k, v in self.D.items()}

    def __iter__(self):
        return iter([(k, v) for k, v in self.D.items()])

    def __contains__(self, item:Union[str,int]):
        if isinstance(item, str):
            return item in self.D
        if isinstance(item, int):
            return item in self.RD
        else:
            raise Exception("illegal argument")

    def tostr(self, x:Union[np.ndarray, torch.Tensor], return_tokens=False):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = list(np.vectorize(lambda e: self(e))(x))
        x = [e for e in x if e != self.padtoken]
        ret = []
        for xe in x:
            if len(ret) > 0 and ret[-1] == self.endtoken:
                break
            ret.append(xe)
        if return_tokens:
            return ret
        else:
            return " ".join(ret)


class FixedVocab(Vocab):
    def __init__(self, padid:int=0, unkid:int=1, vocab:Dict=None, **kw):
        super(FixedVocab, self).__init__(padid, unkid, **kw)
        self.D = vocab
        self.growing = False

    def add_token(self, token, seen=True):
        print("Warning: trying to add token to fixed vocab")
        pass

    def do_rare(self, min_freq=0, top_k=np.infty):
        print("Warning: trying to do rare on fixed vocab")
        pass


def try_vocab():
    vocab = Vocab()
    tokens = "a b c d e a b c d a b c a b a a a a b e d g m o i p p x x i i b b ai ai bi bi bb bb abc abg abh abf".split()
    for t in tokens:
        vocab.add_token(t)
    vocab.do_rare(min_freq=2, top_k=15)
    print(vocab.rare_tokens)
    print(vocab.rare_ids)


class VocabBuilder(ABC):
    @abstractmethod
    def inc_build_vocab(self, x:str, seen:bool=True):
        raise NotImplemented()

    @abstractmethod
    def finalize_vocab(self, min_freq:int=0, top_k:int=np.infty):
        raise NotImplemented()

    @abstractmethod
    def vocabs_finalized(self):
        raise NotImplemented()

    
class SequenceEncoder(VocabBuilder):
    endtoken = "@END@"
    def __init__(self, tokenizer:Callable[[str], List[str]], vocab:Vocab=None, add_end_token=False, **kw):
        super(SequenceEncoder, self).__init__(**kw)
        self.tokenizer = tokenizer
        self.vocab = vocab if vocab is not None else Vocab()
        self.vocab_final = False
        self.add_end_token = add_end_token
        
    def inc_build_vocab(self, x:str, seen:bool=True):
        if not self.vocab_final:
            tokens = self.tokenizer(x)
            if self.add_end_token:
                tokens.append(self.endtoken)
            for token in tokens:
                self.vocab.add_token(token, seen=seen)
            return tokens
        else:
            return []
    
    def finalize_vocab(self, min_freq:int=0, top_k:int=np.infty, keep_rare=False):
        self.vocab_final = True
        self.vocab.finalize(min_freq=min_freq, top_k=top_k, keep_rare=keep_rare)
        
    def vocabs_finalized(self):
        return self.vocab_final
    
    def convert(self, x:str, return_what="tensor"):     # "tensor", "ids", "tokens" or comma-separated combo of all
        rets = [r.strip() for r in return_what.split(",")]
        tokens = self.tokenizer(x)
        if self.add_end_token:
            tokens.append(self.endtoken)
        ids = [self.vocab[token] for token in tokens]
        tensor = torch.tensor(ids, dtype=torch.long)
        ret = {"tokens": tokens, "ids": ids, "tensor": tensor}
        ret = [ret[r] for r in rets]
        return ret


if __name__ == '__main__':
    try_vocab()
    # try_func_query_encoder()