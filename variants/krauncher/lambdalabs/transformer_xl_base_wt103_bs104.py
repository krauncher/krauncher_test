"""Krauncher variant of tasks/lambdalabs/transformer_xl_base_wt103_bs104.py (the neutral form, with the
source and the reduction it documents); the expected forecast and its
sources are in tasks/lambdalabs/transformer_xl_base_wt103_bs104.json.

The task function holds the neutral file's code — the imports, the sections
copied verbatim from NVIDIA DeepLearningExamples (headed by file and lines,
licence headers kept) and the run — indented into one self-contained
function. Difference from the neutral form: the preparation phase is the
platform's data delivery — the WikiText-103 archive comes through a
registered data source (/data, read-only) and the task unpacks it into its
working directory and renames the three files, as prepare_data() does. FUNC
is the task, OPTIONS the arguments of its @client.task decorator, KWARGS the
arguments of the call.
tools/build_dle_transformer_xl.py rebuilds this file from a checkout of the source.
"""

OPTIONS = dict(
    timeout=3600,
    data="wikitext-103-v1",
)

# Data source the task reads (OPTIONS["data"]), registered on the account when
# missing: WikiText-103 (CC BY-SA, Salesforce Research), word-level archive.
DATA_SOURCES = [
    dict(name="wikitext-103-v1", urls=["https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"], size_gb=0.2),
]

KWARGS = dict(batch_size=104, max_step=400)


def train_transformer_xl(batch_size: int = 104, max_step: int = 400):
    import argparse
    import contextlib
    import functools
    import glob
    import itertools
    import math
    import os
    import time
    import types
    from collections import Counter
    from collections import OrderedDict
    from contextlib import contextmanager

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.optim import Optimizer

    # --- licence: PyTorch/LanguageModeling/Transformer-XL/NOTICE ---
    # Transformer-XL for PyTorch
    #
    # This repository includes software from https://github.com/kimiyoung/transformer-xl licensed under the Apache License 2.0.
    #
    # This repository includes software from https://github.com/salesforce/awd-lstm-lm licensed under the BSD-3-Clause license.
    #
    # This repository includes software from https://github.com/cybertronai/transformer-xl licensed under the Apache License 2.0.
    #
    # This repository includes software from https://github.com/cybertronai/pytorch-lamb licensed under the MIT license.


    # --- licence header: PyTorch/LanguageModeling/Transformer-XL/pytorch/utils/distributed.py:1-14 ---
    # Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #       http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.


    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/utils/distributed.py:38-43 ---
    def barrier():
        """
        Call torch.distributed.barrier() if distritubed is in use
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()


    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/utils/distributed.py:46-54 ---
    def get_rank():
        """
        Gets distributed rank or returns zero if distributed is not initialized.
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        return rank


    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/utils/distributed.py:57-66 ---
    def get_world_size():
        """
        Gets total number of distributed workers or returns one if distributed is
        not initialized.
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1
        return world_size


    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/utils/distributed.py:114-121 ---
    @contextmanager
    def sync_workers():
        """
        Yields distributed rank and synchronizes all workers on exit.
        """
        rank = get_rank()
        yield rank
        barrier()


    # The package `utils` that the copied data code refers to.
    utils = types.SimpleNamespace(distributed=types.SimpleNamespace(
        get_rank=get_rank, get_world_size=get_world_size, sync_workers=sync_workers))


    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/utils/vocabulary.py:25-187 ---
    class Vocab(object):
        def __init__(self, special=[], min_freq=0, max_size=None, lower_case=True,
                     delimiter=None, vocab_file=None):
            self.counter = Counter()
            self.special = special
            self.min_freq = min_freq
            self.max_size = max_size
            self.lower_case = lower_case
            self.delimiter = delimiter
            self.vocab_file = vocab_file

        def tokenize(self, line, add_eos=False, add_double_eos=False):
            line = line.strip()
            # convert to lower case
            if self.lower_case:
                line = line.lower()

            # empty delimiter '' will evaluate False
            if self.delimiter == '':
                symbols = line
            else:
                symbols = line.split(self.delimiter)

            if add_double_eos:  # lm1b
                return ['<S>'] + symbols + ['<S>']
            elif add_eos:
                return symbols + ['<eos>']
            else:
                return symbols

        def count_file(self, path, verbose=False, add_eos=False):
            if verbose:
                print('counting file {} ...'.format(path))
            assert os.path.exists(path)

            sents = []
            with open(path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    if verbose and idx > 0 and idx % 500000 == 0:
                        print('    line {}'.format(idx))
                    symbols = self.tokenize(line, add_eos=add_eos)
                    self.counter.update(symbols)
                    sents.append(symbols)

            return sents

        def count_sents(self, sents, verbose=False):
            """
                sents : a list of sentences, each a list of tokenized symbols
            """
            if verbose:
                print('counting {} sents ...'.format(len(sents)))
            for idx, symbols in enumerate(sents):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                self.counter.update(symbols)

        def _build_from_file(self, vocab_file):
            self.idx2sym = []
            self.sym2idx = OrderedDict()

            with open(vocab_file, 'r', encoding='utf-8') as f:
                for line in f:
                    symb = line.strip().split()[0]
                    self.add_symbol(symb)
            self.unk_idx = self.sym2idx['<UNK>']

        def build_vocab(self):
            if self.vocab_file:
                print('building vocab from {}'.format(self.vocab_file))
                self._build_from_file(self.vocab_file)
                print('final vocab size {}'.format(len(self)))
            else:
                print('building vocab with min_freq={}, max_size={}'.format(
                    self.min_freq, self.max_size))
                self.idx2sym = []
                self.sym2idx = OrderedDict()

                for sym in self.special:
                    self.add_special(sym)

                for sym, cnt in self.counter.most_common(self.max_size):
                    if cnt < self.min_freq:
                        break
                    self.add_symbol(sym)

                print('final vocab size {} from {} unique tokens'.format(
                    len(self), len(self.counter)))

        def encode_file(self, path, ordered=False, verbose=False, add_eos=True,
                        add_double_eos=False):
            if verbose:
                print('encoding file {} ...'.format(path))
            assert os.path.exists(path)
            encoded = []
            with open(path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    if verbose and idx > 0 and idx % 500000 == 0:
                        print('    line {}'.format(idx))
                    symbols = self.tokenize(line, add_eos=add_eos,
                                            add_double_eos=add_double_eos)
                    encoded.append(self.convert_to_tensor(symbols))

            if ordered:
                encoded = torch.cat(encoded)

            return encoded

        def encode_sents(self, sents, ordered=False, verbose=False):
            if verbose:
                print('encoding {} sents ...'.format(len(sents)))
            encoded = []
            for idx, symbols in enumerate(sents):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                encoded.append(self.convert_to_tensor(symbols))

            if ordered:
                encoded = torch.cat(encoded)

            return encoded

        def add_special(self, sym):
            if sym not in self.sym2idx:
                self.idx2sym.append(sym)
                self.sym2idx[sym] = len(self.idx2sym) - 1
                setattr(self, '{}_idx'.format(sym.strip('<>')), self.sym2idx[sym])

        def add_symbol(self, sym):
            if sym not in self.sym2idx:
                self.idx2sym.append(sym)
                self.sym2idx[sym] = len(self.idx2sym) - 1

        def get_sym(self, idx):
            assert 0 <= idx < len(self), 'Index {} out of range'.format(idx)
            return self.idx2sym[idx]

        def get_idx(self, sym):
            if sym in self.sym2idx:
                return self.sym2idx[sym]
            else:
                # print('encounter unk {}'.format(sym))
                assert '<eos>' not in sym
                assert hasattr(self, 'unk_idx')
                return self.sym2idx.get(sym, self.unk_idx)

        def get_symbols(self, indices):
            return [self.get_sym(idx) for idx in indices]

        def get_indices(self, symbols):
            return [self.get_idx(sym) for sym in symbols]

        def convert_to_tensor(self, symbols):
            return torch.LongTensor(self.get_indices(symbols))

        def convert_to_sent(self, indices, exclude=None):
            if exclude is None:
                return ' '.join([self.get_sym(idx) for idx in indices])
            else:
                return ' '.join([self.get_sym(idx) for idx in indices if idx not in exclude])

        def __len__(self):
            return len(self.idx2sym)


    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/utils/vocabulary.py:192-237 ---
    class OpenAIVocab(Vocab):
        def __init__(self, max_size=None, vocab_file=None):
            from pytorch_transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.EOT = self.tokenizer.encoder['<|endoftext|>']
            self.max_size = max_size
            self.vocab_file = vocab_file

            pad = 8
            vocab_size = len(self.tokenizer)
            padded_vocab_size = (vocab_size + pad - 1) // pad * pad
            for i in range(0, padded_vocab_size - vocab_size):
                token = f'madeupword{i:09d}'
                self.tokenizer.add_tokens([token])

        def __len__(self):
            return len(self.tokenizer)

        def count_file(self, path, verbose=False, add_eos=False):
            # TODO: train from scratch, respect self.max_size
            pass

        def build_vocab(self):
            pass

        def encode_file(self, path, ordered=False, verbose=False, add_eos=True, add_double_eos=False) -> torch.LongTensor:
            cached = path + '.bpe'
            if os.path.exists(cached):
                return torch.load(cached)
            print(f'encoding file {path} ...')
            assert os.path.exists(path), f"{path} doesn't exist"

            with open(path, encoding='utf-8') as f:
                # Suppress warnings about length.
                with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
                    out = torch.LongTensor(self.tokenizer.encode(f.read()) + [self.EOT])
                    with utils.distributed.sync_workers() as rank:
                        if rank == 0:
                            torch.save(out, cached)
                    return out

        def tokenize(self, line, add_eos=False, add_double_eos=False):
            return self.tokenizer.encode(line)

        def convert_to_tensor(self, symbols):
            return torch.LongTensor(symbols)


    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/data_utils.py:29-116 ---
    class LMOrderedIterator(object):
        def __init__(self, data, bsz, bptt, device='cpu', mem_len=None, ext_len=None, warmup=True):
            """
                data -- LongTensor -- the LongTensor is strictly ordered
            """
            self.bsz = bsz
            self.bptt = bptt
            self.ext_len = ext_len if ext_len is not None else 0
            self.mem_len = mem_len
            self.warmup = warmup

            self.device = device

            # Work out how cleanly we can divide the dataset into bsz parts.
            n_step = data.size(0) // bsz

            # Trim off any extra elements that wouldn't cleanly fit (remainders).
            data = data[:n_step * bsz]

            # Evenly divide the data across the bsz batches.
            self.data = data.view(bsz, -1).t().contiguous().pin_memory()

            if mem_len and warmup:
                self.warmup_batches = (mem_len + bptt - 1) // bptt
                self.warmup_elems = self.warmup_batches * bptt

                warmup_data = self.data.roll((self.warmup_elems, 1), (0, 1))[:self.warmup_elems]
                self.data = torch.cat((warmup_data, self.data))

            # Partition data for DistributedDataParallel
            world_size = utils.distributed.get_world_size()
            rank = utils.distributed.get_rank()
            self.data = self.data.chunk(world_size, dim=1)[rank]

            # Number of mini-batches
            self.n_batch = (self.data.size(0) + self.bptt - 1) // self.bptt

            self.last_iter = None

        def roll(self, seed):
            rng = torch.Generator()
            rng.manual_seed(seed)
            for i in range(self.data.size(1)):
                row = self.data[:, i]
                shift = torch.randint(0, self.data.size(0), (1,), generator=rng)
                row = torch.cat((row[shift:], row[:shift]))
                self.data[:, i] = row

        def get_batch(self, i, bptt=None):
            if bptt is None:
                bptt = self.bptt

            seq_len = min(bptt, self.data.size(0) - 1 - i)

            end_idx = i + seq_len
            beg_idx = max(0, i - self.ext_len)

            data = self.data[beg_idx:end_idx].to(self.device, non_blocking=True)
            target = self.data[i+1:i+1+seq_len].to(self.device, non_blocking=True)

            if self.mem_len and self.warmup:
                warm = i >= self.warmup_elems
            else:
                warm = True

            return data, target, seq_len, warm

        def get_fixlen_iter(self, start=0):
            if start != 0:
                start += self.bptt
            for i in range(start, self.data.size(0) - 1, self.bptt):
                self.last_iter = i
                yield self.get_batch(i)

        def get_varlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
            max_len = self.bptt + max_deviation * std
            i = start
            while True:
                bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
                bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))
                data, target, seq_len = self.get_batch(i, bptt)
                i += seq_len
                yield data, target, seq_len
                if i >= self.data.size(0) - 2:
                    break

        def __iter__(self):
            return self.get_fixlen_iter()


    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/data_utils.py:119-196 ---
    class LMShuffledIterator(object):
        def __init__(self, data, bsz, bptt, device='cpu', ext_len=None, shuffle=False):
            """
                data -- list[LongTensor] -- there is no order among the LongTensors
            """
            self.data = data

            self.bsz = bsz
            self.bptt = bptt
            self.ext_len = ext_len if ext_len is not None else 0

            self.device = device
            self.shuffle = shuffle

        def get_sent_stream(self):
            # index iterator
            epoch_indices = np.random.permutation(len(self.data)) if self.shuffle \
                else np.array(range(len(self.data)))

            # sentence iterator
            for idx in epoch_indices:
                yield self.data[idx]

        def stream_iterator(self, sent_stream):
            # streams for each data in the batch
            streams = [None] * self.bsz

            data = torch.LongTensor(self.bptt, self.bsz)
            target = torch.LongTensor(self.bptt, self.bsz)

            n_retain = 0

            while True:
                # data   : [n_retain+bptt x bsz]
                # target : [bptt x bsz]
                data[n_retain:].fill_(-1)
                target.fill_(-1)

                valid_batch = True

                for i in range(self.bsz):
                    n_filled = 0
                    try:
                        while n_filled < self.bptt:
                            if streams[i] is None or len(streams[i]) <= 1:
                                streams[i] = next(sent_stream)
                            # number of new tokens to fill in
                            n_new = min(len(streams[i]) - 1, self.bptt - n_filled)
                            # first n_retain tokens are retained from last batch
                            data[n_retain+n_filled:n_retain+n_filled+n_new, i] = \
                                streams[i][:n_new]
                            target[n_filled:n_filled+n_new, i] = \
                                streams[i][1:n_new+1]
                            streams[i] = streams[i][n_new:]
                            n_filled += n_new
                    except StopIteration:
                        valid_batch = False
                        break

                if not valid_batch:
                    return

                data = data.to(self.device)
                target = target.to(self.device)

                yield data, target, self.bptt

                n_retain = min(data.size(0), self.ext_len)
                if n_retain > 0:
                    data[:n_retain] = data[-n_retain:]
                data.resize_(n_retain + self.bptt, data.size(1))

        def __iter__(self):
            # sent_stream is an iterator
            sent_stream = self.get_sent_stream()

            for batch in self.stream_iterator(sent_stream):
                yield batch


    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/data_utils.py:199-229 ---
    class LMMultiFileIterator(LMShuffledIterator):
        def __init__(self, paths, vocab, bsz, bptt, device='cpu', ext_len=None,
                     shuffle=False):

            self.paths = paths
            self.vocab = vocab

            self.bsz = bsz
            self.bptt = bptt
            self.ext_len = ext_len if ext_len is not None else 0

            self.device = device
            self.shuffle = shuffle

        def get_sent_stream(self, path):
            sents = self.vocab.encode_file(path, add_double_eos=True)
            if self.shuffle:
                np.random.shuffle(sents)
            sent_stream = iter(sents)

            return sent_stream

        def __iter__(self):
            if self.shuffle:
                np.random.shuffle(self.paths)

            for path in self.paths:
                # sent_stream is an iterator
                sent_stream = self.get_sent_stream(path)
                for batch in self.stream_iterator(sent_stream):
                    yield batch


    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/data_utils.py:232-292 ---
    class Corpus(object):
        def __init__(self, path, dataset, vocab, *args, **kwargs):
            self.dataset = dataset
            if vocab == 'word':
                self.vocab = Vocab(*args, **kwargs)
            elif vocab == 'bpe':
                self.vocab = OpenAIVocab()
            else:
                raise RuntimeError('Unsupported vocab')

            if self.dataset in ['ptb', 'wt2', 'enwik8', 'text8']:
                self.vocab.count_file(os.path.join(path, 'train.txt'))
                self.vocab.count_file(os.path.join(path, 'valid.txt'))
                self.vocab.count_file(os.path.join(path, 'test.txt'))
            elif self.dataset == 'wt103':
                self.vocab.count_file(os.path.join(path, 'train.txt'))
            elif self.dataset == 'lm1b':
                train_path_pattern = os.path.join(
                    path, '1-billion-word-language-modeling-benchmark-r13output',
                    'training-monolingual.tokenized.shuffled', 'news.en-*')
                train_paths = glob.glob(train_path_pattern)
                # the vocab will load from file when build_vocab() is called

            self.vocab.build_vocab()

            if self.dataset in ['ptb', 'wt2', 'wt103']:
                self.train = self.vocab.encode_file(
                    os.path.join(path, 'train.txt'), ordered=True)
                self.valid = self.vocab.encode_file(
                    os.path.join(path, 'valid.txt'), ordered=True)
                self.test = self.vocab.encode_file(
                    os.path.join(path, 'test.txt'), ordered=True)
            elif self.dataset in ['enwik8', 'text8']:
                self.train = self.vocab.encode_file(
                    os.path.join(path, 'train.txt'), ordered=True, add_eos=False)
                self.valid = self.vocab.encode_file(
                    os.path.join(path, 'valid.txt'), ordered=True, add_eos=False)
                self.test = self.vocab.encode_file(
                    os.path.join(path, 'test.txt'), ordered=True, add_eos=False)
            elif self.dataset == 'lm1b':
                self.train = train_paths
                self.valid = self.vocab.encode_file(
                    os.path.join(path, 'valid.txt'), ordered=False, add_double_eos=True)
                self.test = self.vocab.encode_file(
                    os.path.join(path, 'test.txt'), ordered=False, add_double_eos=True)

        def get_iterator(self, split, *args, **kwargs):
            if split == 'train':
                if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8']:
                    data_iter = LMOrderedIterator(self.train, *args, **kwargs)
                elif self.dataset == 'lm1b':
                    kwargs['shuffle'] = True
                    data_iter = LMMultiFileIterator(self.train, self.vocab, *args, **kwargs)
            elif split in ['valid', 'test']:
                data = self.valid if split == 'valid' else self.test
                if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8']:
                    data_iter = LMOrderedIterator(data, *args, **kwargs)
                elif self.dataset == 'lm1b':
                    data_iter = LMShuffledIterator(data, *args, **kwargs)

            return data_iter


    # --- licence header: PyTorch/LanguageModeling/Transformer-XL/pytorch/utils/log_uniform_sampler.py:1-0 ---



    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/utils/log_uniform_sampler.py:6-46 ---
    class LogUniformSampler(object):
        def __init__(self, range_max, n_sample):
            """
            Reference : https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/candidate_sampling_ops.py
                `P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)`

            expected count can be approximated by 1 - (1 - p)^n
            and we use a numerically stable version -expm1(num_tries * log1p(-p))

            Our implementation fixes num_tries at 2 * n_sample, and the actual #samples will vary from run to run
            """
            with torch.no_grad():
                self.range_max = range_max
                log_indices = torch.arange(1., range_max+2., 1.).log_()
                self.dist = (log_indices[1:] - log_indices[:-1]) / log_indices[-1]
                # print('P', self.dist.numpy().tolist()[-30:])

                self.log_q = (- (-self.dist.double().log1p_() * 2 * n_sample).expm1_()).log_().float()

            self.n_sample = n_sample

        def sample(self, labels):
            """
                labels: [b1, b2]
            Return
                true_log_probs: [b1, b2]
                samp_log_probs: [n_sample]
                neg_samples: [n_sample]
            """

            # neg_samples = torch.empty(0).long()
            n_sample = self.n_sample
            n_tries = 2 * n_sample

            with torch.no_grad():
                neg_samples = torch.multinomial(self.dist, n_tries, replacement=True).unique()
                device = labels.device
                neg_samples = neg_samples.to(device)
                true_log_probs = self.log_q[labels].to(device)
                samp_log_probs = self.log_q[neg_samples].to(device)
                return true_log_probs, samp_log_probs, neg_samples


    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/utils/log_uniform_sampler.py:48-79 ---
    def sample_logits(embedding, bias, labels, inputs, sampler):
        """
            embedding: an nn.Embedding layer
            bias: [n_vocab]
            labels: [b1, b2]
            inputs: [b1, b2, n_emb]
            sampler: you may use a LogUniformSampler
        Return
            logits: [b1, b2, 1 + n_sample]
        """
        true_log_probs, samp_log_probs, neg_samples = sampler.sample(labels)
        n_sample = neg_samples.size(0)
        b1, b2 = labels.size(0), labels.size(1)
        all_ids = torch.cat([labels.view(-1), neg_samples])
        all_w = embedding(all_ids)
        true_w = all_w[: -n_sample].view(b1, b2, -1)
        sample_w = all_w[- n_sample:].view(n_sample, -1)

        all_b = bias[all_ids]
        true_b = all_b[: -n_sample].view(b1, b2)
        sample_b = all_b[- n_sample:]

        hit = (labels[:, :, None] == neg_samples).detach()

        true_logits = torch.einsum('ijk,ijk->ij',
                                   true_w, inputs) + true_b - true_log_probs
        sample_logits = torch.einsum('lk,ijk->ijl',
                                     sample_w, inputs) + sample_b - samp_log_probs
        sample_logits.masked_fill_(hit, -1e30)
        logits = torch.cat([true_logits[:, :, None], sample_logits], -1)

        return logits


    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/utils/proj_adaptive_softmax.py:20-31 ---
    class OptionalParameterList(nn.ParameterList):
        def extra_repr(self):
            child_lines = []
            for k, p in self._parameters.items():
                if p is not None:
                    size_str = 'x'.join(str(size) for size in p.size())
                    device_str = '' if not p.is_cuda else ' (GPU {})'.format(p.get_device())
                    parastr = 'Parameter containing: [{} of size {}{}]'.format(
                        torch.typename(p), size_str, device_str)
                    child_lines.append('  (' + str(k) + '): ' + parastr)
            tmpstr = '\n'.join(child_lines)
            return tmpstr


    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/utils/proj_adaptive_softmax.py:34-208 ---
    class ProjectedAdaptiveLogSoftmax(nn.Module):
        def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1,
                     tie_projs=None, out_layers_weights=None, out_projs=None,
                     keep_order=False):
            super().__init__()

            self.n_token = n_token
            self.d_embed = d_embed
            self.d_proj = d_proj

            self.cutoffs = cutoffs + [n_token]
            self.cutoff_ends = [0] + self.cutoffs
            self.div_val = div_val

            self.shortlist_size = self.cutoffs[0]
            self.n_clusters = len(self.cutoffs) - 1
            self.head_size = self.shortlist_size + self.n_clusters

            self.tie_projs = tie_projs

            if self.n_clusters > 0:
                self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters, self.d_embed))
                self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))

            if not out_layers_weights:
                self.out_layers_weights = nn.ParameterList()
            else:
                self.out_layers_weights = out_layers_weights

            self.out_layers_biases = nn.ParameterList()

            self.shared_out_projs = out_projs
            self.out_projs = OptionalParameterList()

            if div_val == 1:
                if d_proj != d_embed:
                    for i in range(len(self.cutoffs)):
                        if tie_projs[i]:
                            self.out_projs.append(None)
                        else:
                            self.out_projs.append(
                                nn.Parameter(torch.zeros(d_proj, d_embed))
                            )
                else:
                    # self.out_projs = [None] * len(self.cutoffs)
                    self.out_projs.append(None)

                self.out_layers_biases.append(
                    nn.Parameter(torch.zeros(n_token))
                    )

                if not out_layers_weights:
                    self.out_layers_weights.append(
                        nn.Parameter(torch.zeros(n_token, d_embed))
                        )
            else:
                for i in range(len(self.cutoffs)):
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                    d_emb_i = d_embed // (div_val ** i)

                    if tie_projs[i]:
                        self.out_projs.append(None)
                    else:
                        self.out_projs.append(
                            nn.Parameter(torch.zeros(d_proj, d_emb_i))
                        )

                    self.out_layers_biases.append(
                        nn.Parameter(torch.zeros(r_idx - l_idx))
                        )
                    if not out_layers_weights:
                        self.out_layers_weights.append(
                            nn.Parameter(torch.zeros(r_idx - l_idx, d_emb_i))
                            )

            self.keep_order = keep_order

        def _compute_logit(self, hidden, weight, bias, proj):
            if proj is None:
                logit = F.linear(hidden, weight, bias=bias)
            else:
                logit = torch.einsum('bd,de,ev->bv', hidden, proj, weight.t())
                if bias is not None:
                    logit = logit + bias
            return logit

        def get_out_proj(self, i):
            if self.tie_projs[i]:
                if len(self.shared_out_projs) == 0:
                    return None
                elif len(self.shared_out_projs) == 1:
                    return self.shared_out_projs[0]
                else:
                    return self.shared_out_projs[i]
            else:
                return self.out_projs[i]

        def forward(self, hidden, target, keep_order=False):
            '''
                hidden :: [len*bsz x d_proj]
                target :: [len*bsz]
            '''

            if hidden.size(0) != target.size(0):
                raise RuntimeError('Input and target should have the same size '
                                   'in the batch dimension.')

            if self.n_clusters == 0:
                logit = self._compute_logit(hidden, self.out_layers_weights[0],
                                            self.out_layers_biases[0], self.get_out_proj(0))
                nll = -F.log_softmax(logit, dim=-1) \
                        .gather(1, target.unsqueeze(1)).squeeze(1)
            else:
                # construct weights and biases
                weights, biases = [], []
                for i in range(len(self.cutoffs)):
                    if self.div_val == 1:
                        l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                        weight_i = self.out_layers_weights[0][l_idx:r_idx]
                        bias_i = self.out_layers_biases[0][l_idx:r_idx]
                    else:
                        weight_i = self.out_layers_weights[i]
                        bias_i = self.out_layers_biases[i]

                    if i == 0:
                        weight_i = torch.cat(
                            [weight_i, self.cluster_weight], dim=0)
                        bias_i = torch.cat(
                            [bias_i, self.cluster_bias], dim=0)

                    weights.append(weight_i)
                    biases.append(bias_i)

                head_weight, head_bias, head_proj = weights[0], biases[0], self.get_out_proj(0)

                head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
                head_logprob = F.log_softmax(head_logit, dim=1)

                nll = torch.zeros_like(target, dtype=hidden.dtype, device=hidden.device)

                offset = 0
                cutoff_values = [0] + self.cutoffs
                for i in range(len(cutoff_values) - 1):
                    l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]

                    mask_i = (target >= l_idx) & (target < r_idx)
                    indices_i = mask_i.nonzero(as_tuple=False).squeeze()

                    if indices_i.numel() == 0:
                        continue

                    target_i = target.index_select(0, indices_i) - l_idx
                    head_logprob_i = head_logprob.index_select(0, indices_i)

                    if i == 0:
                        logprob_i = head_logprob_i.gather(1, target_i[:, None]).squeeze(1)
                    else:
                        weight_i, bias_i, proj_i = weights[i], biases[i], self.get_out_proj(i)

                        hidden_i = hidden.index_select(0, indices_i)

                        tail_logit_i = self._compute_logit(hidden_i, weight_i, bias_i, proj_i)
                        tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)

                        logprob_i = head_logprob_i[:, -i] \
                            + tail_logprob_i.gather(1, target_i[:, None]).squeeze(1)

                    if self.keep_order or keep_order:
                        nll.index_copy_(0, indices_i, -logprob_i)
                    else:
                        nll[offset:offset+logprob_i.size(0)].copy_(-logprob_i)

                    offset += logprob_i.size(0)

            return nll


    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/mem_transformer.py:24-26 ---
    @torch.jit.script
    def add_and_scale(tensor1, tensor2, alpha: float):
        return alpha * (tensor1 + tensor2)


    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/mem_transformer.py:29-45 ---
    class PositionalEmbedding(nn.Module):
        def __init__(self, demb):
            super(PositionalEmbedding, self).__init__()

            self.demb = demb

            inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
            self.register_buffer('inv_freq', inv_freq)

        def forward(self, pos_seq, bsz=None):
            sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
            pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

            if bsz is not None:
                return pos_emb[:, None, :].expand(-1, bsz, -1)
            else:
                return pos_emb[:, None, :]


    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/mem_transformer.py:48-81 ---
    class PositionwiseFF(nn.Module):
        def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
            super(PositionwiseFF, self).__init__()

            self.d_model = d_model
            self.d_inner = d_inner
            self.dropout = dropout

            self.CoreNet = nn.Sequential(
                nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(d_inner, d_model),
                nn.Dropout(dropout),
            )

            self.layer_norm = nn.LayerNorm(d_model)

            self.pre_lnorm = pre_lnorm

        def forward(self, inp):
            if self.pre_lnorm:
                # layer normalization + positionwise feed-forward
                core_out = self.CoreNet(self.layer_norm(inp))

                # residual connection
                output = core_out + inp
            else:
                # positionwise feed-forward
                core_out = self.CoreNet(inp)

                # residual connection + layer normalization
                output = self.layer_norm(inp + core_out)

            return output


    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/mem_transformer.py:84-156 ---
    class MultiHeadAttn(nn.Module):
        def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                     pre_lnorm=False):
            super(MultiHeadAttn, self).__init__()

            self.n_head = n_head
            self.d_model = d_model
            self.d_head = d_head
            self.dropout = dropout

            self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
            self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

            self.drop = nn.Dropout(dropout)
            self.dropatt = nn.Dropout(dropatt)
            self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

            self.layer_norm = nn.LayerNorm(d_model)

            self.scale = 1 / (d_head ** 0.5)

            self.pre_lnorm = pre_lnorm

        def forward(self, h, attn_mask=None, mems=None):
            # multihead attention
            # [hlen x bsz x n_head x d_head]

            if mems is not None:
                c = torch.cat([mems, h], 0)
            else:
                c = h

            if self.pre_lnorm:
                # layer normalization
                c = self.layer_norm(c)

            head_q = self.q_net(h)
            head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

            head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
            head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
            head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

            # [bsz x n_head x qlen x klen]
            attn_score = torch.einsum('ibnd,jbnd->bnij', head_q, head_k)
            attn_score.mul_(self.scale)
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    attn_score.masked_fill_(attn_mask[None, None, :, :], -float('inf'))
                elif attn_mask.dim() == 3:
                    attn_score.masked_fill_(attn_mask[:, None, :, :], -float('inf'))

            # [bsz x qlen x klen x n_head]
            attn_prob = F.softmax(attn_score, dim=3)
            attn_prob = self.dropatt(attn_prob)

            # [bsz x n_head x qlen x klen] * [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
            attn_vec = torch.einsum('bnij,jbnd->ibnd', attn_prob, head_v)
            attn_vec = attn_vec.contiguous().view(
                attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

            # linear projection
            attn_out = self.o_net(attn_vec)
            attn_out = self.drop(attn_out)

            if self.pre_lnorm:
                # residual connection
                output = h + attn_out
            else:
                # residual connection + layer normalization
                output = self.layer_norm(h + attn_out)

            return output


    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/mem_transformer.py:159-226 ---
    class RelMultiHeadAttn(nn.Module):
        def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                     tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False):
            super(RelMultiHeadAttn, self).__init__()

            self.n_head = n_head
            self.d_model = d_model
            self.d_head = d_head
            self.dropout = dropout

            self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

            self.drop = nn.Dropout(dropout)
            self.dropatt = nn.Dropout(dropatt)
            self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

            self.layer_norm = nn.LayerNorm(d_model)

            self.scale = 1 / (d_head ** 0.5)

            self.pre_lnorm = pre_lnorm

        def _parallelogram_mask(self, h, w, left=False):
            mask = torch.ones((h, w)).byte()
            m = min(h, w)
            mask[:m, :m] = torch.triu(mask[:m, :m])
            mask[-m:, -m:] = torch.tril(mask[-m:, -m:])

            if left:
                return mask.bool()
            else:
                return mask.flip(0).bool()

        def _shift(self, x, qlen, klen, mask, left=False):
            if qlen > 1:
                zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                       device=x.device, dtype=x.dtype)
            else:
                zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

            if left:
                mask = mask.flip(1)
                x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
            else:
                x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

            x = x_padded.masked_select(mask[:, :, None, None]) \
                        .view(qlen, klen, x.size(2), x.size(3))

            return x

        def _rel_shift(self, x, zero_triu=False):
            zero_pad = torch.zeros((x.size(0), x.size(1), x.size(2), 1),
                                   device=x.device, dtype=x.dtype)
            x_padded = torch.cat([zero_pad, x], dim=3)

            x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))

            x = x_padded.narrow(2, 1, x_padded.size(2) - 1).view_as(x)

            if zero_triu:
                ones = torch.ones((x.size(2), x.size(3)))
                x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

            return x

        def forward(self, w, r, attn_mask=None, mems=None):
            raise NotImplementedError


    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/mem_transformer.py:229-305 ---
    class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
        def __init__(self, *args, **kwargs):
            super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

            self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

        def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
            qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

            if mems is not None:
                cat = torch.cat([mems, w], 0)
                if self.pre_lnorm:
                    w_heads = self.qkv_net(self.layer_norm(cat))
                else:
                    w_heads = self.qkv_net(cat)
                r_head_k = self.r_net(r)

                w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
                w_head_q = w_head_q[-qlen:]
            else:
                if self.pre_lnorm:
                    w_heads = self.qkv_net(self.layer_norm(w))
                else:
                    w_heads = self.qkv_net(w)
                r_head_k = self.r_net(r)

                w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            klen = w_head_k.size(0)

            w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
            w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)  # klen x bsz x n_head x d_head
            w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)  # klen x bsz x n_head x d_head

            r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)       # qlen x n_head x d_head

            # compute attention score
            rw_head_q = w_head_q + r_w_bias                                # qlen x bsz x n_head x d_head
            AC = torch.einsum('ibnd,jbnd->bnij', rw_head_q, w_head_k)      # bsz x n_head x qlen x klen

            rr_head_q = w_head_q + r_r_bias
            BD = torch.einsum('ibnd,jnd->bnij', rr_head_q, r_head_k)       # bsz x n_head x qlen x klen
            BD = self._rel_shift(BD)

            # [bsz x n_head x qlen x klen]
            attn_score = add_and_scale(AC, BD, self.scale)

            # compute attention probability
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    attn_score.masked_fill_(attn_mask[None, None, :, :], -float('inf'))
                elif attn_mask.dim() == 3:
                    attn_score.masked_fill_(attn_mask[:, None, :, :], -float('inf'))

            # [bsz x n_head x qlen x klen]
            attn_prob = F.softmax(attn_score, dim=3)
            attn_prob = self.dropatt(attn_prob)

            # compute attention vector
            attn_vec = torch.einsum('bnij,jbnd->ibnd', attn_prob, w_head_v)

            # [qlen x bsz x n_head x d_head]
            attn_vec = attn_vec.contiguous().view(
                attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

            # linear projection
            attn_out = self.o_net(attn_vec)
            attn_out = self.drop(attn_out)

            if self.pre_lnorm:
                # residual connection
                output = w + attn_out
            else:
                # residual connection + layer normalization
                output = self.layer_norm(w + attn_out)

            return output


    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/mem_transformer.py:308-392 ---
    class RelLearnableMultiHeadAttn(RelMultiHeadAttn):
        def __init__(self, *args, **kwargs):
            super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None):
            # r_emb: [klen, n_head, d_head], used for term B
            # r_w_bias: [n_head, d_head], used for term C
            # r_bias: [klen, n_head], used for term D

            qlen, bsz = w.size(0), w.size(1)

            if mems is not None:
                cat = torch.cat([mems, w], 0)
                if self.pre_lnorm:
                    w_heads = self.qkv_net(self.layer_norm(cat))
                else:
                    w_heads = self.qkv_net(cat)
                w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

                w_head_q = w_head_q[-qlen:]
            else:
                if self.pre_lnorm:
                    w_heads = self.qkv_net(self.layer_norm(w))
                else:
                    w_heads = self.qkv_net(w)
                w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            klen = w_head_k.size(0)

            w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
            w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
            w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

            if klen > r_emb.size(0):
                r_emb_pad = r_emb[0:1].expand(klen-r_emb.size(0), -1, -1)
                r_emb = torch.cat([r_emb_pad, r_emb], 0)
                r_bias_pad = r_bias[0:1].expand(klen-r_bias.size(0), -1)
                r_bias = torch.cat([r_bias_pad, r_bias], 0)
            else:
                r_emb = r_emb[-klen:]
                r_bias = r_bias[-klen:]

            r_bias = r_bias.t()

            # compute attention score
            rw_head_q = w_head_q + r_w_bias[None]                      # qlen x bsz x n_head x d_head

            AC = torch.einsum('ibnd,jbnd->bnij', rw_head_q, w_head_k)  # bsz x n_head x qlen x klen
            B_ = torch.einsum('ibnd,jnd->bnij', w_head_q, r_emb)       # bsz x n_head x qlen x klen
            D_ = r_bias[None, :, None, :]                              # 1   x n_head x    1 x klen
            BD = self._rel_shift(B_ + D_)

            # [bsz x qlen x klen x n_head]
            attn_score = add_and_scale(AC, BD, self.scale)

            # compute attention probability
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    attn_score.masked_fill_(attn_mask[None, None, :, :], -float('inf'))
                elif attn_mask.dim() == 3:
                    attn_score.masked_fill_(attn_mask[:, None, :, :], -float('inf'))

            # [bsz x n_head x qlen x klen]
            attn_prob = F.softmax(attn_score, dim=3)
            attn_prob = self.dropatt(attn_prob)

            # compute attention vector
            attn_vec = torch.einsum('bnij,jbnd->ibnd', attn_prob, w_head_v)

            # [qlen x bsz x n_head x d_head]
            attn_vec = attn_vec.contiguous().view(
                attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

            # linear projection
            attn_out = self.o_net(attn_vec)
            attn_out = self.drop(attn_out)

            if self.pre_lnorm:
                # residual connection
                output = w + attn_out
            else:
                # residual connection + layer normalization
                output = self.layer_norm(w + attn_out)

            return output


    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/mem_transformer.py:395-409 ---
    class DecoderLayer(nn.Module):
        def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
            super(DecoderLayer, self).__init__()

            self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
            self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                         pre_lnorm=kwargs.get('pre_lnorm'))

        def forward(self, dec_inp, dec_attn_mask=None, mems=None):

            output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask,
                                   mems=mems)
            output = self.pos_ff(output)

            return output


    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/mem_transformer.py:412-429 ---
    class RelLearnableDecoderLayer(nn.Module):
        def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                     **kwargs):
            super(RelLearnableDecoderLayer, self).__init__()

            self.dec_attn = RelLearnableMultiHeadAttn(n_head, d_model, d_head,
                                                      dropout, **kwargs)
            self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                         pre_lnorm=kwargs.get('pre_lnorm'))

        def forward(self, dec_inp, r_emb, r_w_bias, r_bias, dec_attn_mask=None, mems=None):

            output = self.dec_attn(dec_inp, r_emb, r_w_bias, r_bias,
                                   attn_mask=dec_attn_mask,
                                   mems=mems)
            output = self.pos_ff(output)

            return output


    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/mem_transformer.py:432-450 ---
    class RelPartialLearnableDecoderLayer(nn.Module):
        def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                     **kwargs):
            super(RelPartialLearnableDecoderLayer, self).__init__()

            self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                                                             d_head, dropout,
                                                             **kwargs)
            self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                         pre_lnorm=kwargs.get('pre_lnorm'))

        def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):

            output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                                   attn_mask=dec_attn_mask,
                                   mems=mems)
            output = self.pos_ff(output)

            return output


    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/mem_transformer.py:453-513 ---
    class AdaptiveEmbedding(nn.Module):
        def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1,
                     sample_softmax=False):
            super(AdaptiveEmbedding, self).__init__()

            self.n_token = n_token
            self.d_embed = d_embed

            self.cutoffs = cutoffs + [n_token]
            self.div_val = div_val
            self.d_proj = d_proj

            self.emb_scale = d_proj ** 0.5

            self.cutoff_ends = [0] + self.cutoffs

            self.emb_layers = nn.ModuleList()
            self.emb_projs = nn.ParameterList()
            if div_val == 1:
                self.emb_layers.append(
                    nn.Embedding(n_token, d_embed, sparse=(sample_softmax > 0))
                )
                if d_proj != d_embed:
                    self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed).zero_()))
            else:
                for i in range(len(self.cutoffs)):
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                    d_emb_i = d_embed // (div_val ** i)
                    self.emb_layers.append(nn.Embedding(r_idx-l_idx, d_emb_i))
                    self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_emb_i).zero_()))

        def forward(self, inp):
            if self.div_val == 1:
                embed = self.emb_layers[0](inp)
                if self.d_proj != self.d_embed:
                    embed = F.linear(embed, self.emb_projs[0])
            else:
                param = next(self.parameters())
                inp_flat = inp.view(-1)
                emb_flat = torch.zeros([inp_flat.size(0), self.d_proj],
                                       dtype=param.dtype, device=param.device)
                for i in range(len(self.cutoffs)):
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                    mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                    indices_i = mask_i.nonzero(as_tuple=False).squeeze()

                    if indices_i.numel() == 0:
                        continue

                    inp_i = inp_flat.index_select(0, indices_i) - l_idx
                    emb_i = self.emb_layers[i](inp_i)
                    emb_i = F.linear(emb_i, self.emb_projs[i]).to(emb_flat.dtype)

                    emb_flat.index_copy_(0, indices_i, emb_i)

                embed = emb_flat.view(*inp.size(), self.d_proj)

            embed.mul_(self.emb_scale)

            return embed


    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/mem_transformer.py:516-800 ---
    class MemTransformerLM(nn.Module):
        def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                     dropout, dropatt, dtype, tie_weight=True, d_embed=None,
                     div_val=1, tie_projs=[False], pre_lnorm=False,
                     tgt_len=None, ext_len=None, mem_len=None,
                     cutoffs=[], adapt_inp=False,
                     same_length=False, attn_type=0, clamp_len=-1,
                     sample_softmax=-1):
            super(MemTransformerLM, self).__init__()
            self.n_token = n_token

            d_embed = d_model if d_embed is None else d_embed
            self.d_embed = d_embed
            self.d_model = d_model
            self.n_head = n_head
            self.d_head = d_head

            self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs,
                                              div_val=div_val)

            self.drop = nn.Dropout(dropout)

            self.tie_weight = tie_weight
            self.tie_projs = tie_projs
            self.div_val = div_val

            self.n_layer = n_layer

            self.tgt_len = tgt_len
            self.mem_len = mem_len
            self.ext_len = ext_len
            self.max_klen = tgt_len + ext_len + mem_len

            self.attn_type = attn_type

            self.layers = nn.ModuleList()
            # the default attention
            if attn_type == 0:
                for i in range(n_layer):
                    self.layers.append(
                        RelPartialLearnableDecoderLayer(
                            n_head, d_model, d_head, d_inner, dropout,
                            tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                            dropatt=dropatt, pre_lnorm=pre_lnorm)
                    )
            # learnable embeddings
            elif attn_type == 1:
                for i in range(n_layer):
                    self.layers.append(
                        RelLearnableDecoderLayer(
                            n_head, d_model, d_head, d_inner, dropout,
                            tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                            dropatt=dropatt, pre_lnorm=pre_lnorm)
                    )
            # absolute embeddings
            elif attn_type in [2, 3]:
                for i in range(n_layer):
                    self.layers.append(
                        DecoderLayer(
                            n_head, d_model, d_head, d_inner, dropout,
                            dropatt=dropatt, pre_lnorm=pre_lnorm)
                    )

            self.sample_softmax = sample_softmax
            # use sampled softmax
            if sample_softmax > 0:
                self.out_layer = nn.Linear(d_model, n_token)
                self.tie_weight = tie_weight
                self.sampler = LogUniformSampler(n_token, sample_softmax)

            # use adaptive softmax (including standard softmax)
            else:
                if tie_weight:
                    emb_layers = [i.weight for i in self.word_emb.emb_layers]
                else:
                    emb_layers = None

                emb_projs = self.word_emb.emb_projs

                self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model,
                                                        cutoffs, div_val=div_val,
                                                        tie_projs=tie_projs,
                                                        out_projs=emb_projs,
                                                        out_layers_weights=emb_layers)


            self.same_length = same_length
            self.clamp_len = clamp_len

            self._create_params()

        def backward_compatible(self):
            self.sample_softmax = -1

        def _create_params(self):
            # default attention
            if self.attn_type == 0:
                self.pos_emb = PositionalEmbedding(self.d_model)
                self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head).zero_())
                self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head).zero_())
            # learnable
            elif self.attn_type == 1:
                self.r_emb = nn.Parameter(torch.Tensor(
                        self.n_layer, self.max_klen, self.n_head, self.d_head).zero_())
                self.r_w_bias = nn.Parameter(torch.Tensor(
                        self.n_layer, self.n_head, self.d_head).zero_())
                self.r_bias = nn.Parameter(torch.Tensor(
                        self.n_layer, self.max_klen, self.n_head).zero_())
            # absolute standard
            elif self.attn_type == 2:
                self.pos_emb = PositionalEmbedding(self.d_model)
            # absolute deeper SA
            elif self.attn_type == 3:
                self.r_emb = nn.Parameter(torch.Tensor(
                        self.n_layer, self.max_klen, self.d_model).zero_())

        def reset_length(self, tgt_len, ext_len, mem_len):
            if tgt_len < 1:
                raise RuntimeError(f'tgt_len should be >= 1, but got {tgt_len}')
            if ext_len < 0:
                raise RuntimeError(f'ext_len should be >= 0, but got {ext_len}')
            if mem_len < 0:
                raise RuntimeError(f'mem_len should be >= 0, but got {mem_len}')
            self.tgt_len = tgt_len
            self.mem_len = mem_len
            self.ext_len = ext_len

        def init_mems(self):
            if self.mem_len > 0:
                param = next(self.parameters())
                mems = torch.empty(self.n_layer, 0, dtype=param.dtype,
                                   device=param.device)
                return mems
            else:
                return None

        def _update_mems(self, hids, mems, qlen, mlen):
            # does not deal with None
            if mems is None:
                return None

            # mems is not None
            assert len(hids) == len(mems), 'len(hids) != len(mems)'

            # There are `mlen + qlen` steps that can be cached into mems
            # For the next step, the last `ext_len` of the `qlen` tokens
            # will be used as the extended context. Hence, we only cache
            # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
            # to `mlen + qlen - self.ext_len`.
            with torch.no_grad():
                stacked = torch.stack(hids)
                if (
                    self.mem_len == self.tgt_len
                    and self.ext_len == 0
                    and stacked.size(1) == self.mem_len
                ):
                    new_mems = stacked.detach()
                else:
                    end_idx = mlen + max(0, qlen - self.ext_len)
                    beg_idx = max(0, end_idx - self.mem_len)
                    if mems.numel():
                        cat = torch.cat([mems, stacked], dim=1)
                    else:
                        cat = stacked
                    new_mems = cat[:, beg_idx:end_idx].detach()

            return new_mems

        def _forward(self, dec_inp, mems=None):
            qlen, bsz = dec_inp.size()

            word_emb = self.word_emb(dec_inp)

            mlen = mems[0].size(0) if mems is not None else 0
            klen = mlen + qlen
            if self.same_length:
                all_ones = word_emb.new_ones(qlen, klen)
                mask_len = klen - self.mem_len - 1
                if mask_len > 0:
                    mask_shift_len = qlen - mask_len
                else:
                    mask_shift_len = qlen
                dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                                 + torch.tril(all_ones, -mask_shift_len)).bool()
            else:
                dec_attn_mask = torch.triu(
                    word_emb.new_ones(qlen, klen), diagonal=1+mlen).bool()

            hids = []
            # default
            if self.attn_type == 0:
                pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device,
                                       dtype=word_emb.dtype)
                if self.clamp_len > 0:
                    pos_seq.clamp_(max=self.clamp_len)
                pos_emb = self.pos_emb(pos_seq)

                core_out = self.drop(word_emb)
                pos_emb = self.drop(pos_emb)

                for i, layer in enumerate(self.layers):
                    hids.append(core_out.detach())
                    mems_i = None if mems is None else mems[i]
                    core_out = layer(core_out, pos_emb, self.r_w_bias,
                                     self.r_r_bias, dec_attn_mask=dec_attn_mask,
                                     mems=mems_i)
            # learnable
            elif self.attn_type == 1:
                core_out = self.drop(word_emb)
                for i, layer in enumerate(self.layers):
                    hids.append(core_out.detach())
                    if self.clamp_len > 0:
                        r_emb = self.r_emb[i][-self.clamp_len:]
                        r_bias = self.r_bias[i][-self.clamp_len:]
                    else:
                        r_emb, r_bias = self.r_emb[i], self.r_bias[i]

                    mems_i = None if mems is None else mems[i]
                    core_out = layer(core_out, r_emb, self.r_w_bias[i],
                                     r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
            # absolute
            elif self.attn_type == 2:
                pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
                                       dtype=word_emb.dtype)
                if self.clamp_len > 0:
                    pos_seq.clamp_(max=self.clamp_len)
                pos_emb = self.pos_emb(pos_seq)

                core_out = self.drop(word_emb + pos_emb[-qlen:])

                for i, layer in enumerate(self.layers):
                    hids.append(core_out.detach())
                    mems_i = None if mems is None else mems[i]
                    if mems_i is not None and len(mems_i) and i == 0:
                        mems_i += pos_emb[:mlen]
                    core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                     mems=mems_i)
            elif self.attn_type == 3:
                core_out = self.drop(word_emb)

                for i, layer in enumerate(self.layers):
                    hids.append(core_out.detach())
                    mems_i = None if mems is None else mems[i]
                    if mems_i is not None and len(mems_i) and mlen > 0:
                        cur_emb = self.r_emb[i][:-qlen]
                        cur_size = cur_emb.size(0)
                        if cur_size < mlen:
                            cur_emb_pad = cur_emb[0:1].expand(mlen-cur_size, -1, -1)
                            cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                        else:
                            cur_emb = cur_emb[-mlen:]
                        mems_i += cur_emb.view(mlen, 1, -1)
                    core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

                    core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                     mems=mems_i)

            core_out = self.drop(core_out)

            new_mems = self._update_mems(hids, mems, qlen, mlen)

            return core_out, new_mems

        def forward(self, data, target, mems):
            # nn.DataParallel does not allow size(0) tensors to be broadcasted.
            # So, have to initialize size(0) mems inside the model forward.
            # Moreover, have to return new_mems to allow nn.DataParallel to piece
            # them together.
            if mems is None:
                mems = self.init_mems()

            tgt_len = target.size(0)
            hidden, new_mems = self._forward(data, mems=mems)

            pred_hid = hidden[-tgt_len:]
            if self.sample_softmax > 0 and self.training:
                assert self.tie_weight
                logit = sample_logits(self.word_emb, self.out_layer.bias, target,
                                      pred_hid, self.sampler)
                loss = -F.log_softmax(logit, -1)[:, :, 0]
            else:
                loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
                loss = loss.view(tgt_len, -1)

            return (loss, new_mems)


    # --- licence header: PyTorch/LanguageModeling/Transformer-XL/pytorch/lamb.py:1-38 ---
    # Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #       http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

    # MIT License
    #
    # Copyright (c) 2019 cybertronai
    #
    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in all
    # copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    # SOFTWARE.

    """Lamb optimizer."""


    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/lamb.py:43-146 ---
    class Lamb(Optimizer):
        r"""Implements Lamb algorithm.

        It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.

        Arguments:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            lr (float, optional): learning rate (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square (default: (0.9, 0.999))
            eps (float, optional): term added to the denominator to improve
                numerical stability (default: 1e-8)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            adam (bool, optional): always use trust ratio = 1, which turns this into
                Adam. Useful for comparison purposes.

        .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
            https://arxiv.org/abs/1904.00962
        """

        def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                     weight_decay=0, adam=False):
            if not 0.0 <= lr:
                raise ValueError("Invalid learning rate: {}".format(lr))
            if not 0.0 <= eps:
                raise ValueError("Invalid epsilon value: {}".format(eps))
            if not 0.0 <= betas[0] < 1.0:
                raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
            if not 0.0 <= betas[1] < 1.0:
                raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
            defaults = dict(lr=lr, betas=betas, eps=eps,
                            weight_decay=weight_decay)
            self.adam = adam
            super(Lamb, self).__init__(params, defaults)

        def step(self, closure=None):
            """Performs a single optimization step.

            Arguments:
                closure (callable, optional): A closure that reevaluates the model
                    and returns the loss.
            """
            loss = None
            if closure is not None:
                loss = closure()

            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError('Lamb does not support sparse gradients.')

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p.data)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p.data)

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = group['betas']

                    state['step'] += 1

                    # Decay the first and second moment running average coefficient
                    # m_t
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    # v_t
                    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                    # Paper v3 does not use debiasing.
                    # bias_correction1 = 1 - beta1 ** state['step']
                    # bias_correction2 = 1 - beta2 ** state['step']
                    # Apply bias to lr to avoid broadcast.
                    step_size = group['lr'] # * math.sqrt(bias_correction2) / bias_correction1

                    weight_norm = p.data.norm(p=2).clamp_(0, 10)

                    adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                    if group['weight_decay'] != 0:
                        adam_step.add_(group['weight_decay'], p.data)

                    adam_norm = adam_step.norm(p=2)

                    if weight_norm == 0.0 or adam_norm == 0.0:
                        trust_ratio = 1
                    else:
                        trust_ratio = weight_norm / (adam_norm + group['eps'])

                    state['weight_norm'] = weight_norm
                    state['adam_norm'] = adam_norm
                    state['trust_ratio'] = trust_ratio
                    if self.adam:
                        trust_ratio = 1

                    p.data.add_(-step_size * trust_ratio, adam_step)

            return loss


    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/lamb.py:149-167 ---
    @torch.jit.script
    def lamb_kernel(param, grad, exp_avg, exp_avg_sq, beta1: float,
                    beta2: float, step_size: float, eps: float, weight_decay: float):
        exp_avg = exp_avg * beta1 + (1 - beta1) * grad
        exp_avg_sq = exp_avg_sq * beta2 + (1 - beta2) * (grad * grad)

        adam_step = exp_avg / (exp_avg_sq.sqrt() + eps)
        adam_step = adam_step + weight_decay * param

        weight_norm = param.norm(p=2).clamp(0, 10)
        adam_norm = adam_step.norm(p=2)

        trust_ratio = weight_norm / (adam_norm + eps)
        trust_ratio = (weight_norm == 0.0) * 1.0 + (weight_norm != 0.0) * trust_ratio
        trust_ratio = (adam_norm == 0.0) * 1.0 + (adam_norm != 0.0) * trust_ratio
        trust_ratio = trust_ratio.float()

        param = param - step_size * trust_ratio * adam_step
        return param, exp_avg, exp_avg_sq


    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/lamb.py:170-251 ---
    class JITLamb(Optimizer):
        r"""Implements Lamb algorithm.

        It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.

        Arguments:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            lr (float, optional): learning rate (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square (default: (0.9, 0.999))
            eps (float, optional): term added to the denominator to improve
                numerical stability (default: 1e-8)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            adam (bool, optional): always use trust ratio = 1, which turns this into
                Adam. Useful for comparison purposes.

        .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
            https://arxiv.org/abs/1904.00962
        """

        def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                     weight_decay=0, adam=False):
            if not 0.0 <= lr:
                raise ValueError("Invalid learning rate: {}".format(lr))
            if not 0.0 <= eps:
                raise ValueError("Invalid epsilon value: {}".format(eps))
            if not 0.0 <= betas[0] < 1.0:
                raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
            if not 0.0 <= betas[1] < 1.0:
                raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
            defaults = dict(lr=lr, betas=betas, eps=eps,
                            weight_decay=weight_decay)
            self.adam = adam
            super().__init__(params, defaults)

        def step(self, closure=None):
            """Performs a single optimization step.

            Arguments:
                closure (callable, optional): A closure that reevaluates the model
                    and returns the loss.
            """
            loss = None
            if closure is not None:
                loss = closure()

            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError('Lamb does not support sparse gradients.')

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p.data)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p.data)

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = group['betas']

                    state['step'] += 1
                    step_size = group['lr']

                    param, exp_avg, exp_avg_sq = lamb_kernel(p.data, grad, exp_avg,
                                                             exp_avg_sq, beta1,
                                                             beta2, step_size,
                                                             group['eps'],
                                                             group['weight_decay'],
                                                             )
                    state['exp_avg'] = exp_avg
                    state['exp_avg_sq'] = exp_avg_sq
                    p.data = param

            return loss


    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/utils/exp_utils.py:30-56 ---
    class AverageMeter:
        """
        Computes and stores the average and current value
        """
        def __init__(self, warmup=0, keep=False):
            self.reset()
            self.warmup = warmup
            self.keep = keep

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0
            self.iters = 0
            self.vals = []

        def update(self, val, n=1):
            self.iters += 1
            self.val = val

            if self.iters > self.warmup:
                self.sum += val * n
                self.count += n
                self.avg = self.sum / self.count
                if self.keep:
                    self.vals.append(val)


    # --- licence header: PyTorch/LanguageModeling/Transformer-XL/pytorch/train.py:1-16 ---
    # coding: utf-8

    # Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #       http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.


    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/train.py:365-369 ---
    def init_weight(weight, args):
        if args.init == 'uniform':
            nn.init.uniform_(weight, -args.init_range, args.init_range)
        elif args.init == 'normal':
            nn.init.normal_(weight, 0.0, args.init_std)


    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/train.py:372-373 ---
    def init_bias(bias):
        nn.init.constant_(bias, 0.0)


    # --- verbatim: PyTorch/LanguageModeling/Transformer-XL/pytorch/train.py:376-417 ---
    def weights_init(m, args):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                init_weight(m.weight, args)
            if hasattr(m, 'bias') and m.bias is not None:
                init_bias(m.bias)
        elif classname.find('AdaptiveEmbedding') != -1:
            if hasattr(m, 'emb_projs'):
                for i in range(len(m.emb_projs)):
                    if m.emb_projs[i] is not None:
                        nn.init.normal_(m.emb_projs[i], 0.0, args.proj_init_std)
        elif classname.find('Embedding') != -1:
            if hasattr(m, 'weight'):
                init_weight(m.weight, args)
        elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
            if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
                init_weight(m.cluster_weight, args)
            if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
                init_bias(m.cluster_bias)
            if hasattr(m, 'out_projs'):
                for i in range(len(m.out_projs)):
                    if m.out_projs[i] is not None:
                        nn.init.normal_(m.out_projs[i], 0.0, args.proj_init_std)
            if hasattr(m, 'out_layers_weights'):
                for i in range(len(m.out_layers_weights)):
                    if m.out_layers_weights[i] is not None:
                        init_weight(m.out_layers_weights[i], args)
        elif classname.find('LayerNorm') != -1:
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, 1.0, args.init_std)
            if hasattr(m, 'bias') and m.bias is not None:
                init_bias(m.bias)
        elif classname.find('TransformerLM') != -1:
            if hasattr(m, 'r_emb'):
                init_weight(m.r_emb, args)
            if hasattr(m, 'r_w_bias'):
                init_weight(m.r_w_bias, args)
            if hasattr(m, 'r_r_bias'):
                init_weight(m.r_r_bias, args)
            if hasattr(m, 'r_bias'):
                init_bias(m.r_bias)

    import zipfile

    if not os.path.isdir("wikitext-103"):
        with zipfile.ZipFile("/data/wikitext-103-v1.zip") as z:
            z.extractall(".")
        for split in ("train", "valid", "test"):
            os.rename(os.path.join("wikitext-103", "wiki.%s.tokens" % split),
                      os.path.join("wikitext-103", "%s.txt" % split))

    datadir = os.path.join(".", "wikitext-103")
    torch.cuda.set_device(0)
    device = torch.device('cuda')
    np.random.seed(1111)
    torch.manual_seed(1111)

    corpus = Corpus(datadir, 'wt103', 'word', special=['<eos>'], lower_case=False)
    ntokens = len(corpus.vocab)
    tr_iter = corpus.get_iterator('train', batch_size, 192, device=device, ext_len=0)

    model = MemTransformerLM(
        n_token=ntokens, n_layer=16, n_head=8, d_model=512, d_head=64, d_inner=2048,
        dropout=0.1, dropatt=0.0, dtype=None, tie_weight=True, d_embed=512, div_val=1,
        tie_projs=[False], pre_lnorm=False, tgt_len=192, ext_len=0, mem_len=192, cutoffs=[],
        same_length=False, attn_type=0, clamp_len=-1, sample_softmax=-1)
    init_args = argparse.Namespace(init='normal', init_range=0.1, init_std=0.02, proj_init_std=0.01)
    model.apply(functools.partial(weights_init, args=init_args))
    model.word_emb.apply(functools.partial(weights_init, args=init_args))

    optimizer = JITLamb(model.parameters(), lr=0.0, weight_decay=0.0)
    model = model.to(device)
    scaler = torch.cuda.amp.GradScaler()

    mems = None
    train_step = 0
    train_throughput = AverageMeter(warmup=192 // 192 + 2)

    for epoch in itertools.count(start=1):
        tr_iter.roll(seed=1111 + epoch)
        model.train()
        target_tokens = 0
        log_step = 0
        log_start_time = time.time()

        for batch, (data, target, seq_len, _) in enumerate(tr_iter.get_fixlen_iter(start=0), start=1):
            log_step += 1
            target_tokens += target.numel()

            for param in model.parameters():
                param.grad = None

            data_i = data.contiguous()
            target_i = target.contiguous()
            with torch.cuda.amp.autocast(True):
                loss, mems = model(data_i, target_i, mems)
                loss = loss.float().mean().type_as(loss)
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            scaler.step(optimizer)
            scaler.update()

            train_step += 1
            if train_step < 1000:
                optimizer.param_groups[0]['lr'] = 0.0 * train_step / 1000

            if train_step % 10 == 0:
                elapsed = time.time() - log_start_time
                log_start_time = time.time()
                log_step = 0
                train_throughput.update(target_tokens / elapsed)
                target_tokens = 0

            if train_step == max_step:
                break

        if train_step == max_step:
            break

    return {"train_throughput": train_throughput.avg}


FUNC = train_transformer_xl
