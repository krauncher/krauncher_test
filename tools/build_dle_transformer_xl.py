"""Build the lambdalabs/transformer_xl_* samples from NVIDIA DeepLearningExamples.

Same procedure as build_dle_waveglow.py (see there): the model, optimizer,
data and vocabulary code is copied verbatim from the pinned commit, with the
licence headers; the text written by this builder between the sections is in
the "glue" parts. The task has two phases, data preparation and training.

    python tools/build_dle_transformer_xl.py dle    # rewrites the task files
    git status --short                              # nothing changed
"""

import sys
import textwrap
from pathlib import Path

from build_dle_waveglow import ROOT, fill, verbatim

SRC = "PyTorch/LanguageModeling/Transformer-XL"
P = "pytorch/"
PARTS = [
    ("licence", "NOTICE"),
    (P + "utils/distributed.py", ["barrier", "get_rank", "get_world_size", "sync_workers"]),
    ("glue", """\
# The package `utils` that the copied data code refers to.
utils = types.SimpleNamespace(distributed=types.SimpleNamespace(
    get_rank=get_rank, get_world_size=get_world_size, sync_workers=sync_workers))
"""),
    (P + "utils/vocabulary.py", ["Vocab", "OpenAIVocab"]),
    (P + "data_utils.py", ["LMOrderedIterator", "LMShuffledIterator", "LMMultiFileIterator", "Corpus"]),
    (P + "utils/log_uniform_sampler.py", ["LogUniformSampler", "sample_logits"]),
    (P + "utils/proj_adaptive_softmax.py", ["OptionalParameterList", "ProjectedAdaptiveLogSoftmax"]),
    (P + "mem_transformer.py", [
        "add_and_scale", "PositionalEmbedding", "PositionwiseFF", "MultiHeadAttn", "RelMultiHeadAttn",
        "RelPartialLearnableMultiHeadAttn", "RelLearnableMultiHeadAttn", "DecoderLayer",
        "RelLearnableDecoderLayer", "RelPartialLearnableDecoderLayer", "AdaptiveEmbedding",
        "MemTransformerLM"]),
    (P + "lamb.py", ["Lamb", "lamb_kernel", "JITLamb"]),
    (P + "utils/exp_utils.py", ["AverageMeter"]),
    (P + "train.py", ["init_weight", "init_bias", "weights_init"]),
]

MODELS = {
    "base": dict(title="Transformer-XL base", cfg_name="transformerxlbase", n_layer=16, d_model=512, n_head=8,
                 d_inner=2048, dropout=0.1, dropatt=0.0, warmup_step=1000, tgt_len=192, mem_len=192,
                 flags='--optim jitlamb --lr 0.0 --eta_min 0.001 --warmup_step 1000 --tgt_len 192 --mem_len 192\n'
                       '  --eval_tgt_len 192 --log_interval 10 --eval_interval 5000 --no_eval --roll --cuda --fp16',
                 optim="lamb.JITLamb (train.py:883-884)", optim_code="JITLamb(model.parameters(), lr=0.0, weight_decay=0.0)",
                 roll="        tr_iter.roll(seed=1111 + epoch)\n",
                 roll_doc="the token stream is rolled by a random offset per epoch (--roll, train.py:1029-1030)"),
    "large": dict(title="Transformer-XL large", cfg_name="transformerxllarge", n_layer=18, d_model=1024, n_head=16,
                  d_inner=4096, dropout=0.2, dropatt=0.2, warmup_step=16000, tgt_len=256, mem_len=256,
                  flags='--optim adam --lr 0.0 --warmup_step 16000 --tgt_len 256 --mem_len 256 --eval_tgt_len 128\n'
                        '  --eval_interval 5000 --no_eval --cuda --fp16',
                  optim="Adam (train.py:872-873)", optim_code="optim.Adam(model.parameters(), lr=0.0, weight_decay=0.0)",
                  roll="", roll_doc="no --roll: the token stream is read in order"),
}

# Samples: (model, batch size, steps, GPU config of the benchmark).
SAMPLES = [
    ("base", 104, 400, "config_pytorch_80GB.sh"),
    ("base", 64, 40, "config_pytorch_48GB.sh"),
    ("base", 24, 400, "config_pytorch_24GB.sh"),
    ("large", 48, 400, "config_pytorch_80GB.sh"),
    ("large", 32, 40, "config_pytorch_48GB.sh"),
    ("large", 8, 400, "config_pytorch_24GB.sh"),
]

IMPORTS = """\
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
"""

ARCHIVE = "https://wikitext.smerity.com/wikitext-103-v1.zip"

DOC = '''"""@TITLE@ training on WikiText-103, batch @BATCH@, @STEPS@ steps, fp16 — Lambda GPU benchmark.

Neutral form of the task: a plain script, runnable on any machine with a GPU
(`python tasks/lambdalabs/@NAME@.py`; needs torch, numpy). Two phases:
prepare_data() downloads WikiText-103 into ./data when missing, @FUNC@() is
the measured training. KWARGS are the arguments of @FUNC@. The expected
forecast and its sources are in @NAME@.json; the form a service takes is in
variants/<service>/lambdalabs/@NAME@.py.

Source (the measured workload):
- Benchmark: github.com/lambdal/deeplearning-benchmark, commit aa5addb6
  (2025-12-17), pytorch/scripts/config_v1/@CFG@,
  PyTorch_@CFG_NAME@_FP16_PARAMS: train.py --data
  /data/transformer-xl/wikitext-103 --max_step @STEPS@ --batch_size @BATCH@ --dataset
  wt103 --n_layer @N_LAYER@ --d_model @D_MODEL@ --n_head @N_HEAD@ --d_head 64 --d_inner @D_INNER@
  --dropout @DROPOUT@ --dropatt @DROPATT@
  @FLAGS@
  (scripts/benchmark_pytorch.sh: torchrun --nproc_per_node=1 train.py ...).
- Code it runs: github.com/LambdaLabsML/DeepLearningExamples, branch
  lambda/benchmark, commit 667536cc (2025-12-10),
  PyTorch/LanguageModeling/Transformer-XL (Apache 2.0, NVIDIA Corporation;
  its NOTICE is reproduced below).

The model, the optimizer, the vocabulary and the data iterator below are
copied verbatim from that commit: every section is headed by its file and
lines, and the licence headers of the files are kept. Nothing inside copied
code is changed; `utils` is a namespace of the copied utils/distributed.py
functions, and the imports between the copied modules are one block at the
top. tools/build_dle_transformer_xl.py rebuilds this file from a checkout of
the source.

prepare_data() is getdata.sh:48-57 of the source: the WikiText-103 archive,
unpacked, the three token files renamed to train.txt / valid.txt / test.txt.

@FUNC@() is the run of pytorch/train.py reduced to this command; every choice
follows the source (paths relative to Transformer-XL/pytorch/ at 667536cc):
- seeds 1111 (train.py:228, 772-773); corpus: word vocabulary over train.txt
  with <eos>, case kept, the three files encoded (data_utils.py:309-322);
  training iterator over the token stream on the GPU, batch @BATCH@, target
  length @TGT_LEN@ (train.py:788-789); @ROLL_DOC@;
- model: MemTransformerLM, @N_LAYER@ layers, d_model @D_MODEL@, @N_HEAD@ heads of 64, d_inner @D_INNER@,
  dropout @DROPOUT@, attention dropout @DROPATT@, memory @MEM_LEN@, tied embedding and output
  weights, no adaptive softmax (no --adaptive: cutoffs empty), the other
  parser defaults (train.py:134-175, 798-836); weights initialised by
  weights_init (train.py:837-840);
- optimizer: @OPTIM@, lr 0.0, weight decay 0.0;
- step: gradients dropped, forward and loss under autocast, backward through
  the gradient scaler, unscale, gradient-norm clip at 0.25, scaler step and
  update; the warm-up sets the rate to lr * step / @WARMUP_STEP@ = 0.0
  (train.py:475-500, 527-572);
- @STEPS@ steps, then the run ends (train.py:690-693); no evaluation (--no_eval);
- the reported throughput: target tokens / elapsed time per 10 steps
  (--log_interval), averaged after the first @METER_WARMUP@ such values
  (train.py:590-598, 1018-1019; "Training throughput" in the log,
  compile_results_pytorch.py).

Differences from the source:
- fp16 through torch.cuda.amp (the --amp pytorch branch of the same train.py:
  lines 484-486, 493-494, 549-551, 557-559, 891-892) instead of the default
  APEX AMP at level O2, which exists only in NVIDIA's container. Both compute
  the model in half precision, but they are not the same code path: this is
  the one difference that can change the measured work.
- none of the following changes it: one process on one GPU instead of
  torchrun with one process; the corpus is built from the text files on every
  run instead of being cached in cache.pt (data_utils.py:304-325; the first
  run of the source does the same); the cosine schedule is not created (it is
  never stepped: @STEPS@ steps < warm-up @WARMUP_STEP@); no logging, checkpoint or
  work directory; the validation and test iterators are not created
  (--no_eval); the archive is downloaded from wikitext.smerity.com
  (the URL of getdata.sh no longer serves it).
"""
'''

VARIANT_DOC = '''"""Krauncher variant of tasks/lambdalabs/@NAME@.py (the neutral form, with the
source and the reduction it documents); the expected forecast and its
sources are in tasks/lambdalabs/@NAME@.json.

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
    dict(name="wikitext-103-v1", urls=["@ARCHIVE@"], size_gb=0.2),
]

KWARGS = dict(@KWARGS@)
'''

UNPACK = '''\
    import zipfile

    if not os.path.isdir("wikitext-103"):
        with zipfile.ZipFile("/data/wikitext-103-v1.zip") as z:
            z.extractall(".")
        for split in ("train", "valid", "test"):
            os.rename(os.path.join("wikitext-103", "wiki.%s.tokens" % split),
                      os.path.join("wikitext-103", "%s.txt" % split))

'''

PREPARE = '''\
def prepare_data():
    import os
    import urllib.request
    import zipfile

    root = "@ROOT@"
    if not os.path.isdir(os.path.join(root, "wikitext-103")):
        os.makedirs(root, exist_ok=True)
        archive = os.path.join(root, "wikitext-103-v1.zip")
        urllib.request.urlretrieve("@ARCHIVE@", archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(root)
        for split in ("train", "valid", "test"):
            os.rename(os.path.join(root, "wikitext-103", "wiki.%s.tokens" % split),
                      os.path.join(root, "wikitext-103", "%s.txt" % split))
    return sorted(os.listdir(os.path.join(root, "wikitext-103")))
'''

# The run of train.py for this command; the source lines are in the docstring.
BODY = '''\
    datadir = os.path.join("@ROOT@", "wikitext-103")
    torch.cuda.set_device(0)
    device = torch.device('cuda')
    np.random.seed(1111)
    torch.manual_seed(1111)

    corpus = Corpus(datadir, 'wt103', 'word', special=['<eos>'], lower_case=False)
    ntokens = len(corpus.vocab)
    tr_iter = corpus.get_iterator('train', batch_size, @TGT_LEN@, device=device, ext_len=0)

    model = MemTransformerLM(
        n_token=ntokens, n_layer=@N_LAYER@, n_head=@N_HEAD@, d_model=@D_MODEL@, d_head=64, d_inner=@D_INNER@,
        dropout=@DROPOUT@, dropatt=@DROPATT@, dtype=None, tie_weight=True, d_embed=@D_MODEL@, div_val=1,
        tie_projs=[False], pre_lnorm=False, tgt_len=@TGT_LEN@, ext_len=0, mem_len=@MEM_LEN@, cutoffs=[],
        same_length=False, attn_type=0, clamp_len=-1, sample_softmax=-1)
    init_args = argparse.Namespace(init='normal', init_range=0.1, init_std=0.02, proj_init_std=0.01)
    model.apply(functools.partial(weights_init, args=init_args))
    model.word_emb.apply(functools.partial(weights_init, args=init_args))

    optimizer = @OPTIM_CODE@
    model = model.to(device)
    scaler = torch.cuda.amp.GradScaler()

    mems = None
    train_step = 0
    train_throughput = AverageMeter(warmup=@MEM_LEN@ // @TGT_LEN@ + 2)

    for epoch in itertools.count(start=1):
@ROLL@        model.train()
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
            if train_step < @WARMUP_STEP@:
                optimizer.param_groups[0]['lr'] = 0.0 * train_step / @WARMUP_STEP@

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
'''


def build(dle: Path, model: str, batch: int, steps: int, cfg: str) -> str:
    m = MODELS[model]
    name = f"transformer_xl_{model}_wt103_bs{batch}"
    func = "train_transformer_xl"
    kw = {k.upper(): v for k, v in m.items()}
    kw.update(NAME=name, FUNC=func, BATCH=batch, STEPS=steps, CFG=cfg, ARCHIVE=ARCHIVE,
              METER_WARMUP=m["mem_len"] // m["tgt_len"] + 2, KWARGS=f"batch_size={batch}, max_step={steps}")
    # lamb.py is the optimizer of the base model only
    parts = [p for p in PARTS if model == "base" or p[0] != P + "lamb.py"]
    code = verbatim(dle, parts, {}, SRC)
    sig = f"def {func}(batch_size: int = {batch}, max_step: int = {steps}):\n"
    neutral = (fill(DOC, **kw) + "\n" + IMPORTS
               + f'\nKWARGS = {{"batch_size": {batch}, "max_step": {steps}}}\n\n\n'
               + code + "\n\n" + fill(PREPARE, ROOT="data", **kw) + "\n\n" + sig + fill(BODY, ROOT="data", **kw)
               + f'\n\nif __name__ == "__main__":\n    print(prepare_data())\n    print({func}(**KWARGS))\n')
    variant = (fill(VARIANT_DOC, **kw) + "\n\n" + sig
               + textwrap.indent(IMPORTS + "\n" + code, "    ", lambda line: bool(line.strip()))
               + "\n" + UNPACK + fill(BODY, ROOT=".", **kw)
               + f"\n\nFUNC = {func}\n")
    (ROOT / "tasks" / "lambdalabs" / f"{name}.py").write_text(neutral)
    (ROOT / "variants" / "krauncher" / "lambdalabs" / f"{name}.py").write_text(variant)
    return name


if __name__ == "__main__":
    for sample in SAMPLES:
        print("built", build(Path(sys.argv[1]), *sample))
