"""Build the lambdalabs/waveglow_* samples from NVIDIA DeepLearningExamples.

The model, loss, data and audio code of these tasks is copied verbatim from a
checkout of github.com/LambdaLabsML/DeepLearningExamples (branch
lambda/benchmark) at COMMIT, with the licence headers of the copied files.
The only changes inside copied code are listed in EDITS; the import lines
between the copied modules are replaced by one import block (everything
lives in one file). The Krauncher variant holds the same text indented into
the task function. Run it to check that the committed task files are what
the source says:

    git clone https://github.com/LambdaLabsML/DeepLearningExamples.git dle
    git -C dle checkout 667536cc8138edbd236aed08dbafc8e77bcc7c20
    python tools/build_dle_waveglow.py dle     # rewrites the task files
    git status --short                         # nothing changed
"""

import ast
import sys
import textwrap
from pathlib import Path

COMMIT = "667536cc8138edbd236aed08dbafc8e77bcc7c20"
SRC = "PyTorch/SpeechSynthesis/Tacotron2"
ROOT = Path(__file__).resolve().parent.parent

# (file, top-level definitions copied verbatim), in dependency order.
PARTS = [
    ("tacotron2_common/audio_processing.py", ["window_sumsquare", "dynamic_range_compression", "dynamic_range_decompression"]),
    ("tacotron2_common/stft.py", ["STFT"]),
    ("tacotron2_common/layers.py", ["TacotronSTFT"]),
    ("tacotron2_common/utils.py", ["load_wav_to_torch", "load_filepaths_and_text", "to_gpu"]),
    ("waveglow/model.py", ["fused_add_tanh_sigmoid_multiply", "Invertible1x1Conv", "WN", "WaveGlow", "remove"]),
    ("waveglow/loss_function.py", ["WaveGlowLoss"]),
    ("waveglow/data_function.py", ["MelAudioLoader", "batch_to_gpu"]),
    ("models.py", ["init_bn"]),
]
# Every change inside copied code: a prefix of a module that is now this file.
EDITS = {"waveglow/data_function.py": [("layers.TacotronSTFT(", "TacotronSTFT(")]}

# Samples: (name, batch size, GPU config of the benchmark, cards in the title).
SAMPLES = [
    ("waveglow_ljs625_bs48", 48, "config_pytorch_80GB.sh", "A100 / H100 80 GB"),
    ("waveglow_ljs625_bs32", 32, "config_pytorch_48GB.sh", "RTX 6000 Ada, RTX A6000, Quadro RTX 8000"),
    ("waveglow_ljs625_bs18", 18, "config_pytorch_24GB.sh", "RTX 4090, RTX 3090, A10"),
]

IMPORTS = """\
import argparse
import os
import time

import numpy as np
import librosa.util as librosa_util
import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
from librosa.util import pad_center, tiny
from scipy.io.wavfile import read
from scipy.signal import get_window
from torch.autograd import Variable
from torch.utils.data import DataLoader

torch._C._jit_set_autocast_mode(False)  # waveglow/model.py:28
"""

FILELISTS = ("https://raw.githubusercontent.com/LambdaLabsML/DeepLearningExamples/" + COMMIT
             + "/" + SRC + "/filelists/")
ARCHIVE = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
TRAIN_LIST = "ljs_audio_text_train_subset_625_filelist.txt"
VAL_LIST = "ljs_audio_text_val_filelist.txt"

DOC = '''"""WaveGlow training on LJSpeech (625 clips), batch @BATCH@, 2 epochs — Lambda GPU benchmark.

Neutral form of the task: a plain script, runnable on any machine with a GPU
(`python tasks/lambdalabs/@NAME@.py`; needs torch, librosa, scipy); LJSpeech-1.1
and the benchmark's file lists are downloaded from their public URLs into
./data when missing. KWARGS are the call arguments. The expected forecast and
its sources are in @NAME@.json; the form a service takes is in
variants/<service>/lambdalabs/@NAME@.py.

Source (the measured workload):
- Benchmark: github.com/lambdal/deeplearning-benchmark, commit aa5addb6
  (2025-12-17), pytorch/scripts/config_v1/@CFG@,
  PyTorch_waveglow_FP16_PARAMS: train.py -o ./ --model-name WaveGlow
  --learning-rate 0.0 --epochs 2 --segment-length 8000 --batch-size @BATCH@
  --weight-decay 0 --grad-clip-thresh 65504 --training-files
  filelists/ljs_audio_text_train_subset_625_filelist.txt --cudnn-enabled
  --cudnn-benchmark --amp-run (scripts/benchmark_pytorch.sh,
  benchmark_pytorch_tacotron2: python -m multiproc 1 train.py ...).
- Code it runs: github.com/LambdaLabsML/DeepLearningExamples, branch
  lambda/benchmark, commit 667536cc (2025-12-10),
  PyTorch/SpeechSynthesis/Tacotron2 (BSD 3-Clause, NVIDIA Corporation).
- Precision: the run computes in fp32. train.py's flag is --amp; --amp-run is
  not one of its arguments and parse_known_args drops it (train.py:86, 349,
  382), so autocast and the gradient scaler stay disabled. The published
  numbers agree: on the cards where the fp16 and fp32 CSVs used the same
  batch (the 24 and 48 GB cards) the waveglow throughput of
  pytorch-train-throughput-fp16.csv is 0.99-1.01 of the fp32 one, against
  1.5-2.4 for resnet50.

The model, the loss, the dataset and the audio code below are copied verbatim
from that commit: every section is headed by its file and lines, and the
licence headers of the files are kept. Changes inside copied code: the
`layers.` prefix of TacotronSTFT in MelAudioLoader (the module is this file);
the imports between the copied modules are one block at the top.
tools/build_dle_waveglow.py rebuilds this file from a checkout of the source.

@FUNC@() is the run of train.py reduced to this command; every choice follows
the source (paths relative to Tacotron2/ at 667536cc):
- cudnn enabled and in benchmark mode (train.py:384-385);
- model: WaveGlow with the parser defaults — 80 mel channels, 12 flows,
  groups of 8, early outputs of 2 every 4 flows, WN of 8 layers, kernel 3,
  512 channels (waveglow/arg_parser.py:37-64, models.py:132-144); batch-norm
  weights initialised uniformly, model on the GPU (models.py:88-92,
  train.py:92);
- optimizer: Adam, lr 0.0, weight decay 0 (train.py:401-402); loss:
  WaveGlowLoss, sigma 1.0 (train.py:406-422, loss_functions.py:37-43);
- data: MelAudioLoader over the 625-clip file list, a random 8000-sample
  segment per clip and its mel spectrogram, audio defaults of train.py:125-138;
  DataLoader with 8 workers, shuffled, the default collate, the last partial
  batch dropped (train.py:431-443, data_functions.py:39-40): @STEPS@ steps per epoch at batch @BATCH@;
- step: zero_grad, batch to the GPU, forward and loss under
  autocast(enabled=False), NaN check, backward, gradient-norm clip at 65504,
  optimizer step, zero_grad(set_to_none=True) (train.py:480-517);
- validation after every epoch on ljs_audio_text_val_filelist.txt (100 clips,
  the default of --validation-files): eval mode, no_grad, 1 worker, the same
  batch size, last partial batch kept (train.py:116-118, 273-321, 538-542);
- the reported throughput is the mean of per-step audio samples / step time
  over the last epoch, the step timed between cuda.synchronize calls after
  the batch is loaded (train.py:472-473, 519-523, 533-534;
  compile_results_pytorch_v2.py takes the next-to-last train_items_per_sec).

Differences from the source, none of which changes the measured work: one
process on one GPU instead of `multiproc 1` (world_size 1, no distributed
sampler: train.py:358, 433-438); no DLLogger output; the checkpoint written
after the first epoch (train.py:545-547; outside the timed steps) is not saved; the learning-rate
schedule call is left out (no --anneal-steps: the rate stays 0.0,
train.py:324-342); the unused gradient scaler is not created; the dataset is
unpacked under ./data instead of /data/tacotron2/LJSpeech-1.1.
"""
'''

VARIANT_DOC = '''"""Krauncher variant of tasks/lambdalabs/@NAME@.py (the neutral form, with the
source and the reduction it documents); the expected forecast and its
sources are in tasks/lambdalabs/@NAME@.json.

The task function holds the neutral file's code — the imports, the sections
copied verbatim from NVIDIA DeepLearningExamples (headed by file and lines,
licence headers kept) and the run — indented into one self-contained
function. Differences from the neutral form: the LJSpeech-1.1 archive and the
two file lists come through a registered data source (/data) instead of a
download; the DataLoader workers are forked (the dataset class is local to
the function and cannot be pickled). FUNC is the task, OPTIONS the arguments
of its @client.task decorator, KWARGS the arguments of the call.
tools/build_dle_waveglow.py rebuilds this file from a checkout of the source.
"""

OPTIONS = dict(
    timeout=3600,
    data="ljspeech-11-dle",
    pip=["librosa"],
)

# Data source the task reads (OPTIONS["data"]), registered on the account when
# missing: LJSpeech-1.1 (public domain, keithito.com) and the two file lists
# of the benchmark's code at the pinned commit.
DATA_SOURCES = [
    dict(name="ljspeech-11-dle", urls=[
        "@ARCHIVE@",
        "@FILELISTS@@TRAIN_LIST@",
        "@FILELISTS@@VAL_LIST@",
    ], size_gb=2.6),
]

KWARGS = dict(batch_size=@BATCH@, epochs=2, segment_length=8000)
'''

DATA_NEUTRAL = '''\
    import tarfile
    import urllib.request

    root = "data"
    os.makedirs(root, exist_ok=True)
    lists = {}
    for name in ("@TRAIN_LIST@", "@VAL_LIST@"):
        lists[name] = os.path.join(root, name)
        if not os.path.isfile(lists[name]):
            urllib.request.urlretrieve("@FILELISTS@" + name, lists[name])
    if not os.path.isdir(os.path.join(root, "LJSpeech-1.1")):
        archive = os.path.join(root, "LJSpeech-1.1.tar.bz2")
        if not os.path.isfile(archive):
            urllib.request.urlretrieve("@ARCHIVE@", archive)
        with tarfile.open(archive) as t:
            t.extractall(root)
    workers = {}
'''

DATA_KRAUNCHER = '''\
    import tarfile

    root = "/data"
    lists = {name: os.path.join(root, name) for name in ("@TRAIN_LIST@", "@VAL_LIST@")}
    if not os.path.isdir(os.path.join(root, "LJSpeech-1.1")):
        with tarfile.open(os.path.join(root, "LJSpeech-1.1.tar.bz2")) as t:
            t.extractall(root)
    workers = {"multiprocessing_context": "fork"}
'''

# The run of train.py for this command; the source lines are in the docstring.
BODY = '''\
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    amp = False  # --amp-run is not an argument of train.py; args.amp stays False

    args = argparse.Namespace(
        max_wav_value=32768.0, sampling_rate=22050, filter_length=1024, hop_length=256,
        win_length=1024, mel_fmin=0.0, mel_fmax=8000.0, n_mel_channels=80,
        segment_length=segment_length)

    model = WaveGlow(n_mel_channels=80, n_flows=12, n_group=8, n_early_every=4, n_early_size=2,
                     WN_config=dict(n_layers=8, kernel_size=3, n_channels=512))
    init_bn(model)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, weight_decay=0.0)
    criterion = WaveGlowLoss(sigma=1.0)
    criterion.cuda()

    trainset = MelAudioLoader(root, lists["@TRAIN_LIST@"], args)
    valset = MelAudioLoader(root, lists["@VAL_LIST@"], args)
    train_loader = DataLoader(trainset, num_workers=8, shuffle=True, sampler=None,
                              batch_size=batch_size, pin_memory=False,
                              drop_last=True, **workers)

    val_loss = 0.0
    train_epoch_items_per_sec = 0.0
    num_iters = 0
    model.train()

    for epoch in range(epochs):
        train_epoch_items_per_sec = 0.0
        num_iters = 0

        for i, batch in enumerate(train_loader):
            torch.cuda.synchronize()
            iter_start_time = time.perf_counter()

            model.zero_grad()
            x, y, num_items = batch_to_gpu(batch)

            with torch.cuda.amp.autocast(enabled=amp):
                y_pred = model(x)
                loss = criterion(y_pred, y)

            reduced_loss = loss.item()
            reduced_num_items = num_items.item()
            if np.isnan(reduced_loss):
                raise Exception("loss is NaN")

            num_iters += 1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 65504.0)
            optimizer.step()

            model.zero_grad(set_to_none=True)

            torch.cuda.synchronize()
            iter_stop_time = time.perf_counter()
            iter_time = iter_stop_time - iter_start_time
            train_epoch_items_per_sec += reduced_num_items / iter_time

        # validate(): train.py:273-321
        model.eval()
        with torch.no_grad():
            val_loader = DataLoader(valset, num_workers=1, shuffle=False, sampler=None,
                                    batch_size=batch_size, pin_memory=False,
                                    drop_last=False, **workers)
            val_loss = 0.0
            val_iters = 0
            for i, batch in enumerate(val_loader):
                x, y, num_items = batch_to_gpu(batch)
                with torch.cuda.amp.autocast(enabled=amp):
                    y_pred = model(x)
                    loss = criterion(y_pred, y)
                val_loss += loss.item()
                val_iters += 1
            val_loss = val_loss / val_iters
        model.train()

    return {"train_items_per_sec": train_epoch_items_per_sec / num_iters if num_iters > 0 else 0.0,
            "val_loss": val_loss}
'''


def verbatim(dle: Path) -> str:
    """The copied sections, each headed by file and lines; a licence header
    is emitted once, before the first section of a file that carries it."""
    out, seen = [], set()
    for rel, names in PARTS:
        text = (dle / SRC / rel).read_text()
        lines = text.splitlines()
        tree = ast.parse(text)
        first = next(n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)))
        header = "\n".join(lines[:first.lineno - 1]).strip()
        if " ".join(header.split()) not in seen:
            seen.add(" ".join(header.split()))
            out.append(f"# --- licence header: {SRC}/{rel}:1-{first.lineno - 1} ---\n{header}\n")
        defs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for name in names:
            n = defs[name]
            start = min([n.lineno] + [d.lineno for d in n.decorator_list])
            seg = "\n".join(lines[start - 1:n.end_lineno])
            for a, b in EDITS.get(rel, []):
                assert a in seg or name != "MelAudioLoader"
                seg = seg.replace(a, b)
            out.append(f"# --- verbatim: {SRC}/{rel}:{start}-{n.end_lineno} ---\n{seg}\n")
    return "\n\n".join(out)


def fill(text: str, **kw: object) -> str:
    kw.update(ARCHIVE=ARCHIVE, FILELISTS=FILELISTS, TRAIN_LIST=TRAIN_LIST, VAL_LIST=VAL_LIST)
    for k, v in kw.items():
        text = text.replace(f"@{k}@", str(v))
    return text


def build(dle: Path, name: str, batch: int, cfg: str) -> None:
    func = "train_waveglow"
    kw = dict(NAME=name, BATCH=batch, CFG=cfg, FUNC=func, STEPS=625 // batch)
    code = verbatim(dle)
    sig = f"def {func}(batch_size: int = {batch}, epochs: int = 2, segment_length: int = 8000):\n"
    neutral = (fill(DOC, **kw) + "\n" + IMPORTS
               + f'\nKWARGS = {{"batch_size": {batch}, "epochs": 2, "segment_length": 8000}}\n\n\n'
               + code + "\n\n" + sig + fill(DATA_NEUTRAL, **kw) + "\n" + fill(BODY, **kw)
               + f'\n\nif __name__ == "__main__":\n    print({func}(**KWARGS))\n')
    variant = (fill(VARIANT_DOC, **kw) + "\n\n" + sig
               + textwrap.indent(IMPORTS + "\n" + code, "    ", lambda line: bool(line.strip()))
               + "\n" + fill(DATA_KRAUNCHER, **kw) + "\n" + fill(BODY, **kw)
               + f"\n\nFUNC = {func}\n")
    (ROOT / "tasks" / "lambdalabs" / f"{name}.py").write_text(neutral)
    (ROOT / "variants" / "krauncher" / "lambdalabs" / f"{name}.py").write_text(variant)


if __name__ == "__main__":
    for name, batch, cfg, _cards in SAMPLES:
        build(Path(sys.argv[1]), name, batch, cfg)
        print("built", name)
