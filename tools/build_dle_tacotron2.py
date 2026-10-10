"""Build the lambdalabs/tacotron2_* samples from NVIDIA DeepLearningExamples.

Same procedure as build_dle_waveglow.py (see there): the model, loss, data,
text and audio code is copied verbatim from the pinned commit, with the
licence headers; the changes inside copied code are in EDITS and the text
written by this builder between the sections in the "glue" parts.

    python tools/build_dle_tacotron2.py dle     # rewrites the task files
    git status --short                          # nothing changed
"""

import sys
from pathlib import Path

from build_dle_waveglow import compose, verbatim

T = "tacotron2/text/"
PARTS = [
    ("licence", T + "LICENCE"),
    (T + "cmudict.py", ["valid_symbols"]),
    (T + "symbols.py", None),
    (T + "numbers.py", None),
    ("glue", """\
# Transliteration tables: tacotron2/text/unidecoder/replacements.py and
# homoglyphs.py (2 200 lines) are reduced to the one non-ASCII character of
# the two file lists, 'ü' (replacements.py:50, 115).
_replacements = {'ü': 'ue'}
_homoglyphs = {}
"""),
    (T + "unidecoder/__init__.py", ["unidecoder"]),
    (T + "cleaners.py", None),
    ("glue", """\
# The module `cleaners` that text/__init__.py looks the cleaner up in.
cleaners = types.SimpleNamespace(
    basic_cleaners=basic_cleaners, transliteration_cleaners=transliteration_cleaners,
    english_cleaners=english_cleaners)
"""),
    (T + "__init__.py", None),
    ("tacotron2_common/audio_processing.py", ["window_sumsquare", "dynamic_range_compression", "dynamic_range_decompression"]),
    ("tacotron2_common/stft.py", ["STFT"]),
    ("tacotron2_common/layers.py", ["LinearNorm", "ConvNorm", "TacotronSTFT"]),
    ("tacotron2_common/utils.py", ["get_mask_from_lengths", "load_wav_to_torch", "load_filepaths_and_text", "to_gpu"]),
    ("tacotron2/model.py", ["LocationLayer", "Attention", "Prenet", "Postnet", "Encoder", "Decoder", "Tacotron2"]),
    ("tacotron2/loss_function.py", ["Tacotron2Loss"]),
    ("tacotron2/data_function.py", ["TextMelLoader", "TextMelCollate", "batch_to_gpu"]),
    ("models.py", ["init_bn"]),
]
# Every change inside copied code: prefixes of modules that are now this file.
EDITS = {
    T + "symbols.py": [("cmudict.valid_symbols", "valid_symbols")],
    "tacotron2/data_function.py": [("layers.TacotronSTFT(", "TacotronSTFT(")],
}

# Samples: (name, batch size, epochs, GPU config of the benchmark).
SAMPLES = [
    ("tacotron2_ljs625_bs256", 256, 3, "config_pytorch_80GB.sh"),
    ("tacotron2_ljs625_bs148", 148, 1, "config_pytorch_48GB.sh"),
    ("tacotron2_ljs625_bs88", 88, 2, "config_pytorch_24GB.sh"),
]

IMPORTS = """\
import argparse
import os
import re
import time
import types
import warnings
from math import sqrt

import inflect
import numpy as np
import librosa.util as librosa_util
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
from librosa.util import pad_center, tiny
from scipy.io.wavfile import read
from scipy.signal import get_window
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader

torch._C._jit_set_autocast_mode(False)  # waveglow/model.py:28, imported through models.py:33
"""

DOC = '''"""Tacotron 2 training on LJSpeech (625 clips), batch @BATCH@, @EPOCHS@ — Lambda GPU benchmark.

Neutral form of the task: a plain script, runnable on any machine with a GPU
(`python tasks/lambdalabs/@NAME@.py`; needs torch, librosa, scipy, inflect);
LJSpeech-1.1 and the benchmark's file lists are downloaded from their public
URLs into ./data when missing. KWARGS are the call arguments. The expected
forecast and its sources are in @NAME@.json; the form a service takes is in
variants/<service>/lambdalabs/@NAME@.py.

Source (the measured workload):
- Benchmark: github.com/lambdal/deeplearning-benchmark, commit aa5addb6
  (2025-12-17), pytorch/scripts/config_v1/@CFG@,
  PyTorch_tacotron2_FP16_PARAMS: train.py -o ./ --model-name Tacotron2
  --learning-rate 0.0 --epochs @EPOCHS_N@ --batch-size @BATCH@ --weight-decay 1e-6
  --grad-clip-thresh 1.0 --training-files
  filelists/ljs_audio_text_train_subset_625_filelist.txt --cudnn-enabled
  --amp-run (scripts/benchmark_pytorch.sh, benchmark_pytorch_tacotron2:
  python -m multiproc 1 train.py ...).
- Code it runs: github.com/LambdaLabsML/DeepLearningExamples, branch
  lambda/benchmark, commit 667536cc (2025-12-10),
  PyTorch/SpeechSynthesis/Tacotron2 (BSD 3-Clause, NVIDIA Corporation; the
  text front end MIT, Keith Ito; the transliteration Apache 2.0, NVIDIA).
- Precision: the run computes in fp32. train.py's flag is --amp; --amp-run is
  not one of its arguments and parse_known_args drops it (train.py:86, 349,
  382), so autocast and the gradient scaler stay disabled. The published
  numbers agree: at the same batch the tacotron2 throughput of
  pytorch-train-throughput-fp16.csv is 0.94-1.08 of the fp32 one, against
  1.5-2.4 for resnet50.

The model, the loss, the dataset, the text front end and the audio code
below are copied verbatim from that commit: every section is headed by its
file and lines, and the licence headers of the files are kept. Changes inside
copied code: the `layers.` prefix of TacotronSTFT in TextMelLoader and the
`cmudict.` prefix of valid_symbols (the modules are this file). Not copied:
the transliteration tables (2 200 lines) — the two file lists hold one
non-ASCII character, 'ü', and its replacement is given as in the tables; the
`cleaners` module is a namespace of the copied cleaner functions. The imports
between the copied modules are one block at the top.
tools/build_dle_tacotron2.py rebuilds this file from a checkout of the source.

@FUNC@() is the run of train.py reduced to this command; every choice follows
the source (paths relative to Tacotron2/ at 667536cc):
- cudnn enabled (--cudnn-enabled; no --cudnn-benchmark: train.py:384-385);
- model: Tacotron2 with the parser defaults (tacotron2/arg_parser.py:40-106,
  models.py:98-130), 148 symbols; batch-norm weights initialised uniformly,
  model on the GPU (models.py:88-92, train.py:92);
- optimizer: Adam, lr 0.0, weight decay 1e-6 (train.py:401-402); loss:
  Tacotron2Loss (train.py:422, loss_functions.py:35-36);
- data: TextMelLoader over the 625-clip file list — text through
  english_cleaners to symbol ids, the mel spectrogram of the whole clip, audio
  defaults of train.py:111-138; TextMelCollate pads a batch to its longest
  text and clip (data_functions.py:36-37); DataLoader with 8 workers,
  shuffled, the last partial batch dropped (train.py:431-443): @STEPS@ steps
  per epoch at batch @BATCH@;
- step: zero_grad, batch to the GPU, forward and loss under
  autocast(enabled=False), NaN check, backward, gradient-norm clip at 1.0,
  optimizer step, zero_grad(set_to_none=True) (train.py:480-517);
- validation after every epoch on ljs_audio_text_val_filelist.txt (100 clips,
  the default of --validation-files): eval mode, no_grad, 1 worker, the same
  batch size, last partial batch kept (train.py:116-118, 273-321, 538-542);
- the reported throughput is the mean of per-step mel frames / step time
  over the last epoch, the step timed between cuda.synchronize calls after
  the batch is loaded (train.py:472-473, 519-523, 533-534;
  compile_results_pytorch_v2.py takes the next-to-last train_items_per_sec).

Differences from the source, none of which changes the measured work: one
process on one GPU instead of `multiproc 1` (world_size 1, no distributed
sampler: train.py:358, 433-438); no DLLogger output; the checkpoint written
after the first epoch (train.py:545-547; outside the timed steps) is not
saved; the learning-rate schedule call is left out (no --anneal-steps: the
rate stays 0.0, train.py:324-342); the unused gradient scaler is not created;
the dataset is unpacked under ./data instead of /data/tacotron2/LJSpeech-1.1.
"""
'''

# The run of train.py for this command; the source lines are in the docstring.
BODY = '''\
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    amp = False  # --amp-run is not an argument of train.py; args.amp stays False

    args = argparse.Namespace(
        text_cleaners=['english_cleaners'], load_mel_from_disk=False,
        max_wav_value=32768.0, sampling_rate=22050, filter_length=1024, hop_length=256,
        win_length=1024, mel_fmin=0.0, mel_fmax=8000.0, n_mel_channels=80)

    model = Tacotron2(
        mask_padding=False, n_mel_channels=80, n_symbols=len(symbols), symbols_embedding_dim=512,
        encoder_kernel_size=5, encoder_n_convolutions=3, encoder_embedding_dim=512,
        attention_rnn_dim=1024, attention_dim=128, attention_location_n_filters=32,
        attention_location_kernel_size=31, n_frames_per_step=1, decoder_rnn_dim=1024,
        prenet_dim=256, max_decoder_steps=2000, gate_threshold=0.5, p_attention_dropout=0.1,
        p_decoder_dropout=0.1, postnet_embedding_dim=512, postnet_kernel_size=5,
        postnet_n_convolutions=5, decoder_no_early_stopping=False)
    init_bn(model)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, weight_decay=1e-6)
    criterion = Tacotron2Loss()
    criterion.cuda()

    collate_fn = TextMelCollate(1)
    trainset = TextMelLoader(root, lists["@TRAIN_LIST@"], args)
    valset = TextMelLoader(root, lists["@VAL_LIST@"], args)
    train_loader = DataLoader(trainset, num_workers=8, shuffle=True, sampler=None,
                              batch_size=batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn, **workers)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                                    collate_fn=collate_fn, drop_last=False, **workers)
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

if __name__ == "__main__":
    code = verbatim(Path(sys.argv[1]), PARTS, EDITS)
    for name, batch, epochs, cfg in SAMPLES:
        compose(name, "train_tacotron2", {"batch_size": batch, "epochs": epochs}, code, IMPORTS, DOC, BODY,
                BATCH=batch, CFG=cfg, STEPS=625 // batch, EPOCHS_N=epochs,
                EPOCHS=f"{epochs} epoch" + ("s" if epochs > 1 else ""),
                PIP='"librosa", "inflect"', BUILDER="build_dle_tacotron2.py")
        print("built", name)
