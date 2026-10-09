"""BERT-large fine-tuning on SQuAD v1.1, mixed precision, batch 28 — Lambda GPU benchmark.

Neutral form of the task: a plain script, runnable on any machine with a GPU
(`python tasks/lambdalabs/bert_large_squad_amp_bs28.py`); model and dataset by
their public Hugging Face ids. KWARGS are the call arguments. The expected
forecast and its sources are in bert_large_squad_amp_bs28.json; the form a
service takes is in variants/<service>/lambdalabs/bert_large_squad_amp_bs28.py.

Source (the measured workload):
- Benchmark: github.com/lambdal/deeplearning-benchmark, commit aa5addb6
  (2025-12-17), pytorch/scripts/config_v1/config_pytorch_24GB.sh,
  PyTorch_bert_large_squad_FP16_PARAMS, passed to scripts/run_squad.sh as
  init_checkpoint=bert_large_uncased.pt epochs=2.0 batch_size=28 lr=0.0
  warmup=0.1 precision=fp16 num_gpu=1 seed=1 squad=v1.1 mode=train
  config=bert_config.json max_steps=100 (scripts/benchmark_pytorch.sh,
  benchmark_pytorch_bert_squad).
- Code it runs: github.com/LambdaLabsML/DeepLearningExamples, branch
  lambda/benchmark, commit 667536cc (2025-12-10), PyTorch/LanguageModeling/BERT:
  scripts/run_squad.sh -> run_squad.py --do_train --train_batch_size=28
  --max_seq_length=384 --doc_stride=128 --num_train_epochs=2.0 --max_steps=100
  --fp16 (run_squad.sh:18-31, 80-92).

This file is the smallest faithful reduction of that run to one file; every
choice follows the source (paths relative to BERT/ at 667536cc):
- model: BertForQuestionAnswering on the BERT-large config (24 layers,
  hidden 1024), initialised from the pretrained uncased checkpoint (run_squad.py:942-948);
- data: SQuAD v1.1 train, question + context cut into features of 384 tokens
  with a 128-token stride, start / end token of the answer per feature
  (run_squad.py:209-332, convert_examples_to_features); random sampling
  (RandomSampler), batch 28;
- step: forward, loss = mean of the start and end cross-entropies with
  positions clamped to the sequence (run_squad.py:1066-1079), backward under
  a dynamic loss scale, gradient clipping at norm 1.0, optimizer step,
  zero_grad, loss.item() every step (run_squad.py:1080-1103);
- optimizer: Adam-type with weight decay 0.01 except bias and LayerNorm
  (run_squad.py:960-973);
- length: the loop stops when global_step > max_steps, so max_steps + 1 =
  101 optimizer steps (run_squad.py:1060-1061); learning rate 0.0 as configured.

Differences from the source, and why the step's compute stays the same:
native mixed precision (torch.autocast + GradScaler, dynamic) instead of apex
amp O2 — the matrix products run in fp16 either way, but O2 keeps the model
in fp16 with an fp32 master copy while autocast keeps fp32 weights and casts
per op: peak memory differs somewhat, compute does not; torch.optim.AdamW
(fused) instead of apex FusedAdam with bias_correction=False (same element-
wise work); Hugging Face's BERT and tokenizer instead of DLE's modeling.py /
tokenization.py (same architecture and vocabulary); tokenisation by the
standard Hugging Face question-answering preprocessing, which yields the
same kind of 384 / 128 features.
"""

KWARGS = {"batch_size": 28, "max_steps": 100, "max_seq_length": 384, "doc_stride": 128}


def finetune_bert_large_squad(batch_size: int = 28, max_steps: int = 100,
                             max_seq_length: int = 384, doc_stride: int = 128):
    import time

    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader, RandomSampler, TensorDataset
    from transformers import AutoTokenizer, BertForQuestionAnswering

    device = torch.device("cuda")
    model_id = "google-bert/bert-large-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = BertForQuestionAnswering.from_pretrained(model_id).to(device)

    ds = load_dataset("rajpurkar/squad", split="train")
    enc = tokenizer(ds["question"], ds["context"], truncation="only_second",
                    max_length=max_seq_length, stride=doc_stride, padding="max_length",
                    return_overflowing_tokens=True, return_offsets_mapping=True)
    starts, ends = [], []
    for i, offsets in enumerate(enc["offset_mapping"]):
        ex = enc["overflow_to_sample_mapping"][i]
        a_start = ds[ex]["answers"]["answer_start"][0]
        a_end = a_start + len(ds[ex]["answers"]["text"][0])
        seq = enc.sequence_ids(i)
        ctx = [k for k, s in enumerate(seq) if s == 1]
        s_tok = e_tok = 0
        if offsets[ctx[0]][0] <= a_start and offsets[ctx[-1]][1] >= a_end:
            s_tok = next(k for k in ctx if offsets[k][1] > a_start)
            e_tok = next(k for k in reversed(ctx) if offsets[k][0] < a_end)
        starts.append(s_tok)
        ends.append(e_tok)
    data = TensorDataset(torch.tensor(enc["input_ids"]), torch.tensor(enc["attention_mask"]),
                         torch.tensor(enc["token_type_ids"]), torch.tensor(starts), torch.tensor(ends))
    loader = DataLoader(data, sampler=RandomSampler(data), batch_size=batch_size)

    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(d in n for d in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(d in n for d in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=0.0, fused=True)
    scaler = torch.cuda.amp.GradScaler()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=max_seq_length)

    model.train()
    step, t0 = 0, time.time()
    while step <= max_steps:
        for input_ids, mask, seg, s_pos, e_pos in loader:
            if step > max_steps:
                break
            input_ids, mask, seg = input_ids.to(device), mask.to(device), seg.to(device)
            s_pos, e_pos = s_pos.to(device).clamp_(0, max_seq_length), e_pos.to(device).clamp_(0, max_seq_length)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(input_ids=input_ids, attention_mask=mask, token_type_ids=seg)
                loss = (loss_fct(out.start_logits, s_pos) + loss_fct(out.end_logits, e_pos)) / 2
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            loss_value = loss.item()
            step += 1
    elapsed = time.time() - t0
    return {"features": len(data), "steps": step, "elapsed_sec": round(elapsed, 2),
            "sequences_per_sec": round(step * batch_size / elapsed, 1), "last_loss": round(loss_value, 3)}


if __name__ == "__main__":
    print(finetune_bert_large_squad(**KWARGS))
